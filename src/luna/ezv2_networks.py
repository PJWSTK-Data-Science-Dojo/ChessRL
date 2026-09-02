"""EfficientZeroV2 network architecture: representation, dynamics, prediction.

Key design choices:
- Spatial action planes (from/to plus underpromotion identity) instead of a dense action embedding.
  Cuts ~76% of parameters and gives the conv stack spatially meaningful input.
- SimSiam-style projection + prediction heads for the consistency loss.
- GroupNorm everywhere (stable at batch=1 during MCTS inference, unlike BatchNorm).
- Depthwise-separable residual blocks in the dynamics network for faster MCTS rollouts.
- Mean/std latent normalisation for smoother gradients (replaces min-max).
"""

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from luna.action_encoding import (
    ACTION_PLANES as SPATIAL_ACTION_PLANES,
)
from luna.action_encoding import (
    action_index_to_planes as action_index_to_planes,
)
from luna.action_encoding import (
    action_int_to_planes as action_int_to_planes,
)
from luna.config import EzV2LearnerConfig
from luna.game.chess_game import ACTION_SIZE, ChessGame

_NUM_GROUPS = 8


def _num_groups(channels: int) -> int:
    g = _NUM_GROUPS
    while channels % g != 0 and g > 1:
        g //= 2
    return g


class _ResBlock(nn.Module):
    """Standard residual block with GroupNorm."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        g = _num_groups(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(g, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + residual)


class _DepthwiseSepResBlock(nn.Module):
    """Depthwise-separable residual block -- ~8x fewer FLOPs than standard at 64ch."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        g = _num_groups(channels)
        self.dw1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.gn1 = nn.GroupNorm(g, channels)
        self.dw2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.gn2 = nn.GroupNorm(g, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.gn1(self.pw1(self.dw1(x))))
        out = self.gn2(self.pw2(self.dw2(out)))
        return F.relu(out + residual)


def _make_residual_block(channels: int) -> nn.Module:
    return _ResBlock(channels)


def _make_dw_sep_block(channels: int) -> nn.Module:
    return _DepthwiseSepResBlock(channels)


class RepresentationNetwork(nn.Module):
    def __init__(self, obs_planes: int, channels: int, num_blocks: int = 4) -> None:
        super().__init__()
        g = _num_groups(channels)
        self.conv_in = nn.Conv2d(obs_planes, channels, 3, padding=1, bias=False)
        self.gn_in = nn.GroupNorm(g, channels)
        self.blocks = nn.Sequential(*[_make_residual_block(channels) for _ in range(num_blocks)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gn_in(self.conv_in(obs)))
        return cast(torch.Tensor, self.blocks(x))


class DynamicsNetwork(nn.Module):
    """Latent transition model optimized for recurrent search inference."""

    # from-square, to-square, then knight/rook/bishop underpromotion identity.
    # Keeping promotion identity is essential: these actions share from/to squares
    # but lead to different chess positions.
    ACTION_PLANES = SPATIAL_ACTION_PLANES

    def __init__(self, channels: int, support_size: int, num_blocks: int = 2) -> None:
        super().__init__()
        g = _num_groups(channels)
        self.channels = channels
        self.conv_in = nn.Conv2d(channels + self.ACTION_PLANES, channels, 3, padding=1, bias=False)
        self.gn_in = nn.GroupNorm(g, channels)
        self.blocks = nn.Sequential(*[_make_dw_sep_block(channels) for _ in range(num_blocks)])

        reward_bins = 2 * support_size + 1
        g16 = _num_groups(16)
        self.reward_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.GroupNorm(g16, 16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, reward_bins),
        )

    def forward(self, latent: torch.Tensor, action_planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """action_planes: ``(B, 5, 8, 8)`` spatial action encoding."""
        x = torch.cat([latent, action_planes], dim=1)
        x = F.relu(self.gn_in(self.conv_in(x)))
        # Consistency supervision targets the next player's already-canonical
        # representation, so this network learns that orientation directly.
        next_latent = self.blocks(x)
        reward_logits = self.reward_head(next_latent)
        return next_latent, reward_logits


def _flatten_spatial_policy(raw_logits: torch.Tensor) -> torch.Tensor:
    """Map chess-aligned spatial logits to the canonical 4,288-action vector.

    The first 64 output channels identify the destination square while the
    spatial location identifies the source square. The final 24 channels encode
    the three underpromotion pieces crossed with the eight destination files;
    canonical observations always promote from rank seven.
    """
    expected_shape = (88, 8, 8)
    if raw_logits.ndim != 4 or tuple(raw_logits.shape[1:]) != expected_shape:
        raise ValueError(
            "Spatial policy logits must have shape "
            f"(batch, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}), "
            f"got {tuple(raw_logits.shape)}"
        )

    batch_size = raw_logits.shape[0]

    # (destination, source_rank, source_file) -> (source, destination)
    base_logits = raw_logits[:, :64].permute(0, 2, 3, 1).reshape(batch_size, 4096)

    # Channels are (piece, destination_file). Select canonical source rank seven,
    # then reorder to the action layout (piece, source_file, destination_file).
    promotion_logits = raw_logits[:, 64:, 6, :]
    promotion_logits = promotion_logits.reshape(batch_size, 3, 8, 8)
    promotion_logits = promotion_logits.permute(0, 1, 3, 2).reshape(batch_size, 192)
    return torch.cat((base_logits, promotion_logits), dim=1)


class _SpatialPolicyHead(nn.Module):
    """Convolutional policy head whose axes match the chess action geometry."""

    OUTPUT_CHANNELS = 64 + 3 * 8

    def __init__(self, channels: int, action_size: int) -> None:
        super().__init__()
        if action_size != ACTION_SIZE:
            raise ValueError(f"Spatial policy head requires {ACTION_SIZE} actions, got {action_size}")
        hidden_channels = 64
        self.tower = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(hidden_channels), hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, self.OUTPUT_CHANNELS, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return _flatten_spatial_policy(self.tower(latent))


class PredictionNetwork(nn.Module):
    def __init__(self, channels: int, action_size: int, support_size: int) -> None:
        super().__init__()
        self.policy_head = _SpatialPolicyHead(channels, action_size)

        value_bins = 2 * support_size + 1
        g16 = _num_groups(16)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.GroupNorm(g16, 16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, value_bins),
        )

    def forward(
        self, latent: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy_logits = self.policy_head(latent)
        if valid_mask is not None:
            policy_logits = policy_logits.masked_fill(
                valid_mask <= 0,
                torch.finfo(policy_logits.dtype).min,
            )
        value_logits = self.value_head(latent)
        return policy_logits, value_logits


class SimSiamProjector(nn.Module):
    """SimSiam projection + prediction heads for consistency loss (EfficientZero)."""

    def __init__(self, in_dim: int, proj_dim: int = 256, *, pool_spatial: bool = False) -> None:
        super().__init__()
        self.pool_spatial = pool_spatial
        self.projection = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        features = x.mean(dim=(-2, -1)) if self.pool_spatial else x.flatten(1)
        return cast(torch.Tensor, self.projection(features))

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.predictor(z))


class EZV2Networks(nn.Module):
    """Combined wrapper holding all three sub-networks + SimSiam projector."""

    def __init__(self, game: ChessGame, cfg: EzV2LearnerConfig) -> None:
        super().__init__()
        _bx, _by, bz = game.get_board_size()
        obs_planes = bz
        action_size = game.get_action_size()
        channels = cfg.num_channels
        support_size = cfg.support_size
        repr_blocks = cfg.repr_blocks
        dyn_blocks = cfg.dyn_blocks

        self._obs_planes = obs_planes
        self.representation = RepresentationNetwork(obs_planes, channels, repr_blocks)
        self.dynamics = DynamicsNetwork(channels, support_size, dyn_blocks)
        self.prediction = PredictionNetwork(channels, action_size, support_size)

        latent_flat_dim = channels * 8 * 8
        self.simsiam = SimSiamProjector(latent_flat_dim, cfg.proj_dim)
        self.piece_reconstruction: nn.Module | None = None

        self.support_size = support_size
        self.action_size = action_size

    def initial_inference(
        self,
        observation: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_4d = self._obs_to_planes(observation)
        latent = _scale_latent(self.representation(obs_4d))
        policy_logits, value_logits = self.prediction(latent, valid_mask)
        log_policy = F.log_softmax(policy_logits, dim=1)
        scalar_value = _support_to_scalar(value_logits, self.support_size)
        return log_policy, scalar_value

    def initial_inference_with_latent(
        self,
        observation: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_4d = self._obs_to_planes(observation)
        latent = _scale_latent(self.representation(obs_4d))
        policy_logits, value_logits = self.prediction(latent, valid_mask)
        log_policy = F.log_softmax(policy_logits, dim=1)
        scalar_value = _support_to_scalar(value_logits, self.support_size)
        return latent, log_policy, scalar_value

    def initial_inference_for_training(
        self,
        observation: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the latent state, log policy, and raw value logits for loss computation."""
        obs_4d = self._obs_to_planes(observation)
        latent = _scale_latent(self.representation(obs_4d))
        policy_logits, value_logits = self.prediction(latent, valid_mask)
        return latent, F.log_softmax(policy_logits, dim=1), value_logits

    def recurrent_inference(
        self,
        latent: torch.Tensor,
        action_planes: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run dynamics + prediction. Returns (next_latent, reward_scalar, log_policy, scalar_value)."""
        next_latent, reward_logits = self.dynamics(latent, action_planes)
        next_latent_norm = _scale_latent(next_latent)
        policy_logits, value_logits = self.prediction(next_latent_norm, valid_mask)
        log_policy = F.log_softmax(policy_logits, dim=1)
        scalar_value = _support_to_scalar(value_logits, self.support_size)
        scalar_reward = _support_to_scalar(reward_logits, self.support_size)
        return next_latent_norm, scalar_reward, log_policy, scalar_value

    def _obs_to_planes(self, obs: torch.Tensor) -> torch.Tensor:
        """Reshape flat or HWC observation into (B, C, 8, 8)."""
        C = self._obs_planes
        if obs.dim() == 2:
            obs = obs.view(-1, 8, 8, C)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        if obs.dim() == 4 and obs.shape[1] != C and obs.shape[-1] == C:
            obs = obs.permute(0, 3, 1, 2)
        memory_format = torch.channels_last if obs.is_cuda else torch.contiguous_format
        return obs.contiguous(memory_format=memory_format)


def _scale_latent(latent: torch.Tensor) -> torch.Tensor:
    """Normalise latent per sample using mean/std for smooth gradient flow."""
    B = latent.size(0)
    flat = latent.reshape(B, -1)
    stats_input = flat.float() if flat.dtype in {torch.float16, torch.bfloat16} else flat
    mean = stats_input.mean(dim=1, keepdim=True)
    std = stats_input.std(dim=1, keepdim=True, correction=0).clamp(min=1e-5)
    normalised = (stats_input - mean) / std
    return normalised.to(dtype=latent.dtype).reshape_as(latent)


def scalar_to_support(x: torch.Tensor, support_size: int) -> torch.Tensor:
    """Convert scalar values to categorical support representation."""
    x = x.clamp(-support_size, support_size)
    floor = x.floor().long()
    prob_upper = x - floor.float()
    prob_lower = 1.0 - prob_upper

    bins = 2 * support_size + 1
    target = torch.zeros(x.size(0), bins, device=x.device, dtype=x.dtype)
    floor_idx = (floor + support_size).clamp(0, bins - 1)
    ceil_idx = (floor_idx + 1).clamp(0, bins - 1)

    target.scatter_(1, floor_idx.unsqueeze(1), prob_lower.unsqueeze(1))
    target.scatter_add_(1, ceil_idx.unsqueeze(1), prob_upper.unsqueeze(1))
    return target


def _support_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    """Convert categorical support logits back to scalar values."""
    probs = torch.softmax(logits, dim=1)
    support = torch.arange(-support_size, support_size + 1, device=logits.device, dtype=logits.dtype)
    return (probs * support).sum(dim=1)
