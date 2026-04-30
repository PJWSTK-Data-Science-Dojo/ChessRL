"""EfficientZeroV2 network architecture: representation, dynamics, prediction.

Key design choices:
- Spatial action planes (2-channel from/to encoding) instead of dense Linear(4096, C*8*8).
  Cuts ~76% of parameters and gives the conv stack spatially meaningful input.
- SimSiam-style projection + prediction heads for the consistency loss.
- GroupNorm everywhere (stable at batch=1 during MCTS inference, unlike BatchNorm).
- Depthwise-separable residual blocks in the dynamics network for faster MCTS rollouts.
- Mean/std latent normalisation for smoother gradients (replaces min-max).
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from .config import EzV2LearnerConfig
from .game.chess_game import ChessGame

_NUM_GROUPS = 8


def _num_groups(channels: int) -> int:
    g = _NUM_GROUPS
    while channels % g != 0 and g > 1:
        g //= 2
    return g


# ------------------------------------------------------------------
# Residual blocks
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Sub-networks
# ------------------------------------------------------------------

class RepresentationNetwork(nn.Module):
    """h(observation) -> latent state."""

    def __init__(self, obs_planes: int, channels: int, num_blocks: int = 4) -> None:
        super().__init__()
        g = _num_groups(channels)
        self.conv_in = nn.Conv2d(obs_planes, channels, 3, padding=1, bias=False)
        self.gn_in = nn.GroupNorm(g, channels)
        self.blocks = nn.Sequential(*[_make_residual_block(channels) for _ in range(num_blocks)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_planes, 8, 8) -> latent: (B, channels, 8, 8)."""
        x = F.relu(self.gn_in(self.conv_in(obs)))
        return cast(torch.Tensor, self.blocks(x))


class DynamicsNetwork(nn.Module):
    """g(latent, action_planes) -> (next_latent, reward_logits).

    Uses depthwise-separable blocks for speed (called once per MCTS simulation).
    """

    ACTION_PLANES = 2

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
        """action_planes: (B, 2, 8, 8) with from/to spatial encoding."""
        x = torch.cat([latent, action_planes], dim=1)
        x = F.relu(self.gn_in(self.conv_in(x)))
        next_latent = self.blocks(x)
        reward_logits = self.reward_head(next_latent)
        return next_latent, reward_logits


class PredictionNetwork(nn.Module):
    """f(latent) -> (policy_logits, value_logits)."""

    def __init__(self, channels: int, action_size: int, support_size: int) -> None:
        super().__init__()
        g32 = _num_groups(32)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.GroupNorm(g32, 32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, action_size),
        )

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
            policy_logits = policy_logits - (1 - valid_mask) * 1e6
        value_logits = self.value_head(latent)
        return policy_logits, value_logits


class SimSiamProjector(nn.Module):
    """SimSiam projection + prediction heads for consistency loss (EfficientZero)."""

    def __init__(self, in_dim: int, proj_dim: int = 256) -> None:
        super().__init__()
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
        return cast(torch.Tensor, self.projection(x.flatten(1)))

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.predictor(z))


# ------------------------------------------------------------------
# Combined model
# ------------------------------------------------------------------

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

        self.support_size = support_size
        self.action_size = action_size

    def initial_inference(
        self,
        observation: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run representation + prediction. Returns (log_policy, scalar_value)."""
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
        """Returns (latent, log_policy, scalar_value)."""
        obs_4d = self._obs_to_planes(observation)
        latent = _scale_latent(self.representation(obs_4d))
        policy_logits, value_logits = self.prediction(latent, valid_mask)
        log_policy = F.log_softmax(policy_logits, dim=1)
        scalar_value = _support_to_scalar(value_logits, self.support_size)
        return latent, log_policy, scalar_value

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
        return obs.contiguous()


# ---------------------------------------------------------------------------
# Action encoding helpers
# ---------------------------------------------------------------------------

def _action_to_squares(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode action indices into (from_square, to_square) handling promotions.

    Actions 0..4095 are from_sq*64 + to_sq.  Actions >= 4096 encode underpromotions
    with from_file/to_file offsets (see chess_game.py).

    Underpromotion layout (64 actions each, from_file * 8 + to_file):
        4096..4159  knight
        4160..4223  rook
        4224..4287  bishop
    All underpromotions move from rank 7 (sq 48..55) to rank 8 (sq 56..63).
    """
    is_base = action < 4096
    base_from = action // 64
    base_to = action % 64

    promo_offset = (action - 4096) % 64
    from_file = promo_offset // 8
    to_file = promo_offset % 8
    promo_from = from_file + 48  # rank 7
    promo_to = to_file + 56  # rank 8

    from_sq = torch.where(is_base, base_from, promo_from)
    to_sq = torch.where(is_base, base_to, promo_to)
    return from_sq.clamp(0, 63), to_sq.clamp(0, 63)


def action_index_to_planes(action: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert a batch of action indices to (B, 2, 8, 8) spatial planes.

    Plane 0: one-hot at from_square, Plane 1: one-hot at to_square.
    """
    B = action.shape[0]
    from_sq, to_sq = _action_to_squares(action)
    planes = torch.zeros(B, 2, 64, device=device)
    planes[torch.arange(B, device=device), 0, from_sq] = 1.0
    planes[torch.arange(B, device=device), 1, to_sq] = 1.0
    return planes.view(B, 2, 8, 8)


def action_int_to_planes(action: int, device: torch.device) -> torch.Tensor:
    """Single action index -> (1, 2, 8, 8) spatial planes."""
    action_t = torch.tensor([action], device=device)
    return action_index_to_planes(action_t, device)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _scale_latent(latent: torch.Tensor) -> torch.Tensor:
    """Normalise latent per sample using mean/std for smooth gradient flow."""
    B = latent.size(0)
    flat = latent.reshape(B, -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp(min=1e-5)
    normalised = (flat - mean) / std
    return normalised.reshape_as(latent)


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
