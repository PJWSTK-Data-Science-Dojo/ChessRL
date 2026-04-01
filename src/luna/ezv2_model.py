"""EfficientZeroV2 network architecture: representation, dynamics, prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .game.luna_game import ACTION_SIZE, ChessGame
from .utils import dotdict

LATENT_PLANES = 64
LATENT_SPATIAL = 4


def _make_residual_block(channels: int) -> nn.Module:
    return _ResBlock(channels)


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class RepresentationNetwork(nn.Module):
    """h(observation) -> latent state."""

    def __init__(self, obs_planes: int, channels: int, num_blocks: int = 4) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(obs_planes, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[_make_residual_block(channels) for _ in range(num_blocks)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_planes, 8, 8) -> latent: (B, channels, 8, 8)."""
        x = F.relu(self.bn_in(self.conv_in(obs)))
        return self.blocks(x)


class DynamicsNetwork(nn.Module):
    """g(latent, action) -> (next_latent, reward_logits)."""

    def __init__(self, channels: int, action_size: int, support_size: int, num_blocks: int = 2) -> None:
        super().__init__()
        self.action_size = action_size
        self.channels = channels
        self.action_embed = nn.Linear(action_size, channels * 8 * 8)
        self.conv_in = nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[_make_residual_block(channels) for _ in range(num_blocks)])

        reward_bins = 2 * support_size + 1
        self.reward_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, reward_bins),
        )

    def forward(self, latent: torch.Tensor, action_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (next_latent, reward_logits)."""
        B = latent.size(0)
        action_plane = self.action_embed(action_onehot).view(B, self.channels, 8, 8)
        x = torch.cat([latent, action_plane], dim=1)
        x = F.relu(self.bn_in(self.conv_in(x)))
        next_latent = self.blocks(x)
        reward_logits = self.reward_head(next_latent)
        return next_latent, reward_logits


class PredictionNetwork(nn.Module):
    """f(latent) -> (policy_logits, value_logits)."""

    def __init__(self, channels: int, action_size: int, support_size: int) -> None:
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, action_size),
        )

        value_bins = 2 * support_size + 1
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, value_bins),
        )

    def forward(self, latent: torch.Tensor, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        policy_logits = self.policy_head(latent)
        if valid_mask is not None:
            policy_logits = policy_logits - (1 - valid_mask) * 1e6
        value_logits = self.value_head(latent)
        return policy_logits, value_logits


class EZV2Networks(nn.Module):
    """Combined wrapper holding all three sub-networks."""

    def __init__(self, game: ChessGame, args: dotdict) -> None:
        super().__init__()
        bx, by, bz = game.get_board_size()
        obs_planes = bz
        action_size = game.get_action_size()
        channels = args.get("num_channels", 128)
        support_size = args.get("support_size", 10)
        repr_blocks = args.get("repr_blocks", 4)
        dyn_blocks = args.get("dyn_blocks", 2)

        self.representation = RepresentationNetwork(obs_planes, channels, repr_blocks)
        self.dynamics = DynamicsNetwork(channels, action_size, support_size, dyn_blocks)
        self.prediction = PredictionNetwork(channels, action_size, support_size)

        self.support_size = support_size
        self.action_size = action_size

    def initial_inference(
        self,
        observation: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run representation + prediction. Returns (log_policy, scalar_value)."""
        obs_4d = self._obs_to_planes(observation)
        latent = self.representation(obs_4d)
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
        latent = self.representation(obs_4d)
        policy_logits, value_logits = self.prediction(latent, valid_mask)
        log_policy = F.log_softmax(policy_logits, dim=1)
        scalar_value = _support_to_scalar(value_logits, self.support_size)
        return latent, log_policy, scalar_value

    def recurrent_inference(
        self,
        latent: torch.Tensor,
        action_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run dynamics + prediction. Returns (next_latent, reward_scalar, log_policy, scalar_value)."""
        next_latent, reward_logits = self.dynamics(latent, action_onehot)
        next_latent_norm = _scale_latent(next_latent)
        policy_logits, value_logits = self.prediction(next_latent_norm)
        log_policy = F.log_softmax(policy_logits, dim=1)
        scalar_value = _support_to_scalar(value_logits, self.support_size)
        scalar_reward = _support_to_scalar(reward_logits, self.support_size)
        return next_latent_norm, scalar_reward, log_policy, scalar_value

    def _obs_to_planes(self, obs: torch.Tensor) -> torch.Tensor:
        """Reshape flat (B, 8*8*6) or (B, 8, 8, 6) observation into (B, 6, 8, 8)."""
        if obs.dim() == 2:
            obs = obs.view(-1, 8, 8, 6)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] == 6 and obs.shape[1] != 6:
            obs = obs.permute(0, 3, 1, 2)
        return obs.contiguous()


def _scale_latent(latent: torch.Tensor) -> torch.Tensor:
    """Min-max normalise latent to [0, 1] per sample for stability (EZV2 trick)."""
    B = latent.size(0)
    flat = latent.view(B, -1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    scale = (mx - mn).clamp(min=1e-5)
    normalised = (flat - mn) / scale
    return normalised.view_as(latent)


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
    target.scatter_(1, ceil_idx.unsqueeze(1), prob_upper.unsqueeze(1))
    return target


def _support_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    """Convert categorical support logits back to scalar values."""
    probs = torch.softmax(logits, dim=1)
    support = torch.arange(-support_size, support_size + 1, device=logits.device, dtype=logits.dtype)
    return (probs * support).sum(dim=1)
