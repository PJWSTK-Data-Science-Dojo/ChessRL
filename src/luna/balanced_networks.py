"""Asymmetric SE-ResNet model optimized for single-accelerator latent MCTS."""

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from luna.config import EzV2LearnerConfig
from luna.ezv2_networks import (
    DynamicsNetwork,
    EZV2Networks,
    PredictionNetwork,
    RepresentationNetwork,
    SimSiamProjector,
    _flatten_spatial_policy,
    _scale_latent,
    _SpatialPolicyHead,
    _support_to_scalar,
)
from luna.game.chess_game import ACTION_SIZE, ChessGame


class LayerNorm2d(nn.Module):
    """Apply LayerNorm over channels independently at every board square."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_last = x.permute(0, 2, 3, 1)
        normalized = cast(torch.Tensor, self.norm(channels_last))
        return normalized.permute(0, 3, 1, 2)


class SqueezeExcitation(nn.Module):
    """Channel attention with a small global-context MLP."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden_channels, 1)
        self.fc2 = nn.Conv2d(hidden_channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.gelu(self.fc1(scale))
        return x * torch.sigmoid(self.fc2(scale))


class SEResBlock(nn.Module):
    """Dense residual block with channel attention, LayerNorm2D, and GELU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = LayerNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = LayerNorm2d(channels)
        self.se = SqueezeExcitation(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.se(self.norm2(self.conv2(x)))
        return F.gelu(x + residual)


class BalancedRepresentationNetwork(RepresentationNetwork):
    """Deep representation trunk that holds most model capacity."""

    def __init__(self, obs_planes: int, channels: int, num_blocks: int) -> None:
        nn.Module.__init__(self)
        self.conv_in = nn.Conv2d(obs_planes, channels, 3, padding=1, bias=False)
        self.norm_in = LayerNorm2d(channels)
        self.blocks = nn.Sequential(*(SEResBlock(channels) for _ in range(num_blocks)))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        latent = F.gelu(self.norm_in(self.conv_in(observation)))
        return cast(torch.Tensor, self.blocks(latent))


class BalancedDynamicsNetwork(DynamicsNetwork):
    """Shallow dense latent transition used repeatedly inside MCTS."""

    ACTION_PLANES = 5

    def __init__(self, channels: int, support_size: int, num_blocks: int) -> None:
        nn.Module.__init__(self)
        self.conv_in = nn.Conv2d(channels + self.ACTION_PLANES, channels, 3, padding=1, bias=False)
        self.norm_in = LayerNorm2d(channels)
        self.blocks = nn.Sequential(*(SEResBlock(channels) for _ in range(num_blocks)))
        reward_bins = 2 * support_size + 1
        self.reward_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 32),
            nn.GELU(),
            nn.Linear(32, reward_bins),
        )

    def forward(self, latent: torch.Tensor, action_planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transition = torch.cat((latent, action_planes), dim=1)
        transition = F.gelu(self.norm_in(self.conv_in(transition)))
        next_latent = self.blocks(transition)
        return cast(torch.Tensor, next_latent), self.reward_head(next_latent)


class BalancedPolicyHead(_SpatialPolicyHead):
    """Dense chess-spatial policy head using the repository's 4,288-action layout."""

    OUTPUT_CHANNELS = 88

    def __init__(self, channels: int, action_size: int) -> None:
        nn.Module.__init__(self)
        if action_size != ACTION_SIZE:
            raise ValueError(f"Balanced policy head requires {ACTION_SIZE} actions, got {action_size}")
        self.conv = nn.Conv2d(channels, self.OUTPUT_CHANNELS, 3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return _flatten_spatial_policy(self.conv(latent))


class BalancedPredictionNetwork(PredictionNetwork):
    """Policy and categorical value heads for the balanced model."""

    def __init__(self, channels: int, action_size: int, support_size: int) -> None:
        nn.Module.__init__(self)
        self.policy_head = BalancedPolicyHead(channels, action_size)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            LayerNorm2d(32),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.GELU(),
            nn.Linear(256, 2 * support_size + 1),
        )

    def forward(
        self,
        latent: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy_logits = self.policy_head(latent)
        if valid_mask is not None:
            policy_logits = policy_logits.masked_fill(
                valid_mask <= 0,
                torch.finfo(policy_logits.dtype).min,
            )
        return policy_logits, self.value_head(latent)


class PieceReconstructionHead(nn.Module):
    """Decode empty/white/black piece identity at every square during training."""

    NUM_CLASSES = 13

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(channels, self.NUM_CLASSES, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.classifier(latent))


class BalancedNetworks(EZV2Networks):
    """EfficientZeroV2 API backed by the balanced asymmetric architecture."""

    def __init__(self, game: ChessGame, cfg: EzV2LearnerConfig) -> None:
        nn.Module.__init__(self)
        _board_x, _board_y, obs_planes = game.get_board_size()
        action_size = game.get_action_size()
        self._obs_planes = obs_planes
        self.representation = BalancedRepresentationNetwork(obs_planes, cfg.num_channels, cfg.repr_blocks)
        self.dynamics = BalancedDynamicsNetwork(cfg.num_channels, cfg.support_size, cfg.dyn_blocks)
        self.prediction = BalancedPredictionNetwork(cfg.num_channels, action_size, cfg.support_size)
        self.simsiam = SimSiamProjector(cfg.num_channels, cfg.proj_dim, pool_spatial=True)
        self.piece_reconstruction: PieceReconstructionHead | None = None
        self.support_size = cfg.support_size
        self.action_size = action_size

    def recurrent_inference(
        self,
        latent: torch.Tensor,
        action_planes: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_latent, reward_logits = self.dynamics(latent, action_planes)
        next_latent = _scale_latent(next_latent)
        policy_logits, value_logits = self.prediction(next_latent, valid_mask)
        log_policy = F.log_softmax(policy_logits, dim=1)
        return (
            next_latent,
            _support_to_scalar(reward_logits, self.support_size),
            log_policy,
            _support_to_scalar(value_logits, self.support_size),
        )


class BalancedReconstructionNetworks(BalancedNetworks):
    """Balanced inference model with a training-only chess-state anchor."""

    def __init__(self, game: ChessGame, cfg: EzV2LearnerConfig) -> None:
        super().__init__(game, cfg)
        self.piece_reconstruction = PieceReconstructionHead(cfg.num_channels)
