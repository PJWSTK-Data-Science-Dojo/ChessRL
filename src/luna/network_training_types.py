"""Typed state passed between the EfficientZeroV2 training components."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import torch

from luna.ezv2_networks import SimSiamProjector
from luna.utils import AverageMeter

SoftCrossEntropy = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ConsistencyLoss = Callable[[SimSiamProjector, torch.Tensor, torch.Tensor], torch.Tensor]
ReconstructionLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
PieceTargets = Callable[[torch.Tensor], torch.Tensor]
RawLatentMetrics = Callable[[str, torch.Tensor], dict[str, float]]
ReconstructionMetrics = Callable[[str, torch.Tensor, torch.Tensor], dict[str, float]]
LatentMetrics = Callable[[SimSiamProjector, torch.Tensor, torch.Tensor], dict[str, float]]
NonFiniteGradients = Callable[[Iterable[torch.nn.Parameter]], bool]


@dataclass(frozen=True, slots=True)
class TrainingFunctions:
    soft_cross_entropy: SoftCrossEntropy
    consistency_loss: ConsistencyLoss
    reconstruction_loss: ReconstructionLoss
    piece_targets: PieceTargets
    raw_latent_metrics: RawLatentMetrics
    reconstruction_metrics: ReconstructionMetrics
    latent_metrics: LatentMetrics
    has_non_finite_gradients: NonFiniteGradients
    maximum_amp_skips: int


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    steps: int
    batch_size: int
    micro_batch_size: int
    unroll: int
    support: int
    gradient_accumulation: int
    learning_rate_horizon: int
    discount: float
    consistency_enabled: bool
    diagnostics_interval: int = 50

    def should_report(self, step: int) -> bool:
        return step % self.diagnostics_interval == 0 or step == self.steps


@dataclass(frozen=True, slots=True)
class Microbatch:
    observations: torch.Tensor
    valid_moves: torch.Tensor
    target_values: torch.Tensor
    target_rewards: torch.Tensor
    target_policies: torch.Tensor
    unroll_observations: torch.Tensor
    actions: torch.Tensor
    importance_weights: torch.Tensor
    unroll_mask: torch.Tensor
    consistency_mask: torch.Tensor
    value_mask: torch.Tensor
    unroll_valid_moves: torch.Tensor
    tree_indices: list[int]


@dataclass(frozen=True, slots=True)
class LossComponents:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    consistency: torch.Tensor
    reconstruction: torch.Tensor


@dataclass(frozen=True, slots=True)
class ForwardResult:
    losses: LossComponents
    priority_errors: np.ndarray
    latent_health: dict[str, float]


@dataclass(slots=True)
class StepAccumulation:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    consistency: torch.Tensor
    reconstruction: torch.Tensor
    priority_errors: list[np.ndarray] = field(default_factory=list)
    tree_indices: list[list[int]] = field(default_factory=list)
    latent_health: dict[str, float] = field(default_factory=dict)

    @classmethod
    def empty(cls, device: torch.device) -> StepAccumulation:
        def scalar() -> torch.Tensor:
            return torch.zeros((), device=device, dtype=torch.float32)

        return cls(scalar(), scalar(), scalar(), scalar(), scalar(), scalar())

    def add(self, result: ForwardResult, tree_indices: list[int]) -> None:
        losses = result.losses
        self.total = self.total + losses.total.detach().float()
        self.policy = self.policy + losses.policy.detach().float()
        self.value = self.value + losses.value.detach().float()
        self.reward = self.reward + losses.reward.detach().float()
        self.consistency = self.consistency + losses.consistency.detach().float()
        self.reconstruction = self.reconstruction + losses.reconstruction.detach().float()
        self.priority_errors.append(result.priority_errors)
        self.tree_indices.append(tree_indices)
        self.latent_health.update(result.latent_health)


@dataclass(slots=True)
class TrainingMeters:
    total: AverageMeter = field(default_factory=AverageMeter)
    policy: AverageMeter = field(default_factory=AverageMeter)
    value: AverageMeter = field(default_factory=AverageMeter)
    reward: AverageMeter = field(default_factory=AverageMeter)
    consistency: AverageMeter = field(default_factory=AverageMeter)
    reconstruction: AverageMeter = field(default_factory=AverageMeter)
    step_time: AverageMeter = field(default_factory=AverageMeter)
    grad_norm_preclip: AverageMeter = field(default_factory=AverageMeter)
    grad_norm_postclip: AverageMeter = field(default_factory=AverageMeter)
    grad_clip_coefficient: AverageMeter = field(default_factory=AverageMeter)
    grad_clip_fraction: AverageMeter = field(default_factory=AverageMeter)
    reanalysis_samples: AverageMeter = field(default_factory=AverageMeter)
    reanalysis_positions: AverageMeter = field(default_factory=AverageMeter)
    reanalysis_seconds: AverageMeter = field(default_factory=AverageMeter)

    def losses(self) -> dict[str, float]:
        return {
            "total": self.total.avg,
            "policy": self.policy.avg,
            "value": self.value.avg,
            "reward": self.reward.avg,
            "consistency": self.consistency.avg,
            "reconstruction": self.reconstruction.avg,
        }


@dataclass(frozen=True, slots=True)
class OptimizerOutcome:
    gradient_overflow: bool
    gradient_norm: float
    previous_scale: float
    current_scale: float
