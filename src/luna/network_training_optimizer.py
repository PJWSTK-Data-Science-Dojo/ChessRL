"""Gradient validation and optimizer updates for EfficientZeroV2 training."""

from __future__ import annotations

import torch

from luna.network_training_types import OptimizerOutcome, StepAccumulation, TrainingFunctions
from luna.network_types import NetworkRuntime


def apply_optimizer_update(
    network: NetworkRuntime,
    accumulation: StepAccumulation,
    previous_learning_rates: list[float],
    functions: TrainingFunctions,
) -> OptimizerOutcome:
    _validate_loss(accumulation)
    network.scaler.unscale_(network.optimizer)
    scaler_enabled = network.scaler.is_enabled()
    previous_scale = network.scaler.get_scale() if scaler_enabled else 1.0
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        network.nnet.parameters(),
        network._learner.grad_clip_norm,
        error_if_nonfinite=not scaler_enabled,
    )
    finite_norm = bool(torch.isfinite(gradient_norm))
    gradient_overflow = (
        scaler_enabled and not finite_norm and functions.has_non_finite_gradients(network.nnet.parameters())
    )
    if scaler_enabled and not finite_norm and not gradient_overflow:
        _reject_finite_norm_overflow(network, previous_learning_rates, previous_scale)
    network.scaler.step(network.optimizer)
    network.scaler.update()
    network.optimizer.zero_grad(set_to_none=True)
    current_scale = network.scaler.get_scale() if scaler_enabled else previous_scale
    return OptimizerOutcome(gradient_overflow, float(gradient_norm), previous_scale, current_scale)


def _validate_loss(accumulation: StepAccumulation) -> None:
    if bool(torch.isfinite(accumulation.total).all()):
        return
    raise RuntimeError(
        "Training diverged: loss is NaN or Inf. "
        "Try lowering learning rate, increasing gradient clipping, or checking data preprocessing."
    )


def _reject_finite_norm_overflow(
    network: NetworkRuntime,
    previous_learning_rates: list[float],
    previous_scale: float,
) -> None:
    network.scaler.update(new_scale=previous_scale)
    network.optimizer.zero_grad(set_to_none=True)
    restore_learning_rates(network, previous_learning_rates)
    raise RuntimeError("Gradient norm overflowed despite finite gradient elements; optimizer update was not applied.")


def set_learning_rate(network: NetworkRuntime, learning_rate: float) -> list[float]:
    previous = [float(group["lr"]) for group in network.optimizer.param_groups]
    for group in network.optimizer.param_groups:
        group["lr"] = learning_rate
    return previous


def restore_learning_rates(network: NetworkRuntime, learning_rates: list[float]) -> None:
    for group, learning_rate in zip(network.optimizer.param_groups, learning_rates, strict=True):
        group["lr"] = learning_rate
