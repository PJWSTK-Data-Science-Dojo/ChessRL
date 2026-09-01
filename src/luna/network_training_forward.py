"""Forward and backward work for one EfficientZeroV2 optimizer step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from luna.ezv2_networks import _scale_latent, _support_to_scalar, scalar_to_support
from luna.network_runtime import scale_gradient
from luna.network_training_types import (
    ForwardResult,
    LossComponents,
    Microbatch,
    StepAccumulation,
    TrainingFunctions,
    TrainingSettings,
)
from luna.network_types import NetworkRuntime, PreparedBatch


@dataclass(frozen=True, slots=True)
class RootState:
    latent: torch.Tensor
    value_prediction: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    reconstruction_logits: torch.Tensor | None
    reconstruction_target: torch.Tensor | None
    target_latents: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class UnrollState:
    next_latent: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    reward_loss: torch.Tensor
    consistency_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    reconstruction_logits: torch.Tensor | None
    reconstruction_target: torch.Tensor | None


def run_microbatches(
    network: NetworkRuntime,
    prepared: PreparedBatch,
    step: int,
    settings: TrainingSettings,
    functions: TrainingFunctions,
) -> StepAccumulation:
    accumulation = StepAccumulation.empty(network.device)
    for index in range(settings.gradient_accumulation):
        microbatch = _build_microbatch(network, prepared, index, settings)
        report = settings.should_report(step) and index == settings.gradient_accumulation - 1
        result = _forward_and_backward(network, microbatch, settings, functions, report)
        accumulation.add(result, microbatch.tree_indices)
    return accumulation


def _build_microbatch(
    network: NetworkRuntime,
    prepared: PreparedBatch,
    index: int,
    settings: TrainingSettings,
) -> Microbatch:
    start = index * settings.micro_batch_size
    stop = start + settings.micro_batch_size
    values = {name: value[start:stop] for name, value in prepared.collated.items()}
    return Microbatch(
        observations=_tensor(network, values["observations"]),
        valid_moves=_tensor(network, values["valid_masks"]),
        target_values=_tensor(network, values["target_values"]),
        target_rewards=_tensor(network, values["target_rewards"]),
        target_policies=_tensor(network, values["target_policies"]),
        unroll_observations=_tensor(network, values["observations_unroll"]),
        actions=_tensor(network, values["actions"], dtype=torch.long),
        importance_weights=_tensor(network, prepared.is_weights[start:stop]),
        unroll_mask=_tensor(network, values["unroll_mask"]),
        consistency_mask=_tensor(network, values["consistency_mask"]),
        value_mask=_tensor(network, values["value_mask"]),
        unroll_valid_moves=_tensor(network, values["valid_masks_unroll"]),
        tree_indices=prepared.tree_indices[start:stop],
    )


def _tensor(
    network: NetworkRuntime,
    values: np.ndarray,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.as_tensor(values, dtype=dtype, device=network.device)


def _forward_and_backward(
    network: NetworkRuntime,
    batch: Microbatch,
    settings: TrainingSettings,
    functions: TrainingFunctions,
    report: bool,
) -> ForwardResult:
    with torch.autocast(
        "cuda",
        enabled=network._learner.mixed_precision and network.device.type == "cuda",
        dtype=network._amp_dtype,
    ):
        root = _root_state(network, batch, settings, functions)
        unroll = _unroll_state(network, batch, root, settings, functions)
        losses = _weighted_losses(network, batch, root, unroll, settings)
        health = _diagnostics(network, batch, root, unroll, settings, functions) if report else {}
    torch.autograd.backward(network.scaler.scale(losses.total))
    errors = (root.value_prediction.float() - batch.target_values[:, 0]).abs().detach().cpu().numpy()
    return ForwardResult(losses, errors, health)


def _root_state(
    network: NetworkRuntime,
    batch: Microbatch,
    settings: TrainingSettings,
    functions: TrainingFunctions,
) -> RootState:
    latent, log_policy, value_logits = network._training_initial_inference(batch.observations, batch.valid_moves)
    value_prediction = _support_to_scalar(value_logits, settings.support)
    policy_loss = -(batch.target_policies[:, 0] * log_policy).sum(dim=1) * batch.value_mask[:, 0]
    value_target = scalar_to_support(batch.target_values[:, 0], settings.support)
    value_loss = functions.soft_cross_entropy(value_logits, value_target) * batch.value_mask[:, 0]
    reconstruction_logits, reconstruction_target, reconstruction_loss = _root_reconstruction(
        network,
        latent,
        batch.observations,
        settings.micro_batch_size,
        functions,
    )
    return RootState(
        latent,
        value_prediction,
        policy_loss,
        value_loss,
        reconstruction_loss,
        reconstruction_logits,
        reconstruction_target,
        _target_latents(network, batch, settings),
    )


def _root_reconstruction(
    network: NetworkRuntime,
    latent: torch.Tensor,
    observations: torch.Tensor,
    batch_size: int,
    functions: TrainingFunctions,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    head = _reconstruction_head(network)
    if head is None:
        return None, None, torch.zeros(batch_size, device=network.device)
    logits = head(latent)
    target = functions.piece_targets(observations)
    return logits, target, functions.reconstruction_loss(logits, target)


def _target_latents(
    network: NetworkRuntime,
    batch: Microbatch,
    settings: TrainingSettings,
) -> torch.Tensor | None:
    if not settings.consistency_enabled:
        return None
    with torch.no_grad():
        flat_observations = batch.unroll_observations[:, 1:].reshape(-1, *batch.unroll_observations.shape[2:])
        flat_planes = network.nnet._obs_to_planes(flat_observations)
        targets = _scale_latent(network._training_representation(flat_planes))
    return targets.view(settings.micro_batch_size, settings.unroll, *targets.shape[1:])


def _unroll_state(
    network: NetworkRuntime,
    batch: Microbatch,
    root: RootState,
    settings: TrainingSettings,
    functions: TrainingFunctions,
) -> UnrollState:
    state = _empty_unroll(network, root, settings.micro_batch_size)
    for index in range(settings.unroll):
        state = _advance_unroll(network, batch, root.target_latents, state, index, settings, functions)
    return state


def _empty_unroll(network: NetworkRuntime, root: RootState, batch_size: int) -> UnrollState:
    zeros = torch.zeros(batch_size, device=network.device)
    return UnrollState(
        root.latent, root.policy_loss, root.value_loss, zeros, zeros, root.reconstruction_loss, None, None
    )


def _advance_unroll(
    network: NetworkRuntime,
    batch: Microbatch,
    target_latents: torch.Tensor | None,
    state: UnrollState,
    index: int,
    settings: TrainingSettings,
    functions: TrainingFunctions,
) -> UnrollState:
    action_planes = network._encode_action_planes(batch.actions[:, index])
    dynamics_input = scale_gradient(state.next_latent, network._learner.recurrent_gradient_scale)
    next_latent_raw, reward_logits = network._training_dynamics(dynamics_input, action_planes)
    next_latent = _scale_latent(next_latent_raw)
    policy_logits, value_logits = network._training_prediction(next_latent, batch.unroll_valid_moves[:, index + 1])
    losses = _unroll_losses(
        batch,
        target_latents,
        next_latent,
        reward_logits,
        policy_logits,
        value_logits,
        index,
        settings,
        functions,
        network,
    )
    return UnrollState(
        next_latent,
        state.policy_loss + losses[0],
        state.value_loss + losses[1],
        state.reward_loss + losses[2],
        state.consistency_loss + losses[3],
        state.reconstruction_loss + losses[4],
        losses[5],
        losses[6],
    )


def _unroll_losses(
    batch: Microbatch,
    target_latents: torch.Tensor | None,
    next_latent: torch.Tensor,
    reward_logits: torch.Tensor,
    policy_logits: torch.Tensor,
    value_logits: torch.Tensor,
    index: int,
    settings: TrainingSettings,
    functions: TrainingFunctions,
    network: NetworkRuntime,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
]:
    value_mask = batch.value_mask[:, index + 1]
    reward = (
        functions.soft_cross_entropy(
            reward_logits,
            scalar_to_support(batch.target_rewards[:, index], settings.support),
        )
        * batch.unroll_mask[:, index]
    )
    policy = -(batch.target_policies[:, index + 1] * F.log_softmax(policy_logits, dim=1)).sum(dim=1) * value_mask
    value = (
        functions.soft_cross_entropy(
            value_logits,
            scalar_to_support(batch.target_values[:, index + 1], settings.support),
        )
        * value_mask
    )
    consistency = _consistency_loss(network, batch, target_latents, next_latent, index, functions)
    reconstruction, logits, target = _unroll_reconstruction(network, batch, next_latent, index, functions)
    return policy, value, reward, consistency, reconstruction, logits, target


def _consistency_loss(
    network: NetworkRuntime,
    batch: Microbatch,
    target_latents: torch.Tensor | None,
    next_latent: torch.Tensor,
    index: int,
    functions: TrainingFunctions,
) -> torch.Tensor:
    if target_latents is None:
        return torch.zeros(next_latent.shape[0], device=network.device)
    return (
        functions.consistency_loss(network.nnet.simsiam, next_latent, target_latents[:, index])
        * batch.consistency_mask[:, index]
    )


def _unroll_reconstruction(
    network: NetworkRuntime,
    batch: Microbatch,
    next_latent: torch.Tensor,
    index: int,
    functions: TrainingFunctions,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    head = _reconstruction_head(network)
    if head is None:
        return torch.zeros(next_latent.shape[0], device=network.device), None, None
    logits = head(next_latent)
    target = functions.piece_targets(batch.unroll_observations[:, index + 1])
    loss = functions.reconstruction_loss(logits, target) * batch.consistency_mask[:, index]
    return loss, logits, target


def _reconstruction_head(network: NetworkRuntime) -> torch.nn.Module | None:
    if network._learner.reconstruction_loss_weight <= 0.0:
        return None
    head = network.nnet.piece_reconstruction
    if head is None:
        raise RuntimeError("The configured reconstruction objective has no reconstruction head")
    return head


def _weighted_losses(
    network: NetworkRuntime,
    batch: Microbatch,
    root: RootState,
    unroll: UnrollState,
    settings: TrainingSettings,
) -> LossComponents:
    policy = unroll.policy_loss / batch.value_mask.sum(dim=1).clamp(min=1.0)
    value = unroll.value_loss / batch.value_mask.sum(dim=1).clamp(min=1.0)
    reward = unroll.reward_loss / batch.unroll_mask.sum(dim=1).clamp(min=1.0)
    consistency = unroll.consistency_loss / batch.consistency_mask.sum(dim=1).clamp(min=1.0)
    reconstruction = unroll.reconstruction_loss / (1.0 + batch.consistency_mask.sum(dim=1))
    learner = network._learner
    total = learner.policy_loss_weight * policy + learner.value_loss_weight * value
    total = total + learner.reward_loss_weight * reward + learner.consistency_loss_weight * consistency
    total = total + learner.reconstruction_loss_weight * reconstruction
    scale = batch.importance_weights / settings.gradient_accumulation
    return LossComponents(
        (total * scale).mean(),
        (policy * scale).mean(),
        (value * scale).mean(),
        (reward * scale).mean(),
        (consistency * scale).mean(),
        (reconstruction * scale).mean(),
    )


def _diagnostics(
    network: NetworkRuntime,
    batch: Microbatch,
    root: RootState,
    unroll: UnrollState,
    settings: TrainingSettings,
    functions: TrainingFunctions,
) -> dict[str, float]:
    metrics = functions.raw_latent_metrics("root", root.latent)
    active_dynamics = batch.unroll_mask[:, -1].bool()
    if bool(active_dynamics.any()):
        metrics.update(functions.raw_latent_metrics("predicted", unroll.next_latent[active_dynamics]))
    _add_target_metrics(metrics, batch, settings)
    active_consistency = batch.consistency_mask[:, -1].bool()
    if root.target_latents is not None and bool(active_consistency.any()):
        metrics.update(
            functions.latent_metrics(
                network.nnet.simsiam,
                unroll.next_latent[active_consistency],
                root.target_latents[:, -1][active_consistency],
            )
        )
    if root.reconstruction_logits is not None and root.reconstruction_target is not None:
        metrics.update(functions.reconstruction_metrics("root", root.reconstruction_logits, root.reconstruction_target))
    if (
        unroll.reconstruction_logits is not None
        and unroll.reconstruction_target is not None
        and bool(active_consistency.any())
    ):
        metrics.update(
            functions.reconstruction_metrics(
                "predicted",
                unroll.reconstruction_logits[active_consistency],
                unroll.reconstruction_target[active_consistency],
            )
        )
    return metrics


def _add_target_metrics(
    metrics: dict[str, float],
    batch: Microbatch,
    settings: TrainingSettings,
) -> None:
    active_values = batch.value_mask.bool()
    count = active_values.sum().clamp(min=1)
    active_consistency = batch.consistency_mask[:, -1].bool()
    metrics["train/value_target_nonzero_fraction"] = float(
        ((batch.target_values.abs() > 0.5) & active_values).sum().item() / count.item()
    )
    metrics["train/value_target_mean_abs"] = float(
        (batch.target_values.abs() * batch.value_mask).sum().item() / count.item()
    )
    metrics["train/next_observation_active_fraction"] = float(batch.consistency_mask.mean().item())
    metrics["train/next_observation_active_samples"] = float(active_consistency.sum().item())
    metrics["train/consistency_objective_enabled"] = float(settings.consistency_enabled)
    metrics["train/consistency_active_fraction"] = float(batch.consistency_mask.mean().item())
    metrics["train/latent_health_active_samples"] = float(active_consistency.sum().item())
