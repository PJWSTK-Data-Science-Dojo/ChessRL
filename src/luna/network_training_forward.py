"""Forward and backward work for one EfficientZeroV2 optimizer step."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from luna.ezv2_networks import _scale_latent, _support_to_scalar, scalar_to_support
from luna.network_runtime import scale_gradient
from luna.network_training_diagnostics import training_diagnostics
from luna.network_training_types import (
    ForwardResult,
    LossComponents,
    Microbatch,
    RootState,
    StepAccumulation,
    TrainingFunctions,
    TrainingSettings,
    UnrollState,
)
from luna.network_types import NetworkRuntime, PreparedBatch


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
        policy_mask=_tensor(network, values["policy_mask"]),
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
        health = training_diagnostics(network, batch, root, unroll, settings, functions) if report else {}
    torch.autograd.backward(network.scaler.scale(losses.total))
    errors = _priority_errors(root, batch)
    return ForwardResult(losses, errors, health)


def _priority_errors(root: RootState, batch: Microbatch) -> np.ndarray:
    value_error = (root.value_prediction.float() - batch.target_values[:, 0]).abs()
    policy_target = batch.target_policies[:, 0].float()
    policy_entropy = -torch.xlogy(policy_target, policy_target).sum(dim=1)
    policy_kl = (root.policy_loss.float() - policy_entropy).clamp_min(0.0)
    policy_error = policy_kl * batch.policy_mask[:, 0]
    errors = torch.where(batch.value_mask[:, 0].bool(), value_error, policy_error)
    return errors.detach().cpu().numpy()


def _root_state(
    network: NetworkRuntime,
    batch: Microbatch,
    settings: TrainingSettings,
    functions: TrainingFunctions,
) -> RootState:
    latent, log_policy, value_logits = network._training_initial_inference(batch.observations, batch.valid_moves)
    value_prediction = _support_to_scalar(value_logits, settings.support)
    policy_loss = -(batch.target_policies[:, 0] * log_policy).sum(dim=1) * batch.policy_mask[:, 0]
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
    policy_mask = batch.policy_mask[:, index + 1]
    policy = -(batch.target_policies[:, index + 1] * F.log_softmax(policy_logits, dim=1)).sum(dim=1) * policy_mask
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
    policy = unroll.policy_loss / batch.policy_mask.sum(dim=1).clamp(min=1.0)
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
