"""Root-only LC0 policy and soft WDL objective for online training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from luna.lc0_dataset import Lc0Batch
from luna.network_types import NetworkRuntime


@dataclass(frozen=True, slots=True)
class ExpertAnchorMetrics:
    weighted_loss: torch.Tensor
    policy_cross_entropy: torch.Tensor
    value_wdl_cross_entropy: torch.Tensor
    wdl_accuracy: torch.Tensor
    q_mae: torch.Tensor
    positions: int


def expert_anchor_forward_and_backward(
    network: NetworkRuntime,
    batch: Lc0Batch,
) -> ExpertAnchorMetrics:
    positions = _validate_batch(network, batch)
    totals = torch.zeros(5, device=network.device, dtype=torch.float32)
    chunk_size = max(1, network._learner.batch_size // network._learner.grad_accum_steps)
    for start in range(0, positions, chunk_size):
        stop = min(start + chunk_size, positions)
        losses = _chunk_losses(network, batch, slice(start, stop))
        count = stop - start
        torch.autograd.backward(network.scaler.scale(losses[0] * (count / positions)))
        totals += torch.stack([loss.detach().float() for loss in losses]) * count
    averages = totals / positions
    return ExpertAnchorMetrics(
        averages[0],
        averages[1],
        averages[2],
        averages[3],
        averages[4],
        positions,
    )


def _chunk_losses(
    network: NetworkRuntime,
    batch: Lc0Batch,
    indices: slice,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observations, valid_moves, policies, values = _batch_tensors(network, batch, indices)
    with torch.autocast(
        "cuda",
        enabled=network._learner.mixed_precision and network.device.type == "cuda",
        dtype=network._amp_dtype,
    ):
        _latent, log_policy, value_logits = network._training_initial_inference(observations, valid_moves)
        policy_ce = -(policies * log_policy).sum(dim=1).mean()
        value_wdl_ce = -(values * torch.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
        objective = network._learner.policy_loss_weight * policy_ce
        objective = objective + network._learner.value_loss_weight * value_wdl_ce
        weighted_loss = network._learner.expert_anchor_loss_weight * objective
    wdl_accuracy, q_mae = _value_metrics(value_logits, values)
    return weighted_loss, policy_ce, value_wdl_ce, wdl_accuracy, q_mae


def _batch_tensors(
    network: NetworkRuntime,
    batch: Lc0Batch,
    indices: slice,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        _tensor(network, batch.observations[indices]),
        _tensor(network, batch.valid_moves[indices]),
        _tensor(network, batch.policies[indices]),
        _tensor(network, batch.value_targets[indices]),
    )


def _tensor(network: NetworkRuntime, values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(values, device=network.device, dtype=torch.float32)


def _validate_batch(network: NetworkRuntime, batch: Lc0Batch) -> int:
    positions = len(batch.observations)
    expected_observations = (positions, network.board_x, network.board_y, network.board_z)
    if positions <= 0 or batch.observations.shape != expected_observations:
        raise ValueError(
            f"expert anchor observations must have shape {expected_observations}, got {batch.observations.shape}"
        )
    expected_policy = (positions, network.action_size)
    if batch.policies.shape != expected_policy or batch.valid_moves.shape != expected_policy:
        raise ValueError("expert anchor policy targets and legal masks do not match the network action space")
    if batch.value_targets.shape != (positions, 3):
        raise ValueError(f"expert anchor WDL targets must have shape {(positions, 3)}, got {batch.value_targets.shape}")
    return positions


def _value_metrics(value_logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        probabilities = torch.softmax(value_logits.detach().float(), dim=1)
        accuracy = (probabilities.argmax(dim=1) == targets.argmax(dim=1)).float().mean()
        support = torch.tensor((-1.0, 0.0, 1.0), device=value_logits.device)
        q_mae = ((probabilities @ support) - (targets @ support)).abs().mean()
    return accuracy, q_mae
