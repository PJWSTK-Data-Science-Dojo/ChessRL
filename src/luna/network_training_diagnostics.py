"""Diagnostics collected from EfficientZeroV2 training batches."""

from __future__ import annotations

import torch

from luna.network_training_types import Microbatch, RootState, TrainingFunctions, TrainingSettings, UnrollState
from luna.network_types import NetworkRuntime


def training_diagnostics(
    network: NetworkRuntime,
    batch: Microbatch,
    root: RootState,
    unroll: UnrollState,
    settings: TrainingSettings,
    functions: TrainingFunctions,
) -> dict[str, float]:
    metrics = functions.raw_latent_metrics("root", root.latent)
    if settings.unroll == 0:
        _add_target_metrics(metrics, batch, settings)
        _add_root_reconstruction_metrics(metrics, root, functions)
        return metrics

    active_dynamics = batch.unroll_mask[:, -1].bool()
    if bool(active_dynamics.any()):
        metrics.update(functions.raw_latent_metrics("predicted", unroll.next_latent[active_dynamics]))
    _add_target_metrics(metrics, batch, settings)
    active_consistency = batch.consistency_mask[:, -1].bool()
    _add_consistency_metrics(metrics, network, root, unroll, active_consistency, functions)
    _add_root_reconstruction_metrics(metrics, root, functions)
    _add_unroll_reconstruction_metrics(metrics, unroll, active_consistency, functions)
    return metrics


def _add_consistency_metrics(
    metrics: dict[str, float],
    network: NetworkRuntime,
    root: RootState,
    unroll: UnrollState,
    active: torch.Tensor,
    functions: TrainingFunctions,
) -> None:
    if root.target_latents is None or not bool(active.any()):
        return
    metrics.update(
        functions.latent_metrics(
            network.nnet.simsiam,
            unroll.next_latent[active],
            root.target_latents[:, -1][active],
        )
    )


def _add_root_reconstruction_metrics(
    metrics: dict[str, float],
    root: RootState,
    functions: TrainingFunctions,
) -> None:
    if root.reconstruction_logits is None or root.reconstruction_target is None:
        return
    metrics.update(functions.reconstruction_metrics("root", root.reconstruction_logits, root.reconstruction_target))


def _add_unroll_reconstruction_metrics(
    metrics: dict[str, float],
    unroll: UnrollState,
    active: torch.Tensor,
    functions: TrainingFunctions,
) -> None:
    if unroll.reconstruction_logits is None or unroll.reconstruction_target is None or not bool(active.any()):
        return
    metrics.update(
        functions.reconstruction_metrics(
            "predicted",
            unroll.reconstruction_logits[active],
            unroll.reconstruction_target[active],
        )
    )


def _add_target_metrics(
    metrics: dict[str, float],
    batch: Microbatch,
    settings: TrainingSettings,
) -> None:
    active_values = batch.value_mask.bool()
    count = active_values.sum().clamp(min=1)
    absolute_values = batch.target_values.abs()
    active_consistency = _active_consistency(batch, settings)
    metrics["train/value_target_nonzero_fraction"] = float(
        ((absolute_values > 1e-6) & active_values).sum().item() / count.item()
    )
    metrics["train/value_target_fractional_fraction"] = float(
        (((absolute_values > 1e-6) & (absolute_values < 1.0 - 1e-6)) & active_values).sum().item() / count.item()
    )
    metrics["train/value_target_mean_abs"] = float((absolute_values * batch.value_mask).sum().item() / count.item())
    consistency_fraction = float(batch.consistency_mask.mean().item()) if settings.unroll > 0 else 0.0
    metrics["train/next_observation_active_fraction"] = consistency_fraction
    metrics["train/next_observation_active_samples"] = float(active_consistency.sum().item())
    metrics["train/consistency_objective_enabled"] = float(settings.consistency_enabled)
    metrics["train/consistency_active_fraction"] = consistency_fraction
    metrics["train/latent_health_active_samples"] = float(active_consistency.sum().item())


def _active_consistency(batch: Microbatch, settings: TrainingSettings) -> torch.Tensor:
    if settings.unroll > 0:
        return batch.consistency_mask[:, -1].bool()
    return torch.zeros(batch.target_values.shape[0], dtype=torch.bool, device=batch.target_values.device)
