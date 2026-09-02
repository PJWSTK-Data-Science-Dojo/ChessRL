"""Learner progress aggregation and experiment reporting."""

from __future__ import annotations

import time

import numpy as np
import wandb
from loguru import logger

from luna.network_training_types import OptimizerOutcome, StepAccumulation, TrainingMeters
from luna.network_types import NetworkRuntime, PreparedBatch
from luna.replay_buffer import PrioritizedReplayBuffer


def record_successful_step(
    network: NetworkRuntime,
    replay: PrioritizedReplayBuffer,
    prepared: PreparedBatch,
    accumulation: StepAccumulation,
    outcome: OptimizerOutcome,
    meters: TrainingMeters,
    started_at: float,
) -> None:
    _update_replay_priorities(replay, accumulation)
    batch_size = network._learner.batch_size
    meters.total.update(float(accumulation.total.item()), batch_size)
    meters.policy.update(float(accumulation.policy.item()), batch_size)
    meters.value.update(float(accumulation.value.item()), batch_size)
    meters.reward.update(float(accumulation.reward.item()), batch_size)
    meters.consistency.update(float(accumulation.consistency.item()), batch_size)
    meters.reconstruction.update(float(accumulation.reconstruction.item()), batch_size)
    _update_expert_anchor_meters(accumulation, meters)
    meters.step_time.update(time.time() - started_at)
    _update_gradient_meters(network, meters, outcome.gradient_norm)
    meters.reanalysis_samples.update(float(prepared.reanalysis.selected_samples))
    meters.reanalysis_positions.update(float(prepared.reanalysis.searched_positions))
    meters.reanalysis_seconds.update(prepared.reanalysis.duration_seconds)


def _update_expert_anchor_meters(accumulation: StepAccumulation, meters: TrainingMeters) -> None:
    expert = accumulation.expert_anchor
    if expert is None:
        return
    positions = expert.positions
    meters.expert_anchor.update(float(expert.weighted_loss.item()), positions)
    meters.expert_policy_ce.update(float(expert.policy_cross_entropy.item()), positions)
    meters.expert_value_wdl_ce.update(float(expert.value_wdl_cross_entropy.item()), positions)
    meters.expert_wdl_accuracy.update(float(expert.wdl_accuracy.item()), positions)
    meters.expert_q_mae.update(float(expert.q_mae.item()), positions)
    meters.expert_positions.update(float(positions))


def _update_replay_priorities(replay: PrioritizedReplayBuffer, accumulation: StepAccumulation) -> None:
    indices = [index for microbatch in accumulation.tree_indices for index in microbatch]
    replay.update_priorities(indices, np.concatenate(accumulation.priority_errors))


def _update_gradient_meters(
    network: NetworkRuntime,
    meters: TrainingMeters,
    gradient_norm: float,
) -> None:
    coefficient = min(1.0, network._learner.grad_clip_norm / max(gradient_norm, 1e-12))
    meters.grad_norm_preclip.update(gradient_norm)
    meters.grad_norm_postclip.update(gradient_norm * coefficient)
    meters.grad_clip_coefficient.update(coefficient)
    meters.grad_clip_fraction.update(float(coefficient < 1.0))


def report_training(
    network: NetworkRuntime,
    step: int,
    total_steps: int,
    learning_rate: float,
    meters: TrainingMeters,
    latent_health: dict[str, float],
) -> None:
    logger.info(
        "(step {}/{}) {:.3f}s lr={:.1e} | loss={:.4f} pi={:.4f} v={:.4f} r={:.4f} c={:.4f} "
        "reconstruct={:.4f} expert={:.4f}",
        step,
        total_steps,
        meters.step_time.avg,
        learning_rate,
        meters.total.avg,
        meters.policy.avg,
        meters.value.avg,
        meters.reward.avg,
        meters.consistency.avg,
        meters.reconstruction.avg,
        meters.expert_anchor.avg,
    )
    if wandb.run is not None:
        wandb.log(_wandb_metrics(network, learning_rate, meters, latent_health))
    root_diversity = latent_health.get("train/latent_root_batch_feature_std")
    if root_diversity is not None:
        network._check_representation_diversity(root_diversity, network._global_step)


def _wandb_metrics(
    network: NetworkRuntime,
    learning_rate: float,
    meters: TrainingMeters,
    latent_health: dict[str, float],
) -> dict[str, float | int]:
    learner = network._learner
    batch_size = learner.batch_size
    return {
        "train/loss_total": meters.total.avg,
        "train/loss_policy": meters.policy.avg,
        "train/loss_value": meters.value.avg,
        "train/loss_reward": meters.reward.avg,
        "train/loss_consistency": meters.consistency.avg,
        "train/loss_consistency_weighted": learner.consistency_loss_weight * meters.consistency.avg,
        "train/loss_reconstruction": meters.reconstruction.avg,
        "train/loss_reconstruction_weighted": learner.reconstruction_loss_weight * meters.reconstruction.avg,
        "train/loss_expert_anchor": meters.expert_anchor.avg,
        "train/expert_anchor_policy_ce": meters.expert_policy_ce.avg,
        "train/expert_anchor_value_wdl_ce": meters.expert_value_wdl_ce.avg,
        "train/expert_anchor_wdl_accuracy": meters.expert_wdl_accuracy.avg,
        "train/expert_anchor_q_mae": meters.expert_q_mae.avg,
        "train/expert_anchor_positions": meters.expert_positions.avg,
        "train/lr": learning_rate,
        "train/grad_norm": meters.grad_norm_preclip.avg,
        "train/grad_norm_preclip": meters.grad_norm_preclip.avg,
        "train/grad_norm_postclip": meters.grad_norm_postclip.avg,
        "train/grad_clip_coefficient": meters.grad_clip_coefficient.avg,
        "train/grad_clip_fraction": meters.grad_clip_fraction.avg,
        "train/reanalysis_selected_samples": meters.reanalysis_samples.avg,
        "train/reanalysis_selected_fraction": meters.reanalysis_samples.avg / batch_size,
        "train/reanalysis_searched_positions": meters.reanalysis_positions.avg,
        "train/reanalysis_seconds": meters.reanalysis_seconds.avg,
        "train/step_time": meters.step_time.avg,
        "train/samples_per_second": batch_size / meters.step_time.avg if meters.step_time.avg > 0.0 else 0.0,
        "global_step": network._global_step,
        **latent_health,
    }
