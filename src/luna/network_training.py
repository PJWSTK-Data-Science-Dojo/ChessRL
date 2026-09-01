"""EfficientZeroV2 learner-loop orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

from luna.config import MCTSParams
from luna.network_training_batches import TrainingBatchSource
from luna.network_training_forward import run_microbatches
from luna.network_training_metrics import record_successful_step, report_training
from luna.network_training_optimizer import apply_optimizer_update, set_learning_rate
from luna.network_training_profiler import TrainingProfilerConfig, start_profiler
from luna.network_training_types import TrainingFunctions, TrainingMeters, TrainingSettings
from luna.network_types import NetworkRuntime, PreparedBatch
from luna.replay_buffer import PrioritizedReplayBuffer


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    steps: int
    total_train_steps: int
    discount: float | None
    mcts_for_reanalysis: MCTSParams | None
    profiler: TrainingProfilerConfig


def train_ezv2(
    network: NetworkRuntime,
    replay: PrioritizedReplayBuffer,
    request: TrainingRequest,
    functions: TrainingFunctions,
) -> dict[str, float]:
    settings = _training_settings(network, replay, request)
    network.nnet.train()
    _validate_reconstruction_head(network)
    profiler = start_profiler(network, request.profiler)
    meters = TrainingMeters()
    batches = TrainingBatchSource(network, replay, settings, request.mcts_for_reanalysis)
    batches.start()
    network.optimizer.zero_grad(set_to_none=True)
    completed_steps = 0
    consecutive_amp_skips = 0
    retry_batch: PreparedBatch | None = None
    try:
        while completed_steps < settings.steps:
            step = completed_steps + 1
            training_step = network._global_step + 1
            learning_rate = network._lr_schedule(training_step, settings.learning_rate_horizon)
            previous_rates = set_learning_rate(network, learning_rate)
            started_at = time.time()
            prepared = batches.get(training_step, retry_batch)
            batches.schedule_next(training_step, step)
            accumulation = run_microbatches(network, prepared, step, settings, functions)
            outcome = apply_optimizer_update(network, accumulation, previous_rates, functions)
            if outcome.gradient_overflow:
                consecutive_amp_skips += 1
                retry_batch = prepared
                _report_overflow(
                    step,
                    settings.steps,
                    consecutive_amp_skips,
                    outcome.previous_scale,
                    outcome.current_scale,
                    functions.maximum_amp_skips,
                )
                continue
            retry_batch = None
            consecutive_amp_skips = 0
            network._global_step = training_step
            completed_steps += 1
            record_successful_step(network, replay, prepared, accumulation, outcome, meters, started_at)
            if settings.should_report(step):
                report_training(network, step, settings.steps, learning_rate, meters, accumulation.latent_health)
            if profiler is not None:
                profiler.step()
    finally:
        if profiler is not None:
            profiler.stop()
    return meters.losses()


def _training_settings(
    network: NetworkRuntime,
    replay: PrioritizedReplayBuffer,
    request: TrainingRequest,
) -> TrainingSettings:
    learner = network._learner
    network._validate_training_inputs(
        replay,
        request.steps,
        learner.batch_size,
        learner.unroll_steps,
        learner.td_steps,
    )
    horizon = network._resolve_lr_schedule_total(request.total_train_steps, request.steps)
    discount = request.discount if request.discount is not None else learner.discount
    return TrainingSettings(
        steps=request.steps,
        batch_size=learner.batch_size,
        micro_batch_size=learner.batch_size // learner.grad_accum_steps,
        unroll=learner.unroll_steps,
        support=learner.support_size,
        gradient_accumulation=learner.grad_accum_steps,
        learning_rate_horizon=horizon,
        discount=discount,
        consistency_enabled=learner.consistency_loss_weight > 0.0,
    )


def _validate_reconstruction_head(network: NetworkRuntime) -> None:
    if network._learner.reconstruction_loss_weight > 0.0 and network.nnet.piece_reconstruction is None:
        raise RuntimeError("The configured reconstruction objective has no reconstruction head")


def _report_overflow(
    step: int,
    total_steps: int,
    consecutive_skips: int,
    previous_scale: float,
    current_scale: float,
    maximum_skips: int,
) -> None:
    if consecutive_skips >= maximum_skips:
        raise RuntimeError(
            f"Mixed-precision training stopped after {consecutive_skips} consecutive non-finite gradient updates."
        )
    logger.warning(
        "Retrying optimizer step {}/{} after mixed-precision overflow "
        "(loss scale {:.1f} -> {:.1f}, consecutive skips {}).",
        step,
        total_steps,
        previous_scale,
        current_scale,
        consecutive_skips,
    )
