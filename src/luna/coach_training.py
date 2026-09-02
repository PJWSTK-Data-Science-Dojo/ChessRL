from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from luna.config import TrainingRunConfig
from luna.game.stockfish_eval import validate_ladder_configuration, validate_stockfish_configuration
from luna.profiling import IterProfileStats, write_iter_summaries_json
from luna.replay_buffer import Trajectory
from luna.replay_persistence import load_replay_snapshot, save_replay_snapshot
from luna.self_play_actors import SelfPlayActorPool, derive_actor_seed
from luna.self_play_worker import seed_self_play_rng

if TYPE_CHECKING:
    from luna.coach import Coach


@dataclass(frozen=True)
class _EvaluationSchedule:
    current_iteration: int
    start_iteration: int
    fixed_due: bool
    ladder_due: bool


def optimizer_steps_for_positions(
    run: TrainingRunConfig,
    *,
    positions: int,
    batch_size: int,
) -> int:
    """Choose an update count without amplifying short self-play trajectories."""
    if positions <= 0:
        return 0
    if run.target_replay_ratio is None:
        return run.train_steps_per_iter
    desired_steps = max(1, math.floor(run.target_replay_ratio * positions / batch_size + 0.5))
    return min(run.train_steps_per_iter, desired_steps)


def learn(coach: Coach) -> None:
    schedule = _prepare_training(coach)
    if schedule.start_iteration > coach.run.num_iters:
        _log_completed_training(coach, schedule.current_iteration)
        return
    if schedule.start_iteration > 1:
        logger.info("Resuming training at iteration {} of {}", schedule.start_iteration, coach.run.num_iters)
    _run_with_self_play_actors(coach, schedule.start_iteration)


def _prepare_training(coach: Coach) -> _EvaluationSchedule:
    coach._assert_checkpoint_target()
    coach._assert_checkpoint_lineage()
    _restore_replay(coach)
    current_iteration = coach.nnet._trainer_iteration
    start_iteration = current_iteration + 1
    schedule = _evaluation_schedule(coach.run, current_iteration)
    _validate_due_evaluations(coach.run, schedule)
    coach._initialize_external_evaluation_sidecars(current_iteration)
    if start_iteration <= coach.run.num_iters or schedule.fixed_due or schedule.ladder_due:
        coach.nnet.warmup_mcts_inference(coach.game)
    coach._reconcile_current_evaluations(current_iteration)
    return schedule


def _restore_replay(coach: Coach) -> None:
    if not coach._restore_replay_on_start:
        return
    source_checkpoint = coach.nnet._loaded_checkpoint_path
    if source_checkpoint is None:
        raise RuntimeError("Replay resume requires the loaded checkpoint path")
    expected_iteration = coach.nnet._trainer_iteration
    try:
        restored_iteration = load_replay_snapshot(coach.replay, source_checkpoint.parent, expected_iteration)
    except FileNotFoundError:
        logger.warning(
            "No replay snapshot for checkpoint iteration {}; replay warm-up starts empty.", expected_iteration
        )
    else:
        log = logger.warning if restored_iteration < expected_iteration else logger.info
        log(
            "Restored {} replay positions from iteration {} for checkpoint iteration {} (lag={})",
            coach.replay.size,
            restored_iteration,
            expected_iteration,
            expected_iteration - restored_iteration,
        )
    coach._restore_replay_on_start = False


def _evaluation_schedule(run: TrainingRunConfig, current_iteration: int) -> _EvaluationSchedule:
    start_iteration = current_iteration + 1
    fixed_due = _evaluation_due_now(current_iteration, run.stockfish_eval_every)
    ladder_due = _evaluation_due_now(current_iteration, run.ladder_eval_every)
    return _EvaluationSchedule(current_iteration, start_iteration, fixed_due, ladder_due)


def _evaluation_due_now(current_iteration: int, interval: int) -> bool:
    return interval > 0 and current_iteration > 0 and current_iteration % interval == 0


def _next_evaluation(start_iteration: int, interval: int, last_iteration: int) -> int:
    if interval <= 0:
        return last_iteration + 1
    return ((start_iteration + interval - 1) // interval) * interval


def _validate_due_evaluations(run: TrainingRunConfig, schedule: _EvaluationSchedule) -> None:
    next_fixed = _next_evaluation(schedule.start_iteration, run.stockfish_eval_every, run.num_iters)
    if schedule.fixed_due or next_fixed <= run.num_iters:
        validate_stockfish_configuration(run)
    next_ladder = _next_evaluation(schedule.start_iteration, run.ladder_eval_every, run.num_iters)
    if schedule.ladder_due or next_ladder <= run.num_iters:
        validate_ladder_configuration(run)


def _log_completed_training(coach: Coach, current_iteration: int) -> None:
    logger.info(
        "Checkpoint is already at iteration {}; requested total is {}. Nothing to train.",
        current_iteration,
        coach.run.num_iters,
    )


def _run_with_self_play_actors(coach: Coach, start_iteration: int) -> None:
    worker_count = min(coach.run.self_play_workers, coach.run.num_episodes)
    if worker_count <= 1:
        coach._learn_iterations(start_iteration, actor_pool=None)
        return
    logger.info(
        "Starting {} persistent self-play actors with up to {} batched games each",
        worker_count,
        coach.run.parallel_games,
    )
    with SelfPlayActorPool(
        coach.nnet,
        coach.run,
        worker_count=worker_count,
        base_seed=coach._seed,
    ) as actor_pool:
        coach._learn_iterations(start_iteration, actor_pool=actor_pool)


def learn_iterations(
    coach: Coach,
    start_iteration: int,
    actor_pool: SelfPlayActorPool | None,
) -> None:
    profile_rows: list[IterProfileStats] = []
    _initialize_profiling(coach)
    for iteration in range(start_iteration, coach.run.num_iters + 1):
        _run_iteration(coach, iteration, actor_pool, profile_rows)
    _write_profile_summary(coach, profile_rows)


def _initialize_profiling(coach: Coach) -> None:
    if not coach.run.profile:
        return
    os.makedirs(coach.run.profile_dir, exist_ok=True)
    logger.info(
        "Profiling enabled: dir={} | Kineto steps: iter {} x {} | chrome={} tb_logdir={} with_stack={}",
        os.path.abspath(coach.run.profile_dir),
        coach.run.profile_torch_iter,
        coach.run.profile_torch_steps,
        coach.run.profile_export_chrome,
        coach.run.profile_tensorboard_logdir,
        coach.run.profile_with_stack,
    )


def _run_iteration(
    coach: Coach,
    iteration: int,
    actor_pool: SelfPlayActorPool | None,
    profile_rows: list[IterProfileStats],
) -> None:
    logger.info("Starting Iter #{} ...", iteration)
    iteration_started_at = time.perf_counter()
    stats = IterProfileStats(iter_index=iteration)
    trajectories = _collect_self_play(coach, iteration, actor_pool, stats)
    _save_trajectories(coach, trajectories, stats)
    if not _replay_ready(coach):
        _finish_warmup_iteration(coach, iteration, trajectories, stats, iteration_started_at, profile_rows)
        return
    optimizer_steps = _train_from_replay(coach, iteration, trajectories, stats)
    _finish_trained_iteration(
        coach,
        iteration,
        trajectories,
        stats,
        iteration_started_at,
        optimizer_steps,
        profile_rows,
    )


def _collect_self_play(
    coach: Coach,
    iteration: int,
    actor_pool: SelfPlayActorPool | None,
    stats: IterProfileStats,
) -> list[Trajectory]:
    started_at = time.perf_counter()
    if actor_pool is None:
        seed_self_play_rng(derive_actor_seed(coach._seed, actor_id=0, generation=iteration))
        trajectories = coach.execute_episodes_batched(coach.run.num_episodes)
    else:
        trajectories = actor_pool.collect(coach.run.num_episodes, generation=iteration)
    stats.self_play_s = time.perf_counter() - started_at
    _record_mcts_profile(coach, stats)
    return trajectories


def _record_mcts_profile(coach: Coach, stats: IterProfileStats) -> None:
    if not coach.run.profile or coach._profile_mcts_timings is None:
        return
    timings = coach._profile_mcts_timings
    stats.self_play_env_s = coach._profile_sp_env_s
    stats.self_play_mcts_encode_s = timings.encode_s
    stats.self_play_mcts_initial_inf_s = timings.initial_inf_s
    stats.self_play_mcts_selection_s = timings.selection_s
    stats.self_play_mcts_recurrent_inf_s = timings.recurrent_inf_s
    stats.self_play_mcts_expand_backup_s = timings.expand_backup_s
    stats.self_play_mcts_finalize_s = timings.finalize_s
    stats.self_play_search_batch_calls = timings.search_batch_calls


def _save_trajectories(coach: Coach, trajectories: list[Trajectory], stats: IterProfileStats) -> None:
    started_at = time.perf_counter()
    for trajectory in trajectories:
        coach.replay.save_trajectory(trajectory)
    stats.replay_save_s = time.perf_counter() - started_at


def _replay_ready(coach: Coach) -> bool:
    batch_size = coach.nnet._learner.batch_size
    replay_warmup = max(batch_size, coach.run.replay_warmup_positions)
    if coach.replay.size >= replay_warmup:
        return True
    logger.warning(
        "Replay buffer warm-up ({}/{} positions), skipping training.",
        coach.replay.size,
        replay_warmup,
    )
    return False


def _finish_warmup_iteration(
    coach: Coach,
    iteration: int,
    trajectories: list[Trajectory],
    stats: IterProfileStats,
    iteration_started_at: float,
    profile_rows: list[IterProfileStats],
) -> None:
    started_at = time.perf_counter()
    _publish_iteration_state(coach, iteration)
    stats.checkpoint_publish_s = time.perf_counter() - started_at
    stats.total_s = time.perf_counter() - iteration_started_at
    coach._log_iteration_metrics(iteration, trajectories, stats)
    coach._reconcile_current_evaluations(iteration)
    _append_profile_row(coach, stats, profile_rows)


def _train_from_replay(
    coach: Coach,
    iteration: int,
    trajectories: list[Trajectory],
    stats: IterProfileStats,
) -> int:
    profile_iteration = _training_profile_enabled(coach, iteration)
    _warn_missing_profile_output(coach, iteration)
    new_positions = sum(trajectory.game_length for trajectory in trajectories)
    optimizer_steps = optimizer_steps_for_positions(
        coach.run,
        positions=new_positions,
        batch_size=coach.nnet._learner.batch_size,
    )
    configure_replay_beta_annealing(coach, iteration, optimizer_steps)
    _run_optimizer_steps(coach, iteration, optimizer_steps, new_positions, profile_iteration, stats)
    return optimizer_steps


def _training_profile_enabled(coach: Coach, iteration: int) -> bool:
    return (
        coach.run.profile
        and coach.run.profile_torch_steps > 0
        and iteration == coach.run.profile_torch_iter
        and (coach.run.profile_export_chrome or bool(coach.run.profile_tensorboard_logdir))
    )


def _warn_missing_profile_output(coach: Coach, iteration: int) -> None:
    profile_due = coach.run.profile and coach.run.profile_torch_steps > 0 and iteration == coach.run.profile_torch_iter
    if profile_due and not (coach.run.profile_export_chrome or coach.run.profile_tensorboard_logdir):
        logger.warning(
            "profile_torch_steps>0 but both profile_export_chrome=False and no "
            "profile_tensorboard_logdir — no Kineto export will be produced."
        )


def _run_optimizer_steps(
    coach: Coach,
    iteration: int,
    optimizer_steps: int,
    new_positions: int,
    profile_iteration: bool,
    stats: IterProfileStats,
) -> None:
    logger.info(
        "Training from replay buffer ({} positions) for {} optimizer steps (new positions={}) ...",
        coach.replay.size,
        optimizer_steps,
        new_positions,
    )
    started_at = time.perf_counter()
    loss_info = coach.nnet.train_ezv2(
        coach.replay,
        steps=optimizer_steps,
        total_train_steps=_lr_schedule_total_steps(coach, iteration),
        discount=coach.run.discount,
        mcts_for_reanalyze=replace(coach.run, search_contempt_visit_limit=None),
        expert_anchor=coach.expert_anchor,
        torch_profile_steps=coach.run.profile_torch_steps if profile_iteration else 0,
        torch_profile_dir=coach.run.profile_dir if profile_iteration else None,
        torch_profile_iter=iteration,
        torch_profile_export_chrome=coach.run.profile_export_chrome,
        torch_profile_tensorboard_dir=(coach.run.profile_tensorboard_logdir if profile_iteration else None),
        torch_profile_with_stack=coach.run.profile_with_stack,
    )
    stats.train_s = time.perf_counter() - started_at
    logger.info("Training done: {}", loss_info)


def _lr_schedule_total_steps(coach: Coach, iteration: int) -> int:
    persisted_steps = coach.nnet._lr_schedule_total_steps
    if persisted_steps > 0:
        return persisted_steps
    if coach.run.lr_schedule_total_steps is not None:
        return coach.run.lr_schedule_total_steps
    remaining_iterations = coach.run.num_iters - iteration + 1
    return coach.nnet._global_step + remaining_iterations * coach.run.train_steps_per_iter


def _finish_trained_iteration(
    coach: Coach,
    iteration: int,
    trajectories: list[Trajectory],
    stats: IterProfileStats,
    iteration_started_at: float,
    optimizer_steps: int,
    profile_rows: list[IterProfileStats],
) -> None:
    started_at = time.perf_counter()
    _publish_iteration_state(coach, iteration)
    stats.checkpoint_publish_s = time.perf_counter() - started_at
    stats.total_s = time.perf_counter() - iteration_started_at
    coach._log_iteration_metrics(iteration, trajectories, stats, optimizer_steps=optimizer_steps)
    coach._reconcile_current_evaluations(iteration)
    _append_profile_row(coach, stats, profile_rows)


def _publish_iteration_state(coach: Coach, iteration: int) -> None:
    coach._publish_checkpoint(iteration)
    if not coach._checkpoint_dir_usable():
        return
    path = save_replay_snapshot(coach.replay, coach.run.checkpoint, iteration)
    logger.info("Published replay snapshot {} with {} positions", path, coach.replay.size)


def _append_profile_row(
    coach: Coach,
    stats: IterProfileStats,
    profile_rows: list[IterProfileStats],
) -> None:
    if coach.run.profile:
        profile_rows.append(stats)
        logger.info("\n{}\n", stats.to_log_lines())


def _write_profile_summary(coach: Coach, profile_rows: list[IterProfileStats]) -> None:
    if not coach.run.profile or not profile_rows:
        return
    summary_path = Path(coach.run.profile_dir) / coach.run.profile_summary_json
    write_iter_summaries_json(str(summary_path), profile_rows)
    logger.info("Wrote aggregated phase timings to {}", summary_path.resolve())


def configure_replay_beta_annealing(coach: Coach, iteration: int, optimizer_steps: int) -> None:
    remaining_iterations = coach.run.num_iters - iteration + 1
    remaining_sample_calls = remaining_iterations * optimizer_steps
    coach.replay.configure_beta_annealing(remaining_sample_calls)
    logger.info("PER beta annealing: iteration {}, {} sample calls", iteration, remaining_sample_calls)
