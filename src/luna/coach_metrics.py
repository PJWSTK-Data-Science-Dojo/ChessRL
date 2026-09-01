"""Iteration-level self-play, replay, and performance metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import chess
import numpy as np
import wandb

from luna.profiling import IterProfileStats
from luna.replay_buffer import Trajectory

if TYPE_CHECKING:
    from luna.coach import Coach

MetricValue = int | float


@dataclass
class _SelfPlaySummary:
    games: int = 0
    positions: int = 0
    truncated_games: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    unknown_terminations: int = 0
    policy_entropy_sum: float = 0.0
    truncation_bootstrap_abs_sum: float = 0.0
    guard_attempts: int = 0
    guard_interventions: int = 0
    guard_forced_fallbacks: int = 0
    guard_excluded_actions: int = 0
    terminations: dict[chess.Termination, int] = field(
        default_factory=lambda: {termination: 0 for termination in chess.Termination}
    )


def log_iteration_metrics(
    coach: Coach,
    iteration: int,
    trajectories: list[Trajectory],
    stats: IterProfileStats,
    optimizer_steps: int = 0,
) -> None:
    if wandb.run is None:
        return
    summary = _summarize_trajectories(trajectories)
    metrics: dict[str, MetricValue] = {
        "iteration": iteration,
        "replay_buffer_size": coach.replay.size,
    }
    metrics.update(_self_play_metrics(coach, summary, optimizer_steps))
    metrics.update(_termination_metrics(summary))
    metrics.update(_performance_metrics(coach, summary, stats, optimizer_steps))
    wandb.log(metrics)


def _summarize_trajectories(trajectories: list[Trajectory]) -> _SelfPlaySummary:
    summary = _SelfPlaySummary(games=len(trajectories))
    for trajectory in trajectories:
        summary.positions += trajectory.game_length
        summary.policy_entropy_sum += _policy_entropy(trajectory)
        summary.guard_attempts += trajectory.repetition_guard_attempts
        summary.guard_interventions += trajectory.repetition_guard_interventions
        summary.guard_forced_fallbacks += trajectory.repetition_guard_forced_fallbacks
        summary.guard_excluded_actions += trajectory.repetition_guard_excluded_actions
        _record_outcome(summary, trajectory)
    return summary


def _policy_entropy(trajectory: Trajectory) -> float:
    probabilities = trajectory.root_policies.astype(np.float32)
    positive = probabilities > 0.0
    return -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))


def _record_outcome(summary: _SelfPlaySummary, trajectory: Trajectory) -> None:
    if trajectory.truncated:
        summary.truncated_games += 1
        bootstrap_value = trajectory.truncation_bootstrap_value
        if bootstrap_value is None:
            raise RuntimeError("Truncated trajectory is missing its validated bootstrap value")
        summary.truncation_bootstrap_abs_sum += abs(bootstrap_value)
        return
    if trajectory.termination is None:
        summary.unknown_terminations += 1
    else:
        summary.terminations[trajectory.termination] += 1
    terminal_reward = float(trajectory.rewards[-1])
    if terminal_reward == 0.0:
        summary.draws += 1
    elif _white_reward(trajectory, terminal_reward) > 0.0:
        summary.white_wins += 1
    else:
        summary.black_wins += 1


def _white_reward(trajectory: Trajectory, terminal_reward: float) -> float:
    return terminal_reward if trajectory.game_length % 2 else -terminal_reward


def _fraction(count: int, total: int) -> float:
    return count / total if total else 0.0


def _mean(total_value: float, count: int) -> float:
    return total_value / count if count else 0.0


def _self_play_metrics(
    coach: Coach,
    summary: _SelfPlaySummary,
    optimizer_steps: int,
) -> dict[str, MetricValue]:
    decisive_games = summary.white_wins + summary.black_wins
    positions = summary.positions
    return {
        "selfplay/games": summary.games,
        "selfplay/positions": positions,
        "selfplay/avg_ply": _fraction(positions, summary.games),
        "selfplay/max_ply_fraction": _fraction(summary.truncated_games, summary.games),
        "selfplay/truncated_fraction": _fraction(summary.truncated_games, summary.games),
        "selfplay/truncation_bootstrap_mean_abs": _mean(
            summary.truncation_bootstrap_abs_sum,
            summary.truncated_games,
        ),
        "selfplay/decisive_fraction": _fraction(decisive_games, summary.games),
        "selfplay/draw_fraction": _fraction(summary.draws, summary.games),
        "selfplay/white_win_fraction": _fraction(summary.white_wins, summary.games),
        "selfplay/black_win_fraction": _fraction(summary.black_wins, summary.games),
        "selfplay/policy_entropy": summary.policy_entropy_sum / positions if positions else 0.0,
        "selfplay/replay_samples_per_new_position": (
            optimizer_steps * coach.nnet._learner.batch_size / positions if positions else 0.0
        ),
        "selfplay/repetition_guard_attempts": summary.guard_attempts,
        "selfplay/repetition_guard_interventions": summary.guard_interventions,
        "selfplay/repetition_guard_forced_fallbacks": summary.guard_forced_fallbacks,
        "selfplay/repetition_guard_excluded_actions": summary.guard_excluded_actions,
        "selfplay/repetition_guard_intervention_fraction": _fraction(summary.guard_interventions, positions),
        "selfplay/repetition_guard_attempt_fraction": _fraction(summary.guard_attempts, positions),
    }


def _termination_metrics(summary: _SelfPlaySummary) -> dict[str, MetricValue]:
    games = summary.games
    terminations = summary.terminations
    return {
        "selfplay/checkmate_fraction": _fraction(terminations[chess.Termination.CHECKMATE], games),
        "selfplay/threefold_repetition_fraction": _fraction(
            terminations[chess.Termination.THREEFOLD_REPETITION], games
        ),
        "selfplay/fivefold_repetition_fraction": _fraction(terminations[chess.Termination.FIVEFOLD_REPETITION], games),
        "selfplay/fifty_move_fraction": _fraction(terminations[chess.Termination.FIFTY_MOVES], games),
        "selfplay/seventyfive_move_fraction": _fraction(terminations[chess.Termination.SEVENTYFIVE_MOVES], games),
        "selfplay/stalemate_fraction": _fraction(terminations[chess.Termination.STALEMATE], games),
        "selfplay/insufficient_material_fraction": _fraction(
            terminations[chess.Termination.INSUFFICIENT_MATERIAL], games
        ),
        "selfplay/unknown_termination_fraction": _fraction(summary.unknown_terminations, games),
    }


def _performance_metrics(
    coach: Coach,
    summary: _SelfPlaySummary,
    stats: IterProfileStats,
    optimizer_steps: int,
) -> dict[str, MetricValue]:
    positions_per_second = summary.positions / stats.self_play_s if stats.self_play_s > 0.0 else 0.0
    return {
        "performance/self_play_seconds": stats.self_play_s,
        "performance/self_play_positions_per_second": positions_per_second,
        "performance/train_seconds": stats.train_s,
        "performance/iteration_seconds": stats.total_s,
        "replay/size": coach.replay.size,
        "replay/beta": coach.replay.beta,
        "replay/optimizer_steps": optimizer_steps,
        "replay/step_cap_reached": int(optimizer_steps > 0 and optimizer_steps == coach.run.train_steps_per_iter),
        "replay/target_samples_per_new_position": coach.run.target_replay_ratio or 0.0,
        "replay/warmup_positions": max(
            coach.nnet._learner.batch_size,
            coach.run.replay_warmup_positions,
        ),
    }
