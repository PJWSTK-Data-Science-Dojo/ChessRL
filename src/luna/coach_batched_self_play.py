"""Sliding-pool batched self-play orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Never

import chess
import numpy as np
from tqdm import tqdm

from luna.coach_self_play import (
    enables_threefold_claim,
    non_repetition_actions,
    select_self_play_action,
    self_play_exploration_enabled,
    trajectory_with_terminal_rewards,
)
from luna.game.chess_game import ChessGame
from luna.mcts import BatchedMCTS
from luna.mcts_batched_roots import SearchResult
from luna.profiling import SelfPlayMCTSTimings
from luna.replay_buffer import Trajectory

if TYPE_CHECKING:
    from luna.coach import Coach


@dataclass
class _GameSlot:
    board: chess.Board
    player: int = 1
    steps: int = 0
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    valid_moves: list[np.ndarray] = field(default_factory=list)
    guard_attempts: int = 0
    guard_interventions: int = 0
    guard_forced_fallbacks: int = 0
    guard_excluded_actions: int = 0
    alive: bool = True


@dataclass
class _BatchDecision:
    canonical_boards: list[chess.Board]
    results: list[SearchResult]
    actions: list[int]


@dataclass(frozen=True)
class _RetryPlan:
    rows: list[int]
    safe_actions: list[list[int]]


def execute_episodes_batched(coach: Coach, num_episodes: int, *, progress: bool = True) -> list[Trajectory]:
    """Run self-play games using batched parallel MCTS."""
    if num_episodes <= 0:
        return []
    if coach.run.profile:
        coach._profile_mcts_timings = SelfPlayMCTSTimings()
        coach._profile_sp_env_s = 0.0
    pool_size = min(coach.run.parallel_games, num_episodes)
    with tqdm(total=num_episodes, desc="Self Play (batched)", disable=not progress) as pbar:
        return run_self_play_pool(coach, num_episodes, pool_size, pbar)


def run_self_play_pool(coach: Coach, num_episodes: int, pool_size: int, pbar: tqdm[Never]) -> list[Trajectory]:
    """Keep the inference batch full by replacing each finished game immediately."""
    timings = coach._profile_mcts_timings if coach.run.profile else None
    search = BatchedMCTS(coach.game, coach.nnet, coach.run, timings=timings)
    slots = [_GameSlot(coach.game.get_init_board()) for _ in range(pool_size)]
    completed: list[Trajectory] = []
    while len(completed) < num_episodes:
        if not _run_pool_step(coach, search, slots, completed, num_episodes, pbar):
            break
    return completed


def _run_pool_step(
    coach: Coach,
    search: BatchedMCTS,
    slots: list[_GameSlot],
    completed: list[Trajectory],
    target_games: int,
    pbar: tqdm[Never],
) -> bool:
    environment_started_at = time.perf_counter() if coach.run.profile else None
    active_indices = [index for index, slot in enumerate(slots) if slot.alive]
    if not active_indices:
        _record_environment_time(coach, environment_started_at)
        return False
    batch = _search_active_roots(coach, search, slots, active_indices, environment_started_at)
    search_finished_at = time.perf_counter() if coach.run.profile else None
    _advance_active_roots(coach, slots, active_indices, batch, completed, target_games, pbar)
    _record_environment_time(coach, search_finished_at)
    return True


def _search_active_roots(
    coach: Coach,
    search: BatchedMCTS,
    slots: list[_GameSlot],
    active_indices: list[int],
    environment_started_at: float | None,
) -> _BatchDecision:
    exploration = [
        self_play_exploration_enabled(slots[index].board, slots[index].steps + 1, coach.run) for index in active_indices
    ]
    canonical_boards = [
        coach.game.get_canonical_form(slots[index].board, slots[index].player) for index in active_indices
    ]
    _record_environment_time(coach, environment_started_at)
    results = search.search_batch(canonical_boards, temp=1.0, add_exploration_noise=exploration)
    proposals = list(search.last_actions)
    actions = [
        select_self_play_action(
            coach.run,
            result[0],
            explore=exploration[row],
            gumbel_proposal=proposals[row],
        )
        for row, result in enumerate(results)
    ]
    decision = _BatchDecision(canonical_boards, results, actions)
    _retry_repetitions(coach, search, slots, active_indices, decision)
    return decision


def _retry_repetitions(
    coach: Coach,
    search: BatchedMCTS,
    slots: list[_GameSlot],
    active_indices: list[int],
    decision: _BatchDecision,
) -> None:
    if not coach.run.self_play_repetition_guard:
        return
    plan = _build_retry_plan(coach.game, slots, active_indices, decision)
    if not plan.rows:
        return
    retry_results = search.search_batch(
        [decision.canonical_boards[row] for row in plan.rows],
        temp=1.0,
        add_exploration_noise=[True] * len(plan.rows),
        allowed_root_actions=plan.safe_actions,
    )
    _apply_retry_results(coach, search, decision, plan.rows, retry_results)


def _build_retry_plan(
    game: ChessGame,
    slots: list[_GameSlot],
    active_indices: list[int],
    decision: _BatchDecision,
) -> _RetryPlan:
    rows: list[int] = []
    safe_actions_by_row: list[list[int]] = []
    for row, index in enumerate(active_indices):
        safe_actions = _safe_repetition_alternatives(game, slots[index], decision, row)
        if safe_actions:
            rows.append(row)
            safe_actions_by_row.append(safe_actions)
    return _RetryPlan(rows, safe_actions_by_row)


def _safe_repetition_alternatives(
    game: ChessGame,
    slot: _GameSlot,
    decision: _BatchDecision,
    row: int,
) -> list[int]:
    if not enables_threefold_claim(game, slot.board, slot.player, decision.actions[row]):
        return []
    slot.guard_attempts += 1
    valid_moves = decision.results[row][3]
    safe_actions = non_repetition_actions(game, slot.board, slot.player, valid_moves)
    if not safe_actions:
        slot.guard_forced_fallbacks += 1
        return []
    slot.guard_interventions += 1
    slot.guard_excluded_actions += int(np.count_nonzero(valid_moves)) - len(safe_actions)
    return safe_actions


def _apply_retry_results(
    coach: Coach,
    search: BatchedMCTS,
    decision: _BatchDecision,
    retry_rows: list[int],
    retry_results: list[SearchResult],
) -> None:
    proposals = list(search.last_actions)
    for retry_index, row in enumerate(retry_rows):
        retry_policy, retry_value, _retry_observation, _retry_valid = retry_results[retry_index]
        _old_policy, _old_value, observation, valid_moves = decision.results[row]
        decision.results[row] = (retry_policy, retry_value, observation, valid_moves)
        decision.actions[row] = select_self_play_action(
            coach.run,
            retry_policy,
            explore=True,
            gumbel_proposal=proposals[retry_index],
        )


def _advance_active_roots(
    coach: Coach,
    slots: list[_GameSlot],
    active_indices: list[int],
    decision: _BatchDecision,
    completed: list[Trajectory],
    target_games: int,
    pbar: tqdm[Never],
) -> None:
    results_by_index = dict(zip(active_indices, decision.results, strict=True))
    for row, index in enumerate(active_indices):
        trajectory = _advance_slot(coach, slots[index], decision.actions[row], results_by_index[index])
        if trajectory is not None:
            _record_completion(coach.game, slots, index, trajectory, completed, target_games, pbar)


def _advance_slot(
    coach: Coach,
    slot: _GameSlot,
    action: int,
    result: SearchResult,
) -> Trajectory | None:
    policy, root_value, observation, valid_moves = result
    slot.steps += 1
    slot.observations.append(observation)
    slot.policies.append(policy)
    slot.values.append(root_value)
    slot.valid_moves.append(valid_moves)
    slot.player = coach.game.push_action(slot.board, slot.player, action)
    slot.actions.append(action)
    outcome = coach.game.get_game_outcome(slot.board, slot.player)
    if outcome is not None:
        return _terminal_slot_trajectory(coach.game, slot, outcome)
    if coach.run.max_ply is not None and slot.steps >= coach.run.max_ply:
        return _slot_trajectory(slot, terminal_value=0.0, truncated=True)
    return None


def _terminal_slot_trajectory(game: ChessGame, slot: _GameSlot, outcome: float) -> Trajectory:
    terminal_outcome = slot.board.outcome(claim_draw=game.claim_draw)
    if terminal_outcome is None:
        raise RuntimeError("A terminal self-play state has no chess outcome")
    return _slot_trajectory(slot, terminal_value=outcome, termination=terminal_outcome.termination)


def _slot_trajectory(
    slot: _GameSlot,
    *,
    terminal_value: float,
    truncated: bool = False,
    termination: chess.Termination | None = None,
) -> Trajectory:
    return trajectory_with_terminal_rewards(
        slot.observations,
        slot.actions,
        slot.policies,
        slot.values,
        slot.valid_moves,
        terminal_value,
        truncated,
        termination,
        slot.guard_attempts,
        slot.guard_interventions,
        slot.guard_forced_fallbacks,
        slot.guard_excluded_actions,
    )


def _record_completion(
    game: ChessGame,
    slots: list[_GameSlot],
    index: int,
    trajectory: Trajectory,
    completed: list[Trajectory],
    target_games: int,
    pbar: tqdm[Never],
) -> None:
    if len(completed) < target_games:
        completed.append(trajectory)
        pbar.update(1)
    if len(completed) >= target_games:
        slots[index].alive = False
    else:
        slots[index] = _GameSlot(game.get_init_board())


def _record_environment_time(coach: Coach, started_at: float | None) -> None:
    if started_at is not None:
        coach._profile_sp_env_s += time.perf_counter() - started_at
