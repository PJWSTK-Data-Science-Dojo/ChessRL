"""Replay sampling, target collation, and optional search reanalysis."""

from __future__ import annotations

import time
from dataclasses import replace

import chess
import numpy as np

from luna.config import MCTSParams
from luna.network_types import NetworkRuntime, PreparedBatch, ReanalysisBatchStats
from luna.replay_buffer import PrioritizedReplayBuffer, Trajectory
from luna.targets import build_unroll_targets, collate_batch


def async_batch_prefetch(network: NetworkRuntime, upcoming_steps: int = 0) -> bool:
    if network._prefetch_executor is None:
        return False
    learner = network._learner
    if learner.reanalyze_mcts_sims <= 0 or learner.reanalyze_prob <= 0:
        return True
    return network._global_step + max(0, upcoming_steps) < learner.reanalyze_start_step


def prepare_batch(
    network: NetworkRuntime,
    replay: PrioritizedReplayBuffer,
    batch_size: int,
    unroll: int,
    td_steps: int,
    discount: float,
    training_step: int,
    mcts_params: MCTSParams | None,
) -> PreparedBatch:
    batch, importance_weights, tree_indices = replay.sample(batch_size, unroll)
    roots, policies, requests, boards = _collect_reanalysis_requests(network, batch, unroll, training_step)
    started_at = time.perf_counter() if boards else None
    if boards:
        _apply_reanalysis(network, mcts_params, boards, requests, roots, policies)
    targets = [
        build_unroll_targets(
            trajectory,
            position,
            unroll,
            td_steps,
            discount,
            root_value_override=roots[index],
            policy_override=policies[index],
            train_value_on_truncated=network._learner.train_value_on_truncated,
        )
        for index, (trajectory, position) in enumerate(batch)
    ]
    duration = time.perf_counter() - started_at if started_at is not None else 0.0
    return PreparedBatch(
        collate_batch(targets),
        importance_weights,
        tree_indices,
        ReanalysisBatchStats(
            selected_samples=sum(override is not None for override in roots),
            searched_positions=len(boards),
            duration_seconds=duration,
        ),
    )


def _collect_reanalysis_requests(
    network: NetworkRuntime,
    batch: list[tuple[Trajectory, int]],
    unroll: int,
    training_step: int,
) -> tuple[
    list[dict[int, float] | None],
    list[dict[int, np.ndarray] | None],
    list[tuple[int, int]],
    list[chess.Board],
]:
    roots: list[dict[int, float] | None] = [None] * len(batch)
    policies: list[dict[int, np.ndarray] | None] = [None] * len(batch)
    requests: list[tuple[int, int]] = []
    boards: list[chess.Board] = []
    if not _reanalysis_enabled(network, training_step):
        return roots, policies, requests, boards
    for sample_index, sample in enumerate(batch):
        _append_sample_requests(network, sample, sample_index, unroll, roots, policies, requests, boards)
    return roots, policies, requests, boards


def _reanalysis_enabled(network: NetworkRuntime, training_step: int) -> bool:
    learner = network._learner
    return (
        learner.reanalyze_mcts_sims > 0 and learner.reanalyze_prob > 0 and training_step >= learner.reanalyze_start_step
    )


def _append_sample_requests(
    network: NetworkRuntime,
    sample: tuple[Trajectory, int],
    sample_index: int,
    unroll: int,
    roots: list[dict[int, float] | None],
    policies: list[dict[int, np.ndarray] | None],
    requests: list[tuple[int, int]],
    boards: list[chess.Board],
) -> None:
    if np.random.random() >= network._learner.reanalyze_prob:
        return
    trajectory, start_position = sample
    roots[sample_index] = {}
    if network._learner.reanalyze_policy:
        policies[sample_index] = {}
    board, player = network._game.replay_board_player(trajectory.actions, start_position)
    for offset in range(unroll + 1):
        position = start_position + offset
        if position >= trajectory.game_length:
            break
        canonical = network._game.get_canonical_form(board, player)
        boards.append(canonical.copy(stack=True))
        requests.append((sample_index, position))
        if position + 1 < trajectory.game_length:
            player = network._game.push_action(board, player, int(trajectory.actions[position]))


def _apply_reanalysis(
    network: NetworkRuntime,
    mcts_params: MCTSParams | None,
    boards: list[chess.Board],
    requests: list[tuple[int, int]],
    roots: list[dict[int, float] | None],
    policies: list[dict[int, np.ndarray] | None],
) -> None:
    params = replace(
        mcts_params or MCTSParams(),
        num_mcts_sims=network._learner.reanalyze_mcts_sims,
        dir_noise=False,
    )
    was_training = network.nnet.training
    try:
        results = network._create_reanalysis_search(params).search_batch(
            list(boards),
            temp=1.0,
            add_exploration_noise=False,
        )
    finally:
        network.nnet.train(was_training)
    if len(results) != len(requests):
        raise RuntimeError(f"Reanalysis returned {len(results)} results for {len(requests)} requested positions")
    for request, result in zip(requests, results, strict=True):
        _store_reanalysis_result(request, result, roots, policies)


def _store_reanalysis_result(
    request: tuple[int, int],
    result: tuple[np.ndarray, float, np.ndarray, np.ndarray],
    roots: list[dict[int, float] | None],
    policies: list[dict[int, np.ndarray] | None],
) -> None:
    sample_index, position = request
    policy, root_value, _observation, _valid = result
    root_override = roots[sample_index]
    if root_override is None:
        raise RuntimeError("Reanalysis result has no matching value target")
    root_override[position] = root_value
    policy_override = policies[sample_index]
    if policy_override is not None:
        policy_override[position] = policy.astype(np.float32, copy=False)


def validate_training_inputs(
    replay: PrioritizedReplayBuffer,
    steps: int,
    batch_size: int,
    unroll: int,
    td_steps: int,
) -> None:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if unroll < 0:
        raise ValueError(f"unroll_steps cannot be negative, got {unroll}")
    if td_steps < 0:
        raise ValueError(f"td_steps cannot be negative, got {td_steps}")
    if replay.size == 0:
        raise ValueError("Cannot train on empty replay buffer")
