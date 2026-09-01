"""Tests for isolated persistent self-play actors."""

import threading
from multiprocessing.connection import Connection
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from luna.coach import Coach
from luna.config import EzV2LearnerConfig, TrainingRunConfig, validate_training_configuration
from luna.game.chess_game import ACTION_SIZE, OBS_PLANES, ChessGame
from luna.network import LunaNetwork
from luna.replay_buffer import Trajectory
from luna.self_play_actors import (
    SelfPlayActorError,
    SelfPlayActorPool,
    _actor_learner_config,
    _ActorCollectionDone,
    _ActorTrajectory,
    derive_actor_seed,
    partition_episode_counts,
)


class _BlockingReceiveConnection:
    def __init__(self, delegate: Connection) -> None:
        self._delegate = delegate
        self._closed = threading.Event()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def send(self, message: object) -> None:
        self._delegate.send(message)

    def recv(self) -> object:
        if not self._closed.wait(timeout=2.0):
            raise EOFError("test receive guard expired")
        raise EOFError("connection closed by actor-pool shutdown")

    def close(self) -> None:
        self._closed.set()
        self._delegate.close()


def _trajectory_with_action(action: int) -> Trajectory:
    policy = np.zeros((1, ACTION_SIZE), dtype=np.float32)
    policy[0, action] = 1.0
    valids = np.zeros((1, ACTION_SIZE), dtype=np.bool_)
    valids[0, action] = True
    return Trajectory(
        observations=np.zeros((1, 8, 8, OBS_PLANES), dtype=np.float32),
        actions=[action],
        rewards=[0.0],
        root_policies=policy,
        root_values=[0.0],
        valids=valids,
    )


def test_actor_seeds_are_repeatable_and_unique() -> None:
    seeds = [derive_actor_seed(7, actor_id, generation=12) for actor_id in range(4)]

    assert seeds == [derive_actor_seed(7, actor_id, generation=12) for actor_id in range(4)]
    assert len(set(seeds)) == 4
    assert seeds != [derive_actor_seed(7, actor_id, generation=13) for actor_id in range(4)]


def test_actor_learner_configuration_uses_eager_inference() -> None:
    learner = EzV2LearnerConfig(compile_inference=True, compile_training=True, dataloader_workers=4)

    actor_learner = _actor_learner_config(learner)

    assert not actor_learner.compile_inference
    assert not actor_learner.compile_training
    assert actor_learner.dataloader_workers == 0


def test_episode_partition_is_balanced_and_does_not_create_empty_work() -> None:
    assert partition_episode_counts(10, 3) == [4, 3, 3]
    assert partition_episode_counts(2, 4) == [1, 1]


def test_streamed_actor_trajectories_are_reassembled_in_episode_order() -> None:
    context = torch.multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=True)
    pool = object.__new__(SelfPlayActorPool)
    pool._connections = [parent_connection]
    try:
        child_connection.send(_ActorTrajectory(0, 7, 1, _trajectory_with_action(20)))
        child_connection.send(_ActorTrajectory(0, 7, 0, _trajectory_with_action(10)))
        child_connection.send(_ActorCollectionDone(0, 7, 2))

        trajectories = pool._receive_collection_blocking(0, episode_count=2, generation=7)

        assert [int(trajectory.actions[0]) for trajectory in trajectories] == [10, 20]
    finally:
        parent_connection.close()
        child_connection.close()


def test_streamed_actor_collection_rejects_missing_trajectory() -> None:
    context = torch.multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=True)
    pool = object.__new__(SelfPlayActorPool)
    pool._connections = [parent_connection]
    try:
        child_connection.send(_ActorTrajectory(0, 3, 1, _trajectory_with_action(20)))
        child_connection.send(_ActorCollectionDone(0, 3, 2))

        with pytest.raises(SelfPlayActorError, match=r"missing trajectory indices: \[0\]"):
            pool._receive_collection_blocking(0, episode_count=2, generation=3)
    finally:
        parent_connection.close()
        child_connection.close()


def test_actor_configuration_rejects_invalid_worker_count_and_timeout() -> None:
    learner = EzV2LearnerConfig(device="cpu")

    with pytest.raises(ValueError, match="self_play_workers must be a positive integer"):
        validate_training_configuration(TrainingRunConfig(self_play_workers=0), learner)
    with pytest.raises(ValueError, match="self_play_actor_timeout_s must be finite"):
        validate_training_configuration(TrainingRunConfig(self_play_actor_timeout_s=0.0), learner)


def test_coach_owns_actor_pool_for_the_complete_training_loop(tmp_path: Path) -> None:
    game = ChessGame()
    learner = EzV2LearnerConfig(
        device="cpu",
        num_channels=8,
        repr_blocks=0,
        dyn_blocks=0,
        proj_dim=16,
        dataloader_workers=0,
    )
    run = TrainingRunConfig(
        num_iters=1,
        num_episodes=4,
        self_play_workers=2,
        checkpoint=str(tmp_path),
        stockfish_eval_every=0,
    )
    network = LunaNetwork(game, learner)
    coach = Coach(game, network, run, seed=31)

    with (
        patch("luna.coach_training.SelfPlayActorPool") as actor_pool_type,
        patch.object(coach, "_learn_iterations") as learn_iterations,
    ):
        actor_pool = actor_pool_type.return_value.__enter__.return_value
        coach.learn()

    actor_pool_type.assert_called_once_with(network, run, worker_count=2, base_seed=31)
    learn_iterations.assert_called_once_with(1, actor_pool=actor_pool)


def test_spawned_actors_collect_compact_trajectories_and_fail_fast(
    tmp_path: Path,
) -> None:
    game = ChessGame()
    learner = EzV2LearnerConfig(
        device="cpu",
        num_channels=8,
        repr_blocks=0,
        dyn_blocks=0,
        proj_dim=16,
        dataloader_workers=0,
    )
    run = TrainingRunConfig(
        num_mcts_sims=1,
        num_episodes=3,
        parallel_games=1,
        self_play_workers=2,
        self_play_actor_timeout_s=60.0,
        max_ply=1,
        checkpoint=str(tmp_path),
        stockfish_eval_every=0,
    )
    network = LunaNetwork(game, learner)
    original_state = {name: tensor.detach().clone() for name, tensor in network.nnet.state_dict().items()}
    pool = SelfPlayActorPool(network, run, worker_count=2, base_seed=19)
    try:
        trajectories = pool.collect(3, generation=1)

        assert len(trajectories) == 3
        assert all(trajectory.game_length == 1 for trajectory in trajectories)
        assert all(trajectory.observations.dtype.name == "float16" for trajectory in trajectories)
        assert all(torch.equal(network.nnet.state_dict()[name], value) for name, value in original_state.items())

        next_trajectories = pool.collect(3, generation=2)
        assert len(next_trajectories) == 3
        assert all(trajectory.game_length == 1 for trajectory in next_trajectories)

        pool._connections[0].send(None)
        pool._processes[0].join(timeout=5.0)
        with pytest.raises(SelfPlayActorError, match=r"Actor 0 received an unsupported request: NoneType"):
            pool.collect(2, generation=3)
    finally:
        pool.close()


def test_collection_timeout_terminates_actors_and_joins_io_threads(tmp_path: Path) -> None:
    game = ChessGame()
    learner = EzV2LearnerConfig(
        device="cpu",
        num_channels=8,
        repr_blocks=0,
        dyn_blocks=0,
        proj_dim=16,
        dataloader_workers=0,
    )
    run = TrainingRunConfig(
        num_mcts_sims=1,
        num_episodes=1,
        parallel_games=1,
        self_play_workers=1,
        self_play_actor_timeout_s=60.0,
        max_ply=1,
        checkpoint=str(tmp_path),
        stockfish_eval_every=0,
    )
    pool = SelfPlayActorPool(LunaNetwork(game, learner), run, worker_count=1, base_seed=23)
    cache_root = Path(pool._cache_root.name)
    blocking_connection = _BlockingReceiveConnection(pool._connections[0])
    pool._connections[0] = cast(Connection, blocking_connection)
    pool._timeout_s = 0.05

    with pytest.raises(SelfPlayActorError, match=r"Timed out after 0\.05s waiting for"):
        pool.collect(1, generation=1)

    assert pool._closed
    assert blocking_connection.closed
    assert not pool._processes[0].is_alive()
    assert not cache_root.exists()
    assert not any(thread.name.startswith("luna-actor-io") for thread in threading.enumerate())
