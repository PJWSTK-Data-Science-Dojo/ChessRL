"""Durability tests for replay snapshots."""

from pathlib import Path
from unittest.mock import patch

import chess
import numpy as np
import pytest

from luna.coach import Coach
from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.game.chess_game import ChessGame
from luna.network import LunaNetwork
from luna.replay_buffer import PrioritizedReplayBuffer, ReplaySnapshot
from luna.replay_persistence import (
    REPLAY_SNAPSHOT_NAME,
    ReplaySnapshotError,
    load_replay_snapshot,
    save_replay_snapshot,
)
from tests.conftest import TrajectoryFactory


def test_replay_snapshot_roundtrip_preserves_ring_and_priorities(
    tmp_path: Path,
    make_trajectory: TrajectoryFactory,
) -> None:
    source = PrioritizedReplayBuffer(capacity=4, alpha=0.7, beta=0.25, beta_increment=0.03)
    source.save_trajectory(make_trajectory(3, termination=chess.Termination.CHECKMATE))
    source.save_trajectory(make_trajectory(2, truncated=True, truncation_bootstrap_value=0.4))
    source.update_priorities([0, 2, 3], np.array([2.0, 0.5, 1.5], dtype=np.float32))
    before = source.snapshot(trainer_iteration=7)

    path = save_replay_snapshot(source, tmp_path, trainer_iteration=7)
    restored = PrioritizedReplayBuffer(capacity=4, alpha=0.7, beta=0.9, beta_increment=0.0)
    restored_iteration = load_replay_snapshot(restored, tmp_path, expected_iteration=7)
    after = restored.snapshot(trainer_iteration=7)

    assert path == tmp_path / REPLAY_SNAPSHOT_NAME
    assert restored_iteration == 7
    _assert_snapshots_equal(after, before)
    assert after.entries[0] is not None
    assert after.entries[3] is not None
    assert after.entries[0][0] is after.entries[3][0]
    assert after.entries[0][0].truncated
    assert after.entries[0][0].truncation_bootstrap_value == pytest.approx(0.4)
    assert after.entries[1] is not None
    assert after.entries[1][0].termination is chess.Termination.CHECKMATE


def test_failed_atomic_publish_preserves_previous_snapshot(
    tmp_path: Path,
    make_trajectory: TrajectoryFactory,
) -> None:
    replay = PrioritizedReplayBuffer(capacity=4)
    replay.save_trajectory(make_trajectory(1))
    destination = save_replay_snapshot(replay, tmp_path, trainer_iteration=1)
    original = destination.read_bytes()
    replay.save_trajectory(make_trajectory(1))

    with (
        patch("luna.replay_persistence.os.replace", side_effect=OSError("injected publish failure")),
        pytest.raises(OSError, match="injected publish failure"),
    ):
        save_replay_snapshot(replay, tmp_path, trainer_iteration=2)

    assert destination.read_bytes() == original
    assert list(tmp_path.glob(f".{REPLAY_SNAPSHOT_NAME}.tmp-*")) == []
    restored = PrioritizedReplayBuffer(capacity=4)
    assert load_replay_snapshot(restored, tmp_path, expected_iteration=2) == 1
    assert restored.size == 1


def test_corrupt_snapshot_fails_without_mutating_replay(
    tmp_path: Path,
    make_trajectory: TrajectoryFactory,
) -> None:
    target = PrioritizedReplayBuffer(capacity=4)
    target.save_trajectory(make_trajectory(2))
    before = target.snapshot(trainer_iteration=3)
    (tmp_path / REPLAY_SNAPSHOT_NAME).write_bytes(b"not a zstandard frame")

    with pytest.raises(ReplaySnapshotError, match="Invalid replay snapshot"):
        load_replay_snapshot(target, tmp_path, expected_iteration=3)

    _assert_snapshots_equal(target.snapshot(trainer_iteration=3), before)


def test_coach_restores_replay_on_ordinary_resume(
    tmp_path: Path,
    make_trajectory: TrajectoryFactory,
) -> None:
    game = ChessGame()
    learner = EzV2LearnerConfig(
        device="cpu",
        num_channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        proj_dim=32,
        batch_size=4,
        dataloader_workers=0,
        compile_inference=False,
    )
    source = LunaNetwork(game, learner)
    source._trainer_iteration = 2
    source.save_checkpoint(str(tmp_path), "checkpoint_2.pth.tar")
    replay = PrioritizedReplayBuffer(capacity=8)
    replay.save_trajectory(make_trajectory(3))
    replay.beta = 0.63
    save_replay_snapshot(replay, tmp_path, trainer_iteration=2)
    resumed = LunaNetwork(game, learner)
    resumed.load_checkpoint(str(tmp_path), "checkpoint_2.pth.tar")
    coach = Coach(
        game,
        resumed,
        TrainingRunConfig(
            num_iters=2,
            replay_capacity=8,
            checkpoint=str(tmp_path),
            stockfish_eval_every=0,
            ladder_eval_every=0,
        ),
        restore_replay=True,
    )

    coach.learn()

    assert coach.replay.size == 3
    assert coach.replay.beta == pytest.approx(0.63)


def _assert_snapshots_equal(actual: ReplaySnapshot, expected: ReplaySnapshot) -> None:
    assert actual.schema_version == expected.schema_version
    assert actual.trainer_iteration == expected.trainer_iteration
    assert actual.capacity == expected.capacity
    assert actual.alpha == expected.alpha
    assert actual.beta == expected.beta
    assert actual.beta_increment == expected.beta_increment
    assert actual.max_priority == expected.max_priority
    assert actual.write_pos == expected.write_pos
    assert actual.size == expected.size
    np.testing.assert_array_equal(actual.leaf_priorities, expected.leaf_priorities)
    assert len(actual.entries) == len(expected.entries)
    for actual_entry, expected_entry in zip(actual.entries, expected.entries, strict=True):
        if expected_entry is None:
            assert actual_entry is None
            continue
        assert actual_entry is not None
        actual_trajectory, actual_position = actual_entry
        expected_trajectory, expected_position = expected_entry
        assert actual_position == expected_position
        np.testing.assert_array_equal(actual_trajectory.observations, expected_trajectory.observations)
        np.testing.assert_array_equal(actual_trajectory.actions, expected_trajectory.actions)
        np.testing.assert_array_equal(actual_trajectory.rewards, expected_trajectory.rewards)
        np.testing.assert_array_equal(actual_trajectory.root_policies, expected_trajectory.root_policies)
        np.testing.assert_array_equal(actual_trajectory.root_values, expected_trajectory.root_values)
        np.testing.assert_array_equal(actual_trajectory.valids, expected_trajectory.valids)
        np.testing.assert_array_equal(actual_trajectory.policy_train_mask, expected_trajectory.policy_train_mask)
