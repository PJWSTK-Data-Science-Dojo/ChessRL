"""Tests for prioritized replay buffer."""

import pickle
from typing import Any

import chess
import numpy as np
import pytest

from luna.game.chess_game import ACTION_SIZE, OBS_PLANES
from luna.replay_buffer import PrioritizedReplayBuffer, Trajectory
from tests.conftest import TrajectoryFactory


def _trajectory_inputs(length: int = 2) -> dict[str, Any]:
    policies = np.zeros((length, ACTION_SIZE), dtype=np.float32)
    policies[:, 0] = 1.0
    valids = np.zeros((length, ACTION_SIZE), dtype=np.float32)
    valids[:, 0] = 1.0
    return {
        "observations": np.zeros((length, 8, 8, OBS_PLANES), dtype=np.float32),
        "actions": np.zeros(length, dtype=np.int64),
        "rewards": np.zeros(length, dtype=np.float32),
        "root_policies": policies,
        "root_values": np.zeros(length, dtype=np.float32),
        "valids": valids,
    }


class TestPrioritizedReplayBuffer:
    def test_save_and_size(self, make_trajectory: TrajectoryFactory) -> None:
        buf = PrioritizedReplayBuffer(capacity=500)
        traj = make_trajectory(length=10)
        buf.save_trajectory(traj)
        assert buf.size == 10

    def test_sample_returns_correct_shape(self, make_trajectory: TrajectoryFactory) -> None:
        buf = PrioritizedReplayBuffer(capacity=500)
        for _ in range(5):
            buf.save_trajectory(make_trajectory(length=10))

        batch, weights, indices = buf.sample(batch_size=8, unroll_steps=5)
        assert len(batch) == 8
        assert weights.shape == (8,)
        assert len(indices) == 8
        assert weights.dtype == np.float32
        assert all(w > 0 for w in weights)

    def test_update_priorities(self, make_trajectory: TrajectoryFactory) -> None:
        buf = PrioritizedReplayBuffer(capacity=500)
        buf.save_trajectory(make_trajectory(length=20))
        _, _, indices = buf.sample(batch_size=4, unroll_steps=5)
        td_errors = np.array([0.1, 0.5, 1.0, 2.0])
        buf.update_priorities(indices, td_errors)

    def test_duplicate_priority_updates_keep_largest_error(self, make_trajectory: TrajectoryFactory) -> None:
        buf = PrioritizedReplayBuffer(capacity=4, alpha=0.6)
        buf.save_trajectory(make_trajectory(length=1))

        buf.update_priorities([0, 0], np.array([0.9, 0.1], dtype=np.float32))

        assert buf._tree.tree[buf.capacity] == pytest.approx((0.9 + 1e-6) ** 0.6)

    def test_priority_update_rejects_inactive_index_atomically(self, make_trajectory: TrajectoryFactory) -> None:
        buf = PrioritizedReplayBuffer(capacity=4)
        buf.save_trajectory(make_trajectory(length=1))
        initial_tree = buf._tree.tree.copy()

        with pytest.raises(IndexError, match="not active"):
            buf.update_priorities([0, 1], np.array([0.5, 0.6], dtype=np.float32))

        np.testing.assert_array_equal(buf._tree.tree, initial_tree)

    def test_capacity_wraps(self, make_trajectory: TrajectoryFactory) -> None:
        buf = PrioritizedReplayBuffer(capacity=20)
        for _ in range(5):
            buf.save_trajectory(make_trajectory(length=10))
        assert buf.size == 20

    def test_sample_positions_valid(self, make_trajectory: TrajectoryFactory) -> None:
        buf = PrioritizedReplayBuffer(capacity=500)
        traj = make_trajectory(length=15)
        buf.save_trajectory(traj)
        batch, _, _ = buf.sample(batch_size=10, unroll_steps=3)
        for t, pos in batch:
            assert 0 <= pos < t.game_length

    def test_beta_annealing_reaches_one_and_stays_capped(self, make_trajectory: TrajectoryFactory) -> None:
        buf = PrioritizedReplayBuffer(capacity=8, beta=0.4)
        buf.save_trajectory(make_trajectory(length=2))
        buf.configure_beta_annealing(expected_sample_calls=3)

        observed = []
        for _ in range(4):
            buf.sample(batch_size=1, unroll_steps=0)
            observed.append(buf.beta)

        assert observed == pytest.approx([0.6, 0.8, 1.0, 1.0])

    @pytest.mark.parametrize("sample_calls", [0, -1, True, 1.5])
    def test_beta_annealing_rejects_invalid_sample_counts(self, sample_calls: int) -> None:
        buf = PrioritizedReplayBuffer(capacity=8)

        with pytest.raises(ValueError, match="positive integer"):
            buf.configure_beta_annealing(sample_calls)

    def test_trajectory_truncation_marker_defaults_false_and_survives_ipc_pickle(
        self,
        make_trajectory: TrajectoryFactory,
    ) -> None:
        complete = make_trajectory(length=1)
        truncated = make_trajectory(length=1, truncated=True)

        restored = pickle.loads(pickle.dumps(truncated))

        assert complete.truncated is False
        assert restored.truncated is True
        assert restored.truncation_bootstrap_value == 0.0

    def test_trajectory_termination_survives_ipc_pickle(self, make_trajectory: TrajectoryFactory) -> None:
        trajectory = make_trajectory(length=1, termination=chess.Termination.THREEFOLD_REPETITION)

        restored = pickle.loads(pickle.dumps(trajectory))

        assert restored.termination is chess.Termination.THREEFOLD_REPETITION

    def test_policy_training_mask_defaults_true_and_survives_pickle(self) -> None:
        trajectory = Trajectory(**_trajectory_inputs(), policy_train_mask=[True, False])

        restored = pickle.loads(pickle.dumps(trajectory))

        assert restored.policy_train_mask.tolist() == [True, False]
        assert Trajectory(**_trajectory_inputs()).policy_train_mask.tolist() == [True, True]

    def test_policy_training_mask_requires_one_boolean_per_position(self) -> None:
        with pytest.raises(ValueError, match="must contain booleans"):
            Trajectory(**_trajectory_inputs(), policy_train_mask=[1, 0])
        with pytest.raises(ValueError, match="fields must all have length"):
            Trajectory(**_trajectory_inputs(), policy_train_mask=[True])

    def test_truncated_trajectory_rejects_terminal_outcome(self) -> None:
        with pytest.raises(ValueError, match="truncated trajectory"):
            Trajectory(
                **_trajectory_inputs(),
                truncated=True,
                termination=chess.Termination.THREEFOLD_REPETITION,
            )

    def test_truncated_trajectory_requires_finite_boundary_value(self) -> None:
        with pytest.raises(ValueError, match="requires a finite bootstrap value"):
            Trajectory(**_trajectory_inputs(), truncated=True)

    def test_complete_trajectory_rejects_truncation_boundary_value(self) -> None:
        with pytest.raises(ValueError, match="Only a truncated trajectory"):
            Trajectory(**_trajectory_inputs(), truncation_bootstrap_value=0.25)

    def test_invalid_inputs_fail_fast(self, make_trajectory: TrajectoryFactory) -> None:
        with pytest.raises(ValueError, match="capacity"):
            PrioritizedReplayBuffer(capacity=0)

        buf = PrioritizedReplayBuffer(capacity=8)
        with pytest.raises(ValueError, match="empty"):
            buf.sample(batch_size=1, unroll_steps=1)

        buf.save_trajectory(make_trajectory(length=2))
        with pytest.raises(ValueError, match="same length"):
            buf.update_priorities([0, 1], np.array([0.5], dtype=np.float32))


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("actions", np.array([0.5, 0.0]), "actions must be integers"),
        ("actions", np.array([0, ACTION_SIZE]), r"actions must be in \[0, 4288\)"),
        ("valids", np.full((2, ACTION_SIZE), np.nan), "finite zero/one"),
        ("root_values", np.array([0.0, np.inf]), "root values must be finite"),
    ],
)
def test_trajectory_rejects_corrupt_scalar_data(field: str, replacement: object, message: str) -> None:
    inputs = _trajectory_inputs()
    inputs[field] = replacement

    with pytest.raises(ValueError, match=message):
        Trajectory(**inputs)


def test_trajectory_rejects_illegal_policy_mass() -> None:
    inputs = _trajectory_inputs()
    policies = np.asarray(inputs["root_policies"]).copy()
    policies[:, 1] = 0.25
    policies[:, 0] = 0.75
    inputs["root_policies"] = policies

    with pytest.raises(ValueError, match="zero probability to illegal actions"):
        Trajectory(**inputs)
