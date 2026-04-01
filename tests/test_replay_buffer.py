"""Tests for prioritized replay buffer."""

from __future__ import annotations

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from luna.replay_buffer import PrioritizedReplayBuffer, Trajectory


def _make_trajectory(length: int = 10, action_size: int = 4096) -> Trajectory:
    return Trajectory(
        observations=[np.random.randn(8 * 8 * 6).astype(np.float32) for _ in range(length)],
        actions=[np.random.randint(0, action_size) for _ in range(length)],
        rewards=[(-1.0) ** i * 0.5 for i in range(length)],
        root_policies=[np.random.dirichlet(np.ones(action_size)).astype(np.float32) for _ in range(length)],
        root_values=[np.random.uniform(-1, 1) for _ in range(length)],
        valids=[np.random.randint(0, 2, size=action_size).astype(np.float32) for _ in range(length)],
    )


class TestPrioritizedReplayBuffer:
    def test_save_and_size(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=500)
        traj = _make_trajectory(length=10)
        buf.save_trajectory(traj)
        assert buf.size == 10

    def test_sample_returns_correct_shape(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=500)
        for _ in range(5):
            buf.save_trajectory(_make_trajectory(length=10))

        batch, weights, indices = buf.sample(batch_size=8, unroll_steps=5)
        assert len(batch) == 8
        assert weights.shape == (8,)
        assert len(indices) == 8
        assert weights.dtype == np.float32
        assert all(w > 0 for w in weights)

    def test_update_priorities(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=500)
        buf.save_trajectory(_make_trajectory(length=20))
        _, _, indices = buf.sample(batch_size=4, unroll_steps=5)
        td_errors = np.array([0.1, 0.5, 1.0, 2.0])
        buf.update_priorities(indices, td_errors)

    def test_capacity_wraps(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=20)
        for _ in range(5):
            buf.save_trajectory(_make_trajectory(length=10))
        assert buf.size == 20

    def test_sample_positions_valid(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=500)
        traj = _make_trajectory(length=15)
        buf.save_trajectory(traj)
        batch, _, _ = buf.sample(batch_size=10, unroll_steps=3)
        for t, pos in batch:
            assert 0 <= pos < t.game_length
