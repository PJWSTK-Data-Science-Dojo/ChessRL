"""Tests for prioritized replay buffer."""

import numpy as np

from luna.game.chess_game import ACTION_SIZE
from luna.replay_buffer import PrioritizedReplayBuffer


class TestPrioritizedReplayBuffer:
    def test_save_and_size(self, make_trajectory):
        buf = PrioritizedReplayBuffer(capacity=500)
        traj = make_trajectory(length=10)
        buf.save_trajectory(traj)
        assert buf.size == 10

    def test_sample_returns_correct_shape(self, make_trajectory):
        buf = PrioritizedReplayBuffer(capacity=500)
        for _ in range(5):
            buf.save_trajectory(make_trajectory(length=10))

        batch, weights, indices = buf.sample(batch_size=8, unroll_steps=5)
        assert len(batch) == 8
        assert weights.shape == (8,)
        assert len(indices) == 8
        assert weights.dtype == np.float32
        assert all(w > 0 for w in weights)

    def test_update_priorities(self, make_trajectory):
        buf = PrioritizedReplayBuffer(capacity=500)
        buf.save_trajectory(make_trajectory(length=20))
        _, _, indices = buf.sample(batch_size=4, unroll_steps=5)
        td_errors = np.array([0.1, 0.5, 1.0, 2.0])
        buf.update_priorities(indices, td_errors)

    def test_capacity_wraps(self, make_trajectory):
        buf = PrioritizedReplayBuffer(capacity=20)
        for _ in range(5):
            buf.save_trajectory(make_trajectory(length=10))
        assert buf.size == 20

    def test_sample_positions_valid(self, make_trajectory):
        buf = PrioritizedReplayBuffer(capacity=500)
        traj = make_trajectory(length=15)
        buf.save_trajectory(traj)
        batch, _, _ = buf.sample(batch_size=10, unroll_steps=3)
        for t, pos in batch:
            assert 0 <= pos < t.game_length
