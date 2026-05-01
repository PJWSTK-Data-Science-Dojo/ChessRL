"""Tests for target generation module."""

import numpy as np

from luna.game.chess_game import ACTION_SIZE, OBS_PLANES
from luna.targets import build_unroll_targets, collate_batch, compute_target_value


class TestComputeTargetValue:
    def test_terminal_position(self, make_trajectory):
        traj = make_trajectory(length=5)
        traj.rewards = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        val = compute_target_value(traj, pos_idx=4, td_steps=5, discount=1.0)
        assert abs(val - 1.0) < 1e-6

    def test_bootstrap(self, make_trajectory):
        traj = make_trajectory(length=10)
        traj.rewards = np.zeros(10, dtype=np.float32)
        traj.root_values = np.array([0.0] * 5 + [1.0] * 5, dtype=np.float32)
        val = compute_target_value(traj, pos_idx=0, td_steps=5, discount=1.0)
        assert abs(val - (-1.0)) < 1e-6


class TestBuildUnrollTargets:
    def test_output_shapes(self, make_trajectory):
        traj = make_trajectory(length=10)
        targets = build_unroll_targets(traj, pos_idx=0, unroll_steps=3, td_steps=5)
        assert len(targets["target_values"]) == 4
        assert len(targets["target_rewards"]) == 3
        assert len(targets["target_policies"]) == 4
        assert len(targets["actions"]) == 3
        assert targets["observation"].shape == (8, 8, OBS_PLANES)
        assert len(targets["unroll_mask"]) == 3
        assert len(targets["value_mask"]) == 4

    def test_past_end_padding(self, make_trajectory):
        traj = make_trajectory(length=3)
        targets = build_unroll_targets(traj, pos_idx=1, unroll_steps=5, td_steps=3)
        assert len(targets["target_values"]) == 6
        assert targets["target_values"][-1] == 0.0
        assert targets["unroll_mask"] == [1.0, 1.0, 0.0, 0.0, 0.0]
        assert targets["value_mask"] == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_collation(make_trajectory):
    traj = make_trajectory(length=10)
    samples = [build_unroll_targets(traj, i, unroll_steps=3, td_steps=5) for i in range(4)]
    collated = collate_batch(samples)
    assert collated["observations"].shape[0] == 4
    assert collated["target_values"].shape == (4, 4)
    assert collated["target_rewards"].shape == (4, 3)
    assert collated["actions"].shape == (4, 3)
    assert collated["target_policies"].shape == (4, 4, ACTION_SIZE)
    assert collated["valid_masks_unroll"].shape == (4, 4, ACTION_SIZE)
    assert collated["unroll_mask"].shape == (4, 3)
    assert collated["value_mask"].shape == (4, 4)


def test_root_value_override(make_trajectory):
    traj = make_trajectory(length=10)
    traj.root_values = np.zeros(10, dtype=np.float32)
    traj.rewards = np.zeros(10, dtype=np.float32)
    ov = {5: 0.99}
    val = compute_target_value(traj, pos_idx=0, td_steps=5, discount=1.0, root_value_override=ov)
    assert abs(val - (-0.99)) < 1e-5
