"""Tests for target generation module."""

from __future__ import annotations

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from luna.replay_buffer import Trajectory
from luna.targets import build_unroll_targets, collate_batch, compute_target_value


def _make_trajectory(length: int = 10, action_size: int = 100) -> Trajectory:
    return Trajectory(
        observations=[np.random.randn(8 * 8 * 6).astype(np.float32) for _ in range(length)],
        actions=list(range(length)),
        rewards=[1.0 if i == length - 1 else 0.0 for i in range(length)],
        root_policies=[np.ones(action_size, dtype=np.float32) / action_size for _ in range(length)],
        root_values=[0.5] * length,
        valids=[np.ones(action_size, dtype=np.float32) for _ in range(length)],
    )


class TestComputeTargetValue:
    def test_terminal_position(self) -> None:
        traj = _make_trajectory(length=5)
        traj.rewards = [0.0, 0.0, 0.0, 0.0, 1.0]
        val = compute_target_value(traj, pos_idx=4, td_steps=5, discount=1.0)
        assert abs(val - 1.0) < 1e-6

    def test_bootstrap(self) -> None:
        traj = _make_trajectory(length=10)
        traj.rewards = [0.0] * 10
        traj.root_values = [0.0] * 5 + [1.0] * 5
        val = compute_target_value(traj, pos_idx=0, td_steps=5, discount=1.0)
        assert abs(val - (-1.0)) < 1e-6


class TestBuildUnrollTargets:
    def test_output_shapes(self) -> None:
        traj = _make_trajectory(length=10, action_size=100)
        targets = build_unroll_targets(traj, pos_idx=0, unroll_steps=3, td_steps=5)
        assert len(targets["target_values"]) == 4
        assert len(targets["target_rewards"]) == 3
        assert len(targets["target_policies"]) == 4
        assert len(targets["actions"]) == 3
        assert targets["observation"].shape == (8 * 8 * 6,)

    def test_past_end_padding(self) -> None:
        traj = _make_trajectory(length=3, action_size=100)
        targets = build_unroll_targets(traj, pos_idx=1, unroll_steps=5, td_steps=3)
        assert len(targets["target_values"]) == 6
        assert targets["target_values"][-1] == 0.0

    def test_policies_normalised(self) -> None:
        traj = _make_trajectory(length=10, action_size=100)
        targets = build_unroll_targets(traj, pos_idx=0, unroll_steps=3, td_steps=5)
        for pol in targets["target_policies"]:
            assert abs(pol.sum() - 1.0) < 1e-5


class TestCollateBatch:
    def test_collation(self) -> None:
        traj = _make_trajectory(length=10, action_size=100)
        samples = [build_unroll_targets(traj, i, unroll_steps=3, td_steps=5) for i in range(4)]
        collated = collate_batch(samples)
        assert collated["observations"].shape[0] == 4
        assert collated["target_values"].shape == (4, 4)
        assert collated["target_rewards"].shape == (4, 3)
        assert collated["actions"].shape == (4, 3)
        assert collated["target_policies"].shape == (4, 4, 100)
