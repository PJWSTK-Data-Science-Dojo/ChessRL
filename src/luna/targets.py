"""EfficientZeroV2 target generation: n-step bootstrap values, unroll targets."""

from __future__ import annotations

import numpy as np

from .replay_buffer import Trajectory


def compute_target_value(
    trajectory: Trajectory,
    pos_idx: int,
    td_steps: int,
    discount: float = 0.997,
) -> float:
    """Compute n-step bootstrapped value target for a position.

    V_target = sum_{i=0}^{n-1} discount^i * r_{t+i} + discount^n * v_{t+n}
    where v_{t+n} comes from the stored MCTS root value.
    """
    game_len = trajectory.game_length
    bootstrap_idx = pos_idx + td_steps

    value = 0.0
    for i in range(pos_idx, min(bootstrap_idx, game_len)):
        sign = 1.0 if (i - pos_idx) % 2 == 0 else -1.0
        value += (discount ** (i - pos_idx)) * sign * trajectory.rewards[i]

    if bootstrap_idx < game_len:
        sign = 1.0 if td_steps % 2 == 0 else -1.0
        value += (discount**td_steps) * sign * trajectory.root_values[bootstrap_idx]

    return value


def build_unroll_targets(
    trajectory: Trajectory,
    pos_idx: int,
    unroll_steps: int,
    td_steps: int,
    discount: float = 0.997,
) -> dict[str, list]:
    """Build targets for K-step unroll starting at pos_idx.

    Returns dict with lists of length (unroll_steps + 1):
        target_values: scalar value targets
        target_rewards: scalar reward targets (length unroll_steps, first is for step 0->1)
        target_policies: policy distribution targets
        observations_unroll: observations at steps [t, ..., t+K] for consistency targets
        actions: actions taken (length unroll_steps)
        observation: the root observation at pos_idx
        valid_mask: legal-action mask at pos_idx
    """
    game_len = trajectory.game_length

    target_values: list[float] = []
    target_rewards: list[float] = []
    target_policies: list[np.ndarray] = []
    observations_unroll: list[np.ndarray] = []
    actions: list[int] = []

    for step in range(unroll_steps + 1):
        idx = pos_idx + step
        if idx < game_len:
            target_values.append(compute_target_value(trajectory, idx, td_steps, discount))
        else:
            target_values.append(0.0)

        if step < unroll_steps:
            if idx < game_len:
                actions.append(trajectory.actions[idx])
                target_rewards.append(trajectory.rewards[idx])
            else:
                actions.append(0)
                target_rewards.append(0.0)

        if idx < game_len:
            target_policies.append(trajectory.root_policies[idx])
            observations_unroll.append(trajectory.observations[idx])
        else:
            action_size = len(trajectory.root_policies[0])
            target_policies.append(np.ones(action_size, dtype=np.float32) / action_size)
            observations_unroll.append(trajectory.observations[-1])

    obs = trajectory.observations[pos_idx] if pos_idx < game_len else trajectory.observations[-1]
    valid_mask = trajectory.valids[pos_idx] if pos_idx < game_len else trajectory.valids[-1]

    return {
        "observation": obs,
        "valid_mask": valid_mask,
        "target_values": target_values,
        "target_rewards": target_rewards,
        "target_policies": target_policies,
        "observations_unroll": observations_unroll,
        "actions": actions,
    }


def collate_batch(
    batch_targets: list[dict[str, list]],
) -> dict[str, np.ndarray]:
    """Stack a list of per-sample target dicts into batched numpy arrays."""
    B = len(batch_targets)
    K = len(batch_targets[0]["actions"])

    observations = np.array([t["observation"] for t in batch_targets], dtype=np.float32)
    valid_masks = np.array([t["valid_mask"] for t in batch_targets], dtype=np.float32)

    target_values = np.array([t["target_values"] for t in batch_targets], dtype=np.float32)
    target_rewards = np.array([t["target_rewards"] for t in batch_targets], dtype=np.float32)
    observations_unroll = np.array([t["observations_unroll"] for t in batch_targets], dtype=np.float32)

    target_policies = np.zeros((B, K + 1, len(batch_targets[0]["target_policies"][0])), dtype=np.float32)
    for i, t in enumerate(batch_targets):
        for step, pol in enumerate(t["target_policies"]):
            target_policies[i, step] = pol

    actions = np.array([t["actions"] for t in batch_targets], dtype=np.int64)

    return {
        "observations": observations,
        "valid_masks": valid_masks,
        "target_values": target_values,
        "target_rewards": target_rewards,
        "target_policies": target_policies,
        "observations_unroll": observations_unroll,
        "actions": actions,
    }
