"""EfficientZeroV2 target generation: n-step bootstrap values, unroll targets."""

from typing import Any

import numpy as np

from .replay_buffer import Trajectory


def compute_target_value(
    trajectory: Trajectory,
    pos_idx: int,
    td_steps: int,
    discount: float = 0.997,
    *,
    root_value_override: dict[int, float] | None = None,
) -> float:
    """Compute n-step TD target values with alternating signs for two-player games.

    Formula: V_t = r_t + γ r_{t+1} + γ² r_{t+2} + ... + γⁿ V_{t+n}

    Two-player adjustment: Rewards alternate signs based on player perspective.

    Args:
        trajectory: Game trajectory containing rewards and root values.
        pos_idx: Position index in trajectory to compute target for.
        td_steps: Bootstrap horizon n (how many steps to look ahead).
        discount: Discount factor γ ∈ (0, 1].
        root_value_override: Optional dict mapping position indices to fresh MCTS values
            (for reanalysis-based search value).

    Returns:
        n-step value target with shape matching rewards.
    """
    game_len = trajectory.game_length
    bootstrap_idx = pos_idx + td_steps

    end = min(bootstrap_idx, game_len)
    n = end - pos_idx
    if n > 0:
        rewards = trajectory.rewards[pos_idx:end].astype(np.float64)
        steps = np.arange(n, dtype=np.float64)
        signs = np.where(steps % 2 == 0, 1.0, -1.0)
        discounts = discount**steps
        value = float((discounts * signs * rewards).sum())
    else:
        value = 0.0

    if bootstrap_idx < game_len:
        sign = 1.0 if td_steps % 2 == 0 else -1.0
        if root_value_override is not None and bootstrap_idx in root_value_override:
            v_boot = float(root_value_override[bootstrap_idx])
        else:
            v_boot = float(trajectory.root_values[bootstrap_idx])
        value += (discount**td_steps) * sign * v_boot

    return value


def build_unroll_targets(
    trajectory: Trajectory,
    pos_idx: int,
    unroll_steps: int,
    td_steps: int,
    discount: float = 0.997,
    *,
    root_value_override: dict[int, float] | None = None,
    policy_override: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Build targets for K-step unroll starting at pos_idx.

    Returns dict with lists of length (unroll_steps + 1):
        target_values: scalar value targets
        target_rewards: scalar reward targets (length unroll_steps, first is for step 0->1)
        target_policies: policy distribution targets
        observations_unroll: observations at steps [t, ..., t+K] for consistency targets
        actions: actions taken (length unroll_steps)
        observation: the root observation at pos_idx
        valid_mask: legal-action mask at pos_idx
        valid_masks_unroll: legal masks aligned with observations_unroll / policy rows
        unroll_mask: (unroll_steps,) float mask -- 1.0 for real steps, 0.0 for padding
        value_mask: (unroll_steps + 1,) float mask -- 1.0 for real value targets
    """
    game_len = trajectory.game_length

    target_values: list[float] = []
    target_rewards: list[float] = []
    target_policies: list[np.ndarray] = []
    observations_unroll: list[np.ndarray] = []
    valid_masks_unroll: list[np.ndarray] = []
    actions: list[int] = []
    unroll_mask: list[float] = []
    value_mask: list[float] = []

    for step in range(unroll_steps + 1):
        idx = pos_idx + step
        if idx < game_len:
            target_values.append(
                compute_target_value(
                    trajectory,
                    idx,
                    td_steps,
                    discount,
                    root_value_override=root_value_override,
                )
            )
            value_mask.append(1.0)
        else:
            target_values.append(0.0)
            value_mask.append(0.0)

        if step < unroll_steps:
            if idx < game_len:
                actions.append(int(trajectory.actions[idx]))
                target_rewards.append(float(trajectory.rewards[idx]))
                unroll_mask.append(1.0)
            else:
                actions.append(0)
                target_rewards.append(0.0)
                unroll_mask.append(0.0)

        if idx < game_len:
            if policy_override is not None and idx in policy_override:
                target_policies.append(policy_override[idx])
            else:
                target_policies.append(trajectory.root_policies[idx])
            observations_unroll.append(trajectory.observations[idx])
            valid_masks_unroll.append(trajectory.valids[idx])
        else:
            action_size = trajectory.root_policies.shape[1]
            target_policies.append(np.ones(action_size, dtype=np.float32) / action_size)
            observations_unroll.append(trajectory.observations[-1])
            valid_masks_unroll.append(trajectory.valids[-1])

    obs = trajectory.observations[pos_idx] if pos_idx < game_len else trajectory.observations[-1]
    valid_mask_arr = trajectory.valids[pos_idx] if pos_idx < game_len else trajectory.valids[-1]

    return {
        "observation": obs,
        "valid_mask": valid_mask_arr,
        "target_values": target_values,
        "target_rewards": target_rewards,
        "target_policies": target_policies,
        "observations_unroll": observations_unroll,
        "valid_masks_unroll": valid_masks_unroll,
        "actions": actions,
        "unroll_mask": unroll_mask,
        "value_mask": value_mask,
    }


def collate_batch(
    batch_targets: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Stack a list of per-sample target dicts into batched numpy arrays."""
    B = len(batch_targets)
    K = len(batch_targets[0]["actions"])

    observations = np.stack([t["observation"] for t in batch_targets]).astype(np.float32)
    valid_masks = np.stack([t["valid_mask"] for t in batch_targets]).astype(np.float32)

    target_values = np.array([t["target_values"] for t in batch_targets], dtype=np.float32)
    target_rewards = np.array([t["target_rewards"] for t in batch_targets], dtype=np.float32)
    observations_unroll = np.stack(
        [np.stack(t["observations_unroll"]) for t in batch_targets]
    ).astype(np.float32)

    policies_list = [np.stack(t["target_policies"]) for t in batch_targets]
    target_policies = np.stack(policies_list).astype(np.float32)
    assert target_policies.shape == (B, K + 1, target_policies.shape[2])

    valid_masks_unroll = np.stack([np.stack(t["valid_masks_unroll"]) for t in batch_targets]).astype(
        np.float32
    )

    actions = np.array([t["actions"] for t in batch_targets], dtype=np.int64)
    unroll_mask = np.array([t["unroll_mask"] for t in batch_targets], dtype=np.float32)
    value_mask = np.array([t["value_mask"] for t in batch_targets], dtype=np.float32)

    return {
        "observations": observations,
        "valid_masks": valid_masks,
        "target_values": target_values,
        "target_rewards": target_rewards,
        "target_policies": target_policies,
        "observations_unroll": observations_unroll,
        "valid_masks_unroll": valid_masks_unroll,
        "actions": actions,
        "unroll_mask": unroll_mask,
        "value_mask": value_mask,
    }
