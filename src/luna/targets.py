"""EfficientZeroV2 target generation: n-step bootstrap values, unroll targets."""

import math
from typing import Any

import numpy as np

from luna.replay_buffer import Trajectory


def _validate_target_request(trajectory: Trajectory, pos_idx: int, td_steps: int, discount: float) -> None:
    if isinstance(pos_idx, bool) or not isinstance(pos_idx, int) or not 0 <= pos_idx < trajectory.game_length:
        raise IndexError(f"pos_idx must be in [0, {trajectory.game_length}), got {pos_idx}")
    if isinstance(td_steps, bool) or not isinstance(td_steps, int) or td_steps < 0:
        raise ValueError("td_steps must be a non-negative integer")
    if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be finite and between 0 and 1")


def _validate_policy_override(trajectory: Trajectory, position: int, policy: np.ndarray) -> np.ndarray:
    override = np.asarray(policy, dtype=np.float32)
    expected_shape = trajectory.root_policies[position].shape
    if override.shape != expected_shape:
        raise ValueError(
            f"Policy override at position {position} must have shape {expected_shape}, got {override.shape}"
        )
    if not np.isfinite(override).all() or np.any(override < 0.0):
        raise ValueError(f"Policy override at position {position} must be finite and non-negative")
    if np.any(override[~trajectory.valids[position]] != 0.0):
        raise ValueError(f"Policy override at position {position} assigns probability to illegal actions")
    if not np.isclose(float(override.sum()), 1.0, rtol=0.0, atol=1e-4):
        raise ValueError(f"Policy override at position {position} must sum to one")
    return override


def compute_target_value(
    trajectory: Trajectory,
    pos_idx: int,
    td_steps: int,
    discount: float = 1.0,
    *,
    root_value_override: dict[int, float] | None = None,
) -> float:
    """Compute an alternating-sign n-step value target.

    A fresh search value at the current position takes precedence over the
    trajectory target instead of being interpreted as a later bootstrap.
    """
    _validate_target_request(trajectory, pos_idx, td_steps, discount)
    game_len = trajectory.game_length
    if root_value_override is not None and pos_idx in root_value_override:
        override = float(root_value_override[pos_idx])
        if not math.isfinite(override):
            raise ValueError(f"Root-value override at position {pos_idx} must be finite")
        return override
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
            if not math.isfinite(v_boot):
                raise ValueError(f"Root-value override at position {bootstrap_idx} must be finite")
        else:
            v_boot = float(trajectory.root_values[bootstrap_idx])
        value += (discount**td_steps) * sign * v_boot

    return value


def build_unroll_targets(
    trajectory: Trajectory,
    pos_idx: int,
    unroll_steps: int,
    td_steps: int,
    discount: float = 1.0,
    *,
    root_value_override: dict[int, float] | None = None,
    policy_override: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Build aligned policy, value, reward, legality, and consistency targets."""
    if isinstance(unroll_steps, bool) or not isinstance(unroll_steps, int) or unroll_steps < 0:
        raise ValueError("unroll_steps must be a non-negative integer")
    _validate_target_request(trajectory, pos_idx, td_steps, discount)
    game_len = trajectory.game_length

    target_values: list[float] = []
    target_rewards: list[float] = []
    target_policies: list[np.ndarray] = []
    observations_unroll: list[np.ndarray] = []
    valid_masks_unroll: list[np.ndarray] = []
    actions: list[int] = []
    unroll_mask: list[float] = []
    consistency_mask: list[float] = []
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
                consistency_mask.append(float(idx + 1 < game_len))
            else:
                actions.append(0)
                target_rewards.append(0.0)
                unroll_mask.append(0.0)
                consistency_mask.append(0.0)

        if idx < game_len:
            if policy_override is not None and idx in policy_override:
                target_policies.append(_validate_policy_override(trajectory, idx, policy_override[idx]))
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
        "consistency_mask": consistency_mask,
        "value_mask": value_mask,
    }


def collate_batch(
    batch_targets: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    if not batch_targets:
        raise ValueError("Cannot collate an empty target batch")
    B = len(batch_targets)
    K = len(batch_targets[0]["actions"])

    observations = np.stack([t["observation"] for t in batch_targets]).astype(np.float32)
    valid_masks = np.stack([t["valid_mask"] for t in batch_targets]).astype(np.float32)

    target_values = np.array([t["target_values"] for t in batch_targets], dtype=np.float32)
    target_rewards = np.array([t["target_rewards"] for t in batch_targets], dtype=np.float32)
    observations_unroll = np.stack([np.stack(t["observations_unroll"]) for t in batch_targets]).astype(np.float32)

    policies_list = [np.stack(t["target_policies"]) for t in batch_targets]
    target_policies = np.stack(policies_list).astype(np.float32)
    if target_policies.ndim != 3 or target_policies.shape[:2] != (B, K + 1):
        raise ValueError(f"Policy targets must have shape (batch, unroll + 1, actions), got {target_policies.shape}")

    valid_masks_unroll = np.stack([np.stack(t["valid_masks_unroll"]) for t in batch_targets]).astype(np.float32)

    actions = np.array([t["actions"] for t in batch_targets], dtype=np.int64)
    unroll_mask = np.array([t["unroll_mask"] for t in batch_targets], dtype=np.float32)
    consistency_mask = np.array([t["consistency_mask"] for t in batch_targets], dtype=np.float32)
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
        "consistency_mask": consistency_mask,
        "value_mask": value_mask,
    }
