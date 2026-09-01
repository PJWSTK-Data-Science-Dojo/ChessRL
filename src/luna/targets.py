"""EfficientZeroV2 target generation: n-step bootstrap values, unroll targets."""

import math
from dataclasses import dataclass, field
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
    if root_value_override is not None and pos_idx in root_value_override:
        return _finite_override(root_value_override[pos_idx], pos_idx)
    bootstrap_idx = pos_idx + td_steps
    value = _discounted_rewards(trajectory, pos_idx, bootstrap_idx, discount)
    if bootstrap_idx < trajectory.game_length:
        value += _bootstrap_value(trajectory, bootstrap_idx, td_steps, discount, root_value_override)
    elif trajectory.truncated:
        value += _truncation_bootstrap_value(trajectory, pos_idx, discount, root_value_override)
    return value


def _finite_override(value: float, position: int) -> float:
    override = float(value)
    if not math.isfinite(override):
        raise ValueError(f"Root-value override at position {position} must be finite")
    return override


def _discounted_rewards(trajectory: Trajectory, pos_idx: int, bootstrap_idx: int, discount: float) -> float:
    end = min(bootstrap_idx, trajectory.game_length)
    n = end - pos_idx
    if n <= 0:
        return 0.0
    rewards = trajectory.rewards[pos_idx:end].astype(np.float64)
    steps = np.arange(n, dtype=np.float64)
    signs = np.where(steps % 2 == 0, 1.0, -1.0)
    return float(((discount**steps) * signs * rewards).sum())


def _bootstrap_value(
    trajectory: Trajectory,
    bootstrap_idx: int,
    td_steps: int,
    discount: float,
    overrides: dict[int, float] | None,
) -> float:
    if overrides is not None and bootstrap_idx in overrides:
        value = _finite_override(overrides[bootstrap_idx], bootstrap_idx)
    else:
        value = float(trajectory.root_values[bootstrap_idx])
    return _discounted_bootstrap(value, td_steps, discount)


def _truncation_bootstrap_value(
    trajectory: Trajectory,
    pos_idx: int,
    discount: float,
    overrides: dict[int, float] | None,
) -> float:
    bootstrap_idx = trajectory.game_length
    if overrides is not None and bootstrap_idx in overrides:
        value = _finite_override(overrides[bootstrap_idx], bootstrap_idx)
    else:
        stored_value = trajectory.truncation_bootstrap_value
        if stored_value is None:
            raise RuntimeError("Truncated trajectory is missing its validated bootstrap value")
        value = stored_value
    return _discounted_bootstrap(value, bootstrap_idx - pos_idx, discount)


def _discounted_bootstrap(value: float, steps: int, discount: float) -> float:
    sign = 1.0 if steps % 2 == 0 else -1.0
    return (discount**steps) * sign * value


@dataclass
class _TargetLists:
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)
    observations: list[np.ndarray] = field(default_factory=list)
    valid_masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    unroll_mask: list[float] = field(default_factory=list)
    consistency_mask: list[float] = field(default_factory=list)
    value_mask: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class _TargetRequest:
    trajectory: Trajectory
    pos_idx: int
    unroll_steps: int
    td_steps: int
    discount: float
    root_value_override: dict[int, float] | None
    policy_override: dict[int, np.ndarray] | None


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
    request = _TargetRequest(
        trajectory,
        pos_idx,
        unroll_steps,
        td_steps,
        discount,
        root_value_override,
        policy_override,
    )
    targets = _TargetLists()
    for step in range(unroll_steps + 1):
        _append_target_step(targets, request, step)
    return _target_mapping(targets, request)


def _append_target_step(targets: _TargetLists, request: _TargetRequest, step: int) -> None:
    position = request.pos_idx + step
    active = position < request.trajectory.game_length
    _append_value_target(targets, request, position, active)
    if step < request.unroll_steps:
        _append_transition_target(targets, request.trajectory, position, active)
    _append_state_target(targets, request, position, active)


def _append_value_target(targets: _TargetLists, request: _TargetRequest, position: int, active: bool) -> None:
    if not active:
        targets.values.append(0.0)
        targets.value_mask.append(0.0)
        return
    targets.values.append(
        compute_target_value(
            request.trajectory,
            position,
            request.td_steps,
            request.discount,
            root_value_override=request.root_value_override,
        )
    )
    targets.value_mask.append(1.0)


def _append_transition_target(
    targets: _TargetLists,
    trajectory: Trajectory,
    position: int,
    active: bool,
) -> None:
    targets.actions.append(int(trajectory.actions[position]) if active else 0)
    targets.rewards.append(float(trajectory.rewards[position]) if active else 0.0)
    targets.unroll_mask.append(float(active))
    targets.consistency_mask.append(float(active and position + 1 < trajectory.game_length))


def _append_state_target(targets: _TargetLists, request: _TargetRequest, position: int, active: bool) -> None:
    trajectory = request.trajectory
    if active:
        override = request.policy_override
        policy = (
            _validate_policy_override(trajectory, position, override[position])
            if override is not None and position in override
            else trajectory.root_policies[position]
        )
        targets.policies.append(policy)
        targets.observations.append(trajectory.observations[position])
        targets.valid_masks.append(trajectory.valids[position])
        return
    action_size = trajectory.root_policies.shape[1]
    targets.policies.append(np.ones(action_size, dtype=np.float32) / action_size)
    targets.observations.append(trajectory.observations[-1])
    targets.valid_masks.append(trajectory.valids[-1])


def _target_mapping(targets: _TargetLists, request: _TargetRequest) -> dict[str, Any]:
    trajectory = request.trajectory
    return {
        "observation": trajectory.observations[request.pos_idx],
        "valid_mask": trajectory.valids[request.pos_idx],
        "target_values": targets.values,
        "target_rewards": targets.rewards,
        "target_policies": targets.policies,
        "observations_unroll": targets.observations,
        "valid_masks_unroll": targets.valid_masks,
        "actions": targets.actions,
        "unroll_mask": targets.unroll_mask,
        "consistency_mask": targets.consistency_mask,
        "value_mask": targets.value_mask,
    }


def collate_batch(
    batch_targets: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    if not batch_targets:
        raise ValueError("Cannot collate an empty target batch")
    collated = _collate_state_targets(batch_targets)
    collated.update(_collate_unroll_arrays(batch_targets))
    expected_horizon = len(batch_targets[0]["actions"]) + 1
    policies = collated["target_policies"]
    if policies.ndim != 3 or policies.shape[:2] != (len(batch_targets), expected_horizon):
        raise ValueError(f"Policy targets must have shape (batch, unroll + 1, actions), got {policies.shape}")
    return collated


def _collate_state_targets(batch_targets: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    policies = [np.stack(target["target_policies"]) for target in batch_targets]
    observations = [np.stack(target["observations_unroll"]) for target in batch_targets]
    valid_masks = [np.stack(target["valid_masks_unroll"]) for target in batch_targets]
    return {
        "observations": np.stack([target["observation"] for target in batch_targets]).astype(np.float32),
        "valid_masks": np.stack([target["valid_mask"] for target in batch_targets]).astype(np.float32),
        "target_policies": np.stack(policies).astype(np.float32),
        "observations_unroll": np.stack(observations).astype(np.float32),
        "valid_masks_unroll": np.stack(valid_masks).astype(np.float32),
    }


def _collate_unroll_arrays(batch_targets: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "target_values": np.array([target["target_values"] for target in batch_targets], dtype=np.float32),
        "target_rewards": np.array([target["target_rewards"] for target in batch_targets], dtype=np.float32),
        "actions": np.array([target["actions"] for target in batch_targets], dtype=np.int64),
        "unroll_mask": np.array([target["unroll_mask"] for target in batch_targets], dtype=np.float32),
        "consistency_mask": np.array([target["consistency_mask"] for target in batch_targets], dtype=np.float32),
        "value_mask": np.array([target["value_mask"] for target in batch_targets], dtype=np.float32),
    }
