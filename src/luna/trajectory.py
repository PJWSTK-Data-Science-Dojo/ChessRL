"""Validation and compact storage preparation for self-play trajectories."""

from __future__ import annotations

from dataclasses import dataclass, replace

import chess
import numpy as np

from luna.game.chess_game import ACTION_SIZE, OBS_PLANES

_POLICY_SUM_ATOL = 5e-3


@dataclass(frozen=True, slots=True)
class TrajectoryInput:
    observations: list[np.ndarray] | np.ndarray
    actions: list[int] | np.ndarray
    rewards: list[float] | np.ndarray
    root_policies: list[np.ndarray] | np.ndarray
    root_values: list[float] | np.ndarray
    valids: list[np.ndarray] | np.ndarray
    truncated: bool
    termination: chess.Termination | None
    repetition_guard_attempts: int
    repetition_guard_interventions: int
    repetition_guard_forced_fallbacks: int
    repetition_guard_excluded_actions: int


@dataclass(frozen=True, slots=True)
class TrajectoryArrays:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    root_policies: np.ndarray
    root_values: np.ndarray
    valids: np.ndarray
    game_length: int


@dataclass(frozen=True, slots=True)
class TrajectoryMetadata:
    truncated: bool
    termination: chess.Termination | None
    repetition_guard_attempts: int
    repetition_guard_interventions: int
    repetition_guard_forced_fallbacks: int
    repetition_guard_excluded_actions: int


def prepare_trajectory(values: TrajectoryInput) -> tuple[TrajectoryArrays, TrajectoryMetadata]:
    raw_actions = _validate_raw_actions(values.actions)
    arrays = _convert_arrays(values, raw_actions)
    _validate_lengths(arrays)
    _validate_shapes(arrays)
    _validate_finite_values(arrays)
    arrays = replace(arrays, valids=_validate_legal_moves(arrays))
    metadata = _validate_metadata(values, arrays.game_length)
    return arrays, metadata


def _validate_raw_actions(actions: list[int] | np.ndarray) -> np.ndarray:
    raw = np.asarray(actions)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("A trajectory must contain at least one one-dimensional action sequence")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError("Trajectory actions must be integers")
    if np.any(raw < 0) or np.any(raw >= ACTION_SIZE):
        raise ValueError(f"Trajectory actions must be in [0, {ACTION_SIZE})")
    return raw


def _convert_arrays(values: TrajectoryInput, raw_actions: np.ndarray) -> TrajectoryArrays:
    observations = np.ascontiguousarray(values.observations, dtype=np.float16)
    actions = raw_actions.astype(np.int64, copy=False)
    rewards = np.asarray(values.rewards, dtype=np.float32)
    policies = np.ascontiguousarray(values.root_policies, dtype=np.float16)
    root_values = np.asarray(values.root_values, dtype=np.float32)
    valids = np.asarray(values.valids)
    return TrajectoryArrays(
        observations=observations,
        actions=actions,
        rewards=rewards,
        root_policies=policies,
        root_values=root_values,
        valids=valids,
        game_length=int(actions.shape[0]),
    )


def _validate_lengths(arrays: TrajectoryArrays) -> None:
    lengths = {
        "observations": len(arrays.observations),
        "rewards": len(arrays.rewards),
        "root_policies": len(arrays.root_policies),
        "root_values": len(arrays.root_values),
        "valids": len(arrays.valids),
    }
    mismatched = {name: length for name, length in lengths.items() if length != arrays.game_length}
    if mismatched:
        raise ValueError(f"Trajectory fields must all have length {arrays.game_length}; got {mismatched}")


def _validate_shapes(arrays: TrajectoryArrays) -> None:
    expected_observations = (arrays.game_length, 8, 8, OBS_PLANES)
    if arrays.observations.shape != expected_observations:
        raise ValueError(
            f"Trajectory observations must have shape {expected_observations}, got {arrays.observations.shape}"
        )
    expected_policy = (arrays.game_length, ACTION_SIZE)
    if arrays.root_policies.shape != expected_policy or arrays.valids.shape != expected_policy:
        raise ValueError(
            f"Trajectory root_policies and valids must have shape {expected_policy}; "
            f"got {arrays.root_policies.shape} and {arrays.valids.shape}"
        )
    if arrays.rewards.ndim != 1 or arrays.root_values.ndim != 1:
        raise ValueError("Trajectory rewards and root values must be one-dimensional")


def _validate_finite_values(arrays: TrajectoryArrays) -> None:
    if not np.isfinite(arrays.observations).all():
        raise ValueError("Trajectory observations must be finite")
    if not np.isfinite(arrays.rewards).all() or not np.isfinite(arrays.root_values).all():
        raise ValueError("Trajectory rewards and root values must be finite")
    if not np.isfinite(arrays.root_policies).all() or np.any(arrays.root_policies < 0):
        raise ValueError("Trajectory root policies must be finite and non-negative")


def _validate_legal_moves(arrays: TrajectoryArrays) -> np.ndarray:
    raw_valids = arrays.valids
    if raw_valids.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError("Trajectory valid masks must contain numeric zero/one values")
    if not np.isfinite(raw_valids).all() or not np.all((raw_valids == 0) | (raw_valids == 1)):
        raise ValueError("Trajectory valid masks must contain only finite zero/one values")
    valids = np.ascontiguousarray(raw_valids, dtype=np.bool_)
    if not np.all(valids.any(axis=1)):
        raise ValueError("Every trajectory position must contain at least one legal action")
    if not np.all(valids[np.arange(arrays.game_length), arrays.actions]):
        raise ValueError("Every trajectory action must be legal in its stored position")
    if np.any(arrays.root_policies[~valids] != 0):
        raise ValueError("Trajectory root policies must assign zero probability to illegal actions")
    policy_sums = arrays.root_policies.astype(np.float32).sum(axis=1)
    if not np.allclose(policy_sums, 1.0, rtol=0.0, atol=_POLICY_SUM_ATOL):
        raise ValueError("Every trajectory root policy must sum to one")
    return valids


def _validate_metadata(values: TrajectoryInput, game_length: int) -> TrajectoryMetadata:
    if not isinstance(values.truncated, bool | np.bool_):
        raise ValueError("truncated must be a boolean")
    if values.termination is not None and not isinstance(values.termination, chess.Termination):
        raise TypeError("termination must be a chess.Termination or None")
    if values.truncated and values.termination is not None:
        raise ValueError("A truncated trajectory cannot have a terminal chess outcome")
    counts = _guard_counts(values)
    _validate_guard_counts(counts, game_length)
    return TrajectoryMetadata(
        truncated=bool(values.truncated),
        termination=values.termination,
        repetition_guard_attempts=values.repetition_guard_attempts,
        repetition_guard_interventions=values.repetition_guard_interventions,
        repetition_guard_forced_fallbacks=values.repetition_guard_forced_fallbacks,
        repetition_guard_excluded_actions=values.repetition_guard_excluded_actions,
    )


def _guard_counts(values: TrajectoryInput) -> dict[str, int]:
    return {
        "repetition_guard_attempts": values.repetition_guard_attempts,
        "repetition_guard_interventions": values.repetition_guard_interventions,
        "repetition_guard_forced_fallbacks": values.repetition_guard_forced_fallbacks,
        "repetition_guard_excluded_actions": values.repetition_guard_excluded_actions,
    }


def _validate_guard_counts(counts: dict[str, int], game_length: int) -> None:
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    attempts = counts["repetition_guard_attempts"]
    interventions = counts["repetition_guard_interventions"]
    forced_fallbacks = counts["repetition_guard_forced_fallbacks"]
    excluded_actions = counts["repetition_guard_excluded_actions"]
    if attempts != interventions + forced_fallbacks:
        raise ValueError("repetition guard attempts must equal interventions plus forced fallbacks")
    if attempts > game_length:
        raise ValueError("repetition guard cannot be attempted more than once per trajectory position")
    if interventions > 0 and excluded_actions < interventions:
        raise ValueError("each repetition guard intervention must exclude at least one action")
