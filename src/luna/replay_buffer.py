"""Prioritized trajectory replay buffer for EfficientZeroV2 training.

Large, dense fields use compact storage and are expanded to float32 only while
collating a sampled batch. This keeps a long self-play window practical without
changing the learner's numerical precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from threading import RLock

import chess
import numpy as np

from luna.trajectory import TrajectoryArrays, TrajectoryInput, TrajectoryMetadata, prepare_trajectory

REPLAY_SNAPSHOT_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class ReplaySnapshot:
    """Serializable replay state captured at one completed trainer iteration."""

    schema_version: int
    trainer_iteration: int
    capacity: int
    alpha: float
    beta: float
    beta_increment: float
    max_priority: float
    write_pos: int
    size: int
    leaf_priorities: np.ndarray
    entries: tuple[tuple[Trajectory, int] | None, ...]


class Trajectory:
    """One self-play game trajectory with contiguous array storage."""

    __slots__ = (
        "actions",
        "game_length",
        "observations",
        "policy_train_mask",
        "repetition_guard_attempts",
        "repetition_guard_excluded_actions",
        "repetition_guard_forced_fallbacks",
        "repetition_guard_interventions",
        "rewards",
        "root_policies",
        "root_values",
        "search_contempt_frozen_nodes",
        "search_contempt_opponent_selections",
        "search_contempt_thompson_selections",
        "termination",
        "truncated",
        "truncation_bootstrap_value",
        "valids",
    )

    def __init__(
        self,
        observations: list[np.ndarray] | np.ndarray,
        actions: list[int] | np.ndarray,
        rewards: list[float] | np.ndarray,
        root_policies: list[np.ndarray] | np.ndarray,
        root_values: list[float] | np.ndarray,
        valids: list[np.ndarray] | np.ndarray,
        truncated: bool = False,
        truncation_bootstrap_value: float | None = None,
        termination: chess.Termination | None = None,
        repetition_guard_attempts: int = 0,
        repetition_guard_interventions: int = 0,
        repetition_guard_forced_fallbacks: int = 0,
        repetition_guard_excluded_actions: int = 0,
        search_contempt_opponent_selections: int = 0,
        search_contempt_thompson_selections: int = 0,
        search_contempt_frozen_nodes: int = 0,
        policy_train_mask: list[bool] | np.ndarray | None = None,
    ) -> None:
        arrays, metadata = prepare_trajectory(
            TrajectoryInput(
                observations=observations,
                actions=actions,
                rewards=rewards,
                root_policies=root_policies,
                root_values=root_values,
                valids=valids,
                policy_train_mask=policy_train_mask,
                truncated=truncated,
                truncation_bootstrap_value=truncation_bootstrap_value,
                termination=termination,
                repetition_guard_attempts=repetition_guard_attempts,
                repetition_guard_interventions=repetition_guard_interventions,
                repetition_guard_forced_fallbacks=repetition_guard_forced_fallbacks,
                repetition_guard_excluded_actions=repetition_guard_excluded_actions,
                search_contempt_opponent_selections=search_contempt_opponent_selections,
                search_contempt_thompson_selections=search_contempt_thompson_selections,
                search_contempt_frozen_nodes=search_contempt_frozen_nodes,
            )
        )
        self._store_arrays(arrays)
        self._store_metadata(metadata)

    def _store_arrays(self, arrays: TrajectoryArrays) -> None:
        self.observations = arrays.observations
        self.actions = arrays.actions
        self.rewards = arrays.rewards
        self.root_policies = arrays.root_policies
        self.root_values = arrays.root_values
        self.valids = arrays.valids
        self.policy_train_mask = arrays.policy_train_mask
        self.game_length = arrays.game_length

    def _store_metadata(self, metadata: TrajectoryMetadata) -> None:
        self.truncated = metadata.truncated
        self.truncation_bootstrap_value = metadata.truncation_bootstrap_value
        self.termination = metadata.termination
        self.repetition_guard_attempts = metadata.repetition_guard_attempts
        self.repetition_guard_interventions = metadata.repetition_guard_interventions
        self.repetition_guard_forced_fallbacks = metadata.repetition_guard_forced_fallbacks
        self.repetition_guard_excluded_actions = metadata.repetition_guard_excluded_actions
        self.search_contempt_opponent_selections = metadata.search_contempt_opponent_selections
        self.search_contempt_thompson_selections = metadata.search_contempt_thompson_selections
        self.search_contempt_frozen_nodes = metadata.search_contempt_frozen_nodes


class _SumTree:
    """Fixed-capacity sum-tree for O(log N) priority sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data: list[tuple[Trajectory, int] | None] = [None] * capacity
        self.write_pos = 0
        self.size = 0

    def _propagate(self, idx: int) -> None:
        parent = idx >> 1
        while parent >= 1:
            self.tree[parent] = self.tree[2 * parent] + self.tree[2 * parent + 1]
            parent >>= 1

    def total(self) -> float:
        return float(self.tree[1])

    def add(self, priority: float, data: tuple[Trajectory, int]) -> None:
        idx = self.write_pos + self.capacity
        self.data[self.write_pos] = data
        self.tree[idx] = priority
        self._propagate(idx)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, data_idx: int, priority: float) -> None:
        idx = data_idx + self.capacity
        self.tree[idx] = priority
        self._propagate(idx)

    def get(self, cumsum: float) -> tuple[int, float, tuple[Trajectory, int] | None]:
        """Walk tree to find leaf. Returns (data_idx, priority, data)."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if cumsum <= self.tree[left]:
                idx = left
            else:
                cumsum -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return data_idx, float(self.tree[idx]), self.data[data_idx]


class PrioritizedReplayBuffer:
    """Stores full game trajectories with per-position priority."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 6e-6) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be between 0 and 1")
        if beta_increment < 0.0:
            raise ValueError("beta_increment cannot be negative")
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self._tree = _SumTree(capacity)
        self._max_priority = 1.0
        self._lock = RLock()

    @property
    def size(self) -> int:
        with self._lock:
            return self._tree.size

    def configure_beta_annealing(self, expected_sample_calls: int) -> None:
        """Linearly anneal the current importance exponent to one over future samples."""
        if isinstance(expected_sample_calls, bool) or not isinstance(expected_sample_calls, int):
            raise ValueError("expected_sample_calls must be a positive integer")
        if expected_sample_calls <= 0:
            raise ValueError("expected_sample_calls must be a positive integer")
        with self._lock:
            self.beta_increment = (1.0 - self.beta) / expected_sample_calls

    def save_trajectory(self, trajectory: Trajectory) -> None:
        """Store a trajectory, giving each position the current max priority."""
        with self._lock:
            for pos_idx in range(trajectory.game_length):
                priority = self._max_priority**self.alpha
                self._tree.add(priority, (trajectory, pos_idx))

    def sample(self, batch_size: int, unroll_steps: int) -> tuple[list[tuple[Trajectory, int]], np.ndarray, list[int]]:
        """Sample positions with importance weights and indices for priority updates."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if unroll_steps < 0:
            raise ValueError("unroll_steps cannot be negative")
        with self._lock:
            return self._sample_locked(batch_size)

    def _sample_locked(self, batch_size: int) -> tuple[list[tuple[Trajectory, int]], np.ndarray, list[int]]:
        if self._tree.size == 0:
            raise ValueError("Cannot sample an empty replay buffer")
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch: list[tuple[Trajectory, int]] = []
        indices: list[int] = []
        priorities = np.zeros(batch_size, dtype=np.float64)

        total = self._tree.total()
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("Replay priorities must have a positive finite sum")
        segment = total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            cumsum = np.random.uniform(lo, hi)
            data_idx, prio, data = self._tree.get(cumsum)
            while data is None:
                cumsum = np.random.uniform(0, total)
                data_idx, prio, data = self._tree.get(cumsum)
            traj, pos_idx = data
            pos_idx = min(pos_idx, traj.game_length - 1)
            batch.append((traj, pos_idx))
            indices.append(data_idx)
            priorities[i] = max(prio, 1e-8)

        probs = priorities / (total + 1e-8)
        weights = (self._tree.size * probs) ** (-self.beta)
        weights /= weights.max() + 1e-8
        return batch, weights.astype(np.float32), indices

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:
        """Update priorities based on absolute TD errors."""
        errors = np.asarray(td_errors)
        if errors.ndim != 1:
            raise ValueError("td_errors must be one-dimensional")
        if len(indices) != len(errors):
            raise ValueError("indices and td_errors must have the same length")
        with self._lock:
            raw_priorities: dict[int, float] = {}
            for idx, err in zip(indices, errors, strict=True):
                if (
                    isinstance(idx, bool | np.bool_)
                    or not isinstance(idx, int | np.integer)
                    or not 0 <= idx < self.capacity
                ):
                    raise IndexError(f"Replay index out of range: {idx}")
                normalized_idx = int(idx)
                if self._tree.data[normalized_idx] is None:
                    raise IndexError(f"Replay index is not active: {normalized_idx}")
                if not np.isfinite(err):
                    raise ValueError("TD errors must be finite")
                raw_priority = abs(float(err)) + 1e-6
                raw_priorities[normalized_idx] = max(raw_priorities.get(normalized_idx, 0.0), raw_priority)

            for idx, raw_priority in raw_priorities.items():
                priority = raw_priority**self.alpha
                self._max_priority = max(self._max_priority, raw_priority)
                self._tree.update(idx, priority)

    def snapshot(self, trainer_iteration: int) -> ReplaySnapshot:
        """Capture a consistent replay state after a completed iteration."""
        if isinstance(trainer_iteration, bool) or not isinstance(trainer_iteration, int) or trainer_iteration < 0:
            raise ValueError("trainer_iteration must be a non-negative integer")
        with self._lock:
            return ReplaySnapshot(
                schema_version=REPLAY_SNAPSHOT_SCHEMA_VERSION,
                trainer_iteration=trainer_iteration,
                capacity=self.capacity,
                alpha=self.alpha,
                beta=self.beta,
                beta_increment=self.beta_increment,
                max_priority=self._max_priority,
                write_pos=self._tree.write_pos,
                size=self._tree.size,
                leaf_priorities=self._tree.tree[self.capacity :].copy(),
                entries=tuple(self._tree.data),
            )

    def restore(self, snapshot: ReplaySnapshot, expected_iteration: int) -> int:
        """Validate and atomically install a persisted replay state."""
        tree = self._validated_snapshot_tree(snapshot, expected_iteration)
        with self._lock:
            self.beta = snapshot.beta
            self.beta_increment = snapshot.beta_increment
            self._max_priority = snapshot.max_priority
            self._tree = tree
        return snapshot.trainer_iteration

    def _validated_snapshot_tree(self, snapshot: ReplaySnapshot, expected_iteration: int) -> _SumTree:
        _validate_snapshot_metadata(self, snapshot, expected_iteration)
        entries = list(snapshot.entries)
        priorities = np.asarray(snapshot.leaf_priorities)
        _validate_snapshot_entries(entries, priorities, snapshot)
        tree = _SumTree(self.capacity)
        tree.data = entries
        tree.write_pos = snapshot.write_pos
        tree.size = snapshot.size
        tree.tree[self.capacity :] = priorities
        for index in range(self.capacity - 1, 0, -1):
            tree.tree[index] = tree.tree[2 * index] + tree.tree[2 * index + 1]
        return tree


def _validate_snapshot_metadata(
    replay: PrioritizedReplayBuffer,
    snapshot: ReplaySnapshot,
    expected_iteration: int,
) -> None:
    if not isinstance(snapshot, ReplaySnapshot):
        raise TypeError("Replay snapshot has an unsupported payload type")
    if snapshot.schema_version != REPLAY_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported replay snapshot schema version: {snapshot.schema_version}")
    if snapshot.capacity != replay.capacity:
        raise ValueError(f"Replay capacity changed from {snapshot.capacity} to {replay.capacity}")
    if not math.isclose(snapshot.alpha, replay.alpha, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Replay alpha changed from {snapshot.alpha} to {replay.alpha}")
    if snapshot.trainer_iteration not in {expected_iteration, expected_iteration - 1}:
        raise ValueError(
            f"Replay iteration {snapshot.trainer_iteration} is incompatible with checkpoint iteration {expected_iteration}"
        )
    if not 0.0 <= snapshot.beta <= 1.0:
        raise ValueError("Replay beta must be between zero and one")
    if not math.isfinite(snapshot.beta_increment) or snapshot.beta_increment < 0.0:
        raise ValueError("Replay beta increment must be finite and non-negative")
    if not math.isfinite(snapshot.max_priority) or snapshot.max_priority <= 0.0:
        raise ValueError("Replay max priority must be finite and positive")
    if not 0 <= snapshot.write_pos < snapshot.capacity:
        raise ValueError("Replay write position is outside its capacity")
    if not 0 <= snapshot.size <= snapshot.capacity:
        raise ValueError("Replay size is outside its capacity")


def _validate_snapshot_entries(
    entries: list[tuple[Trajectory, int] | None],
    priorities: np.ndarray,
    snapshot: ReplaySnapshot,
) -> None:
    if len(entries) != snapshot.capacity:
        raise ValueError("Replay entry count differs from its capacity")
    if priorities.shape != (snapshot.capacity,) or priorities.dtype != np.float64:
        raise ValueError("Replay priorities have an invalid shape or dtype")
    if not np.isfinite(priorities).all() or np.any(priorities < 0.0):
        raise ValueError("Replay priorities must be finite and non-negative")
    occupied = np.fromiter((entry is not None for entry in entries), dtype=np.bool_, count=len(entries))
    if int(occupied.sum()) != snapshot.size:
        raise ValueError("Replay size differs from its occupied entry count")
    if np.any(priorities[occupied] <= 0.0) or np.any(priorities[~occupied] != 0.0):
        raise ValueError("Replay priorities do not match occupied entries")
    if snapshot.size < snapshot.capacity and snapshot.write_pos != snapshot.size:
        raise ValueError("Partially filled replay has an invalid write position")
    for entry in entries:
        if entry is None:
            continue
        trajectory, position = entry
        if not isinstance(trajectory, Trajectory):
            raise TypeError("Replay entry contains an unsupported trajectory type")
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < trajectory.game_length:
            raise ValueError("Replay entry contains an invalid trajectory position")
