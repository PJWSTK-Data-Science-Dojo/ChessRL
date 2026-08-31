"""Prioritized trajectory replay buffer for EfficientZeroV2 training.

Large, dense fields use compact storage and are expanded to float32 only while
collating a sampled batch. This keeps a long self-play window practical without
changing the learner's numerical precision.
"""

from threading import RLock

import chess
import numpy as np

from luna.game.chess_game import ACTION_SIZE, OBS_PLANES

_POLICY_SUM_ATOL = 5e-3


class Trajectory:
    """One self-play game trajectory with contiguous array storage."""

    __slots__ = (
        "actions",
        "game_length",
        "observations",
        "rewards",
        "root_policies",
        "root_values",
        "termination",
        "truncated",
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
        termination: chess.Termination | None = None,
    ) -> None:
        raw_actions = np.asarray(actions)
        if raw_actions.ndim != 1 or raw_actions.size == 0:
            raise ValueError("A trajectory must contain at least one one-dimensional action sequence")
        if raw_actions.dtype.kind not in {"i", "u"}:
            raise ValueError("Trajectory actions must be integers")
        if np.any(raw_actions < 0) or np.any(raw_actions >= ACTION_SIZE):
            raise ValueError(f"Trajectory actions must be in [0, {ACTION_SIZE})")

        observations_array = np.ascontiguousarray(observations, dtype=np.float16)
        actions_array = raw_actions.astype(np.int64, copy=False)
        rewards_array = np.asarray(rewards, dtype=np.float32)
        policies_array = np.ascontiguousarray(root_policies, dtype=np.float16)
        values_array = np.asarray(root_values, dtype=np.float32)
        raw_valids = np.asarray(valids)

        game_length = int(actions_array.shape[0])
        named_lengths = {
            "observations": len(observations_array),
            "rewards": len(rewards_array),
            "root_policies": len(policies_array),
            "root_values": len(values_array),
            "valids": len(raw_valids),
        }
        mismatched = {name: length for name, length in named_lengths.items() if length != game_length}
        if mismatched:
            raise ValueError(f"Trajectory fields must all have length {game_length}; got {mismatched}")
        expected_observation_shape = (game_length, 8, 8, OBS_PLANES)
        if observations_array.shape != expected_observation_shape:
            raise ValueError(
                f"Trajectory observations must have shape {expected_observation_shape}, got {observations_array.shape}"
            )
        expected_policy_shape = (game_length, ACTION_SIZE)
        if policies_array.shape != expected_policy_shape or raw_valids.shape != expected_policy_shape:
            raise ValueError(
                f"Trajectory root_policies and valids must have shape {expected_policy_shape}; "
                f"got {policies_array.shape} and {raw_valids.shape}"
            )
        if rewards_array.ndim != 1 or values_array.ndim != 1:
            raise ValueError("Trajectory rewards and root values must be one-dimensional")
        if not np.isfinite(observations_array).all():
            raise ValueError("Trajectory observations must be finite")
        if not np.isfinite(rewards_array).all() or not np.isfinite(values_array).all():
            raise ValueError("Trajectory rewards and root values must be finite")
        if not np.isfinite(policies_array).all() or np.any(policies_array < 0):
            raise ValueError("Trajectory root policies must be finite and non-negative")
        if raw_valids.dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError("Trajectory valid masks must contain numeric zero/one values")
        if not np.isfinite(raw_valids).all() or not np.all((raw_valids == 0) | (raw_valids == 1)):
            raise ValueError("Trajectory valid masks must contain only finite zero/one values")
        valids_array = np.ascontiguousarray(raw_valids, dtype=np.bool_)
        if not np.all(valids_array.any(axis=1)):
            raise ValueError("Every trajectory position must contain at least one legal action")
        if not np.all(valids_array[np.arange(game_length), actions_array]):
            raise ValueError("Every trajectory action must be legal in its stored position")
        if np.any(policies_array[~valids_array] != 0):
            raise ValueError("Trajectory root policies must assign zero probability to illegal actions")
        policy_sums = policies_array.astype(np.float32).sum(axis=1)
        if not np.allclose(policy_sums, 1.0, rtol=0.0, atol=_POLICY_SUM_ATOL):
            raise ValueError("Every trajectory root policy must sum to one")
        if not isinstance(truncated, bool | np.bool_):
            raise ValueError("truncated must be a boolean")
        if termination is not None and not isinstance(termination, chess.Termination):
            raise TypeError("termination must be a chess.Termination or None")
        if truncated and termination is not None:
            raise ValueError("A truncated trajectory cannot have a terminal chess outcome")

        self.observations = observations_array
        self.actions = actions_array
        self.rewards = rewards_array
        self.root_policies = policies_array
        self.root_values = values_array
        self.valids = valids_array
        self.game_length = game_length
        self.truncated = bool(truncated)
        self.termination = termination


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
            for idx, err in zip(indices, errors):
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
