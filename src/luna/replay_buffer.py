"""Prioritized trajectory replay buffer for EfficientZeroV2 training.

Trajectories are stored as contiguous numpy arrays for cache-friendly access
and zero-copy slicing during training.
"""

import numpy as np


class Trajectory:
    """One self-play game trajectory with contiguous array storage."""

    __slots__ = ("actions", "game_length", "observations", "rewards", "root_policies", "root_values", "valids")

    def __init__(
        self,
        observations: list[np.ndarray] | np.ndarray,
        actions: list[int] | np.ndarray,
        rewards: list[float] | np.ndarray,
        root_policies: list[np.ndarray] | np.ndarray,
        root_values: list[float] | np.ndarray,
        valids: list[np.ndarray] | np.ndarray,
    ) -> None:
        self.observations = np.ascontiguousarray(observations, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.int64)
        self.rewards = np.asarray(rewards, dtype=np.float32)
        self.root_policies = np.ascontiguousarray(root_policies, dtype=np.float32)
        self.root_values = np.asarray(root_values, dtype=np.float32)
        self.valids = np.ascontiguousarray(valids, dtype=np.float32)
        self.game_length = int(self.actions.shape[0])


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
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self._tree = _SumTree(capacity)
        self._max_priority = 1.0

    @property
    def size(self) -> int:
        return self._tree.size

    def save_trajectory(self, trajectory: Trajectory) -> None:
        """Store a trajectory, giving each position the current max priority."""
        for pos_idx in range(trajectory.game_length):
            priority = self._max_priority**self.alpha
            self._tree.add(priority, (trajectory, pos_idx))

    def sample(self, batch_size: int, unroll_steps: int) -> tuple[list[tuple[Trajectory, int]], np.ndarray, list[int]]:
        """Sample batch_size (trajectory, position) pairs.

        Returns:
            batch: list of (trajectory, pos_idx) tuples
            weights: importance-sampling weights (batch_size,)
            indices: tree data indices for priority updates
        """
        assert unroll_steps >= 0
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch: list[tuple[Trajectory, int]] = []
        indices: list[int] = []
        priorities = np.zeros(batch_size, dtype=np.float64)

        total = self._tree.total()
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
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max() + 1e-8
        return batch, weights.astype(np.float32), indices

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:
        """Update priorities based on absolute TD errors."""
        for idx, err in zip(indices, td_errors):
            priority = (abs(float(err)) + 1e-6) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            self._tree.update(idx, priority)
