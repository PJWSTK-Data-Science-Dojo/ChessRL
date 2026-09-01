"""Validation metrics for supervised PGN warm-start training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from luna.replay_buffer import Trajectory


class ValidationNetwork(Protocol):
    def batched_initial_inference(
        self,
        obs_batch: np.ndarray,
        valid_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, object]: ...


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    policy_top1: float
    policy_top5: float
    policy_nll: float
    value_mae: float
    positions: int

    def as_wandb(self, global_step: int) -> dict[str, float | int]:
        return {
            "global_step": global_step,
            "validation/policy_top1": self.policy_top1,
            "validation/policy_top5": self.policy_top5,
            "validation/policy_nll": self.policy_nll,
            "validation/value_mae": self.value_mae,
            "validation/positions": self.positions,
        }


@dataclass(frozen=True, slots=True)
class ValidationPlan:
    batch_size: int
    maximum_positions: int
    seed: int


def evaluate_validation(
    network: ValidationNetwork,
    trajectories: Sequence[Trajectory],
    plan: ValidationPlan,
) -> ValidationMetrics:
    positions = _validation_positions(trajectories, plan.maximum_positions, plan.seed)
    totals = np.zeros(4, dtype=np.float64)
    for start in range(0, len(positions), plan.batch_size):
        batch = positions[start : start + plan.batch_size]
        totals += _evaluate_batch(network, batch)
    count = len(positions)
    averages = totals / count
    return ValidationMetrics(
        policy_top1=float(averages[0]),
        policy_top5=float(averages[1]),
        policy_nll=float(averages[2]),
        value_mae=float(averages[3]),
        positions=count,
    )


def _evaluate_batch(network: ValidationNetwork, batch: list[tuple[Trajectory, int]]) -> np.ndarray:
    observations = np.stack([trajectory.observations[position] for trajectory, position in batch])
    valids = np.stack([trajectory.valids[position] for trajectory, position in batch])
    policies, values, _latent = network.batched_initial_inference(observations, valids)
    actions = np.asarray([trajectory.actions[position] for trajectory, position in batch], dtype=np.int64)
    targets = np.asarray([trajectory.root_values[position] for trajectory, position in batch], dtype=np.float32)
    return _batch_totals(policies, values, (actions, targets))


def _validation_positions(
    trajectories: Sequence[Trajectory],
    maximum_positions: int,
    seed: int,
) -> list[tuple[Trajectory, int]]:
    positions = [(trajectory, position) for trajectory in trajectories for position in range(trajectory.game_length)]
    if not positions:
        raise ValueError("Validation trajectories contain no positions")
    if len(positions) <= maximum_positions:
        return positions
    selected = np.random.default_rng(seed).choice(len(positions), size=maximum_positions, replace=False)
    return [positions[index] for index in np.sort(selected)]


def _batch_totals(
    policies: np.ndarray,
    values: np.ndarray,
    labels: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    actions, targets = labels
    rows = np.arange(len(actions))
    selected_probabilities = np.clip(policies[rows, actions], np.finfo(np.float32).tiny, 1.0)
    top_k = min(5, policies.shape[1])
    top_indices = np.argpartition(policies, -top_k, axis=1)[:, -top_k:]
    return np.asarray(
        [
            np.count_nonzero(np.argmax(policies, axis=1) == actions),
            np.count_nonzero(np.any(top_indices == actions[:, None], axis=1)),
            -np.log(selected_probabilities).sum(dtype=np.float64),
            np.abs(values.reshape(-1) - targets).sum(dtype=np.float64),
        ],
        dtype=np.float64,
    )
