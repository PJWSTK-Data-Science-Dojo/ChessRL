"""Shared latent-tree primitives for serial and batched MCTS."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from typing import cast

import chess
import numpy as np
import torch

from luna.config import MCTSParams

_PuctArgmax = Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], int]


def _puct_argmax_impl(
    exploration_scale: float,
    actions: np.ndarray,
    priors: np.ndarray,
    visits: np.ndarray,
    vsum: np.ndarray,
) -> int:
    n = visits.shape[0]
    ucb = np.empty(n, dtype=np.float64)
    for i in range(n):
        visit_count = visits[i]
        if visit_count == 0.0:
            ucb[i] = exploration_scale * priors[i]
        else:
            q_value = vsum[i] / visit_count
            ucb[i] = q_value + exploration_scale * priors[i] / (1.0 + visit_count)
    best_index = 0
    best_value = ucb[0]
    for i in range(1, n):
        if ucb[i] > best_value:
            best_value = ucb[i]
            best_index = i
    return int(actions[best_index])


_puct_argmax_numba: _PuctArgmax = _puct_argmax_impl
try:
    numba_module = importlib.import_module("numba")
    _puct_argmax_numba = cast(_PuctArgmax, numba_module.njit(cache=True)(_puct_argmax_impl))
    _NUMBA_PUCT = True
except (AttributeError, ImportError):
    # Numba is an optional acceleration. Some valid environments expose an
    # incompatible coverage API during Numba import; search must still work.
    _NUMBA_PUCT = False

EPS = 1e-8


def _puct_best_action(cpuct: float, pb_c_base: float, node: _LatentNode) -> int:
    """Pick child with highest PUCT score (vectorized over legal children).

    Matches the tie-breaking of the original per-child Python loop: first child
    among equals wins (dict / array insertion order).
    """
    ch = node.children
    n = len(ch)
    if n == 0:
        return -1
    if n == 1:
        return int(next(iter(ch.keys())))

    sqrt_total = math.sqrt(node.total_child_visits + EPS)
    exploration_scale = (math.log((node.total_child_visits + pb_c_base + 1.0) / pb_c_base) + cpuct) * sqrt_total
    actions = np.empty(n, dtype=np.int32)
    priors = np.empty(n, dtype=np.float64)
    visits = np.empty(n, dtype=np.float64)
    vsum = np.empty(n, dtype=np.float64)
    for i, (a, child) in enumerate(ch.items()):
        actions[i] = int(a)
        priors[i] = child.prior
        visits[i] = child.visit_count
        vsum[i] = child.value_sum

    if _NUMBA_PUCT and n >= 4:
        return int(_puct_argmax_numba(exploration_scale, actions, priors, visits, vsum))

    q = np.divide(vsum, visits, out=np.zeros(n, dtype=np.float64), where=visits > 0)
    ucb0 = exploration_scale * priors
    ucb1 = q + exploration_scale * priors / (1.0 + visits)
    ucb = np.where(visits == 0, ucb0, ucb1)
    return int(actions[int(np.argmax(ucb))])


class _LatentNode:
    """Search node whose edge reward is represented from its parent perspective."""

    __slots__ = (
        "board",
        "children",
        "expanded",
        "latent",
        "prior",
        "raw_value",
        "reward",
        "terminal",
        "total_child_visits",
        "value_sum",
        "visit_count",
    )

    def __init__(self, prior: float, board: chess.Board | None = None) -> None:
        self.prior = prior
        self.value_sum = 0.0
        self.visit_count = 0
        self.total_child_visits = 0
        self.reward = 0.0
        self.raw_value = 0.0
        self.terminal = False
        self.board = board
        self.latent: torch.Tensor | None = None
        self.children: dict[int, _LatentNode] = {}
        self.expanded = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _validate_search(params: MCTSParams, num_sims: int, temp: float) -> None:
    """Fail early on search settings that cannot define a valid policy."""
    if num_sims <= 0:
        raise ValueError("num_sims must be positive")
    if not math.isfinite(temp) or temp < 0:
        raise ValueError("temp must be finite and non-negative")
    if params.search_mode not in {"gumbel", "puct"}:
        raise ValueError(f"unknown MCTS search_mode: {params.search_mode!r}")
    if params.pb_c_base <= 0:
        raise ValueError("pb_c_base must be positive")
    if params.search_mode == "gumbel":
        if params.gumbel_max_considered_actions <= 0:
            raise ValueError("gumbel_max_considered_actions must be positive")
        if params.gumbel_scale < 0:
            raise ValueError("gumbel_scale must be non-negative")
        if params.gumbel_value_scale < 0:
            raise ValueError("gumbel_value_scale must be non-negative")
        if params.gumbel_maxvisit_init < 0:
            raise ValueError("gumbel_maxvisit_init must be non-negative")
    visit_limit = params.search_contempt_visit_limit
    if visit_limit is not None and (
        isinstance(visit_limit, bool) or not isinstance(visit_limit, int) or visit_limit <= 0
    ):
        raise ValueError("search_contempt_visit_limit must be a positive integer when enabled")


def _visit_count_policy(counts: np.ndarray, temp: float) -> np.ndarray:
    """Apply temperature to positive visit counts without overflow."""
    positive = counts > 0
    if not np.any(positive):
        raise ValueError("visit-count policy requires at least one positive count")
    log_counts = np.log(counts[positive])
    with np.errstate(over="ignore"):
        logits = (log_counts - float(log_counts.max())) / temp
    probabilities = np.exp(logits)
    policy = np.zeros_like(counts, dtype=np.float64)
    policy[positive] = probabilities / float(probabilities.sum())
    return policy


class _PendingExpansion:
    __slots__ = ("ancestors", "child")

    def __init__(self, ancestors: list[_LatentNode], child: _LatentNode) -> None:
        self.ancestors = ancestors
        self.child = child


def _backup_latent_path(ancestors: list[_LatentNode], leaf: _LatentNode, q_leaf: float, discount: float) -> None:
    """Back up values with alternating player perspective from leaf to root.

    ``q_leaf`` is the value of the leaf edge from its parent perspective. Each
    preceding edge combines its parent-perspective reward with the negated value
    of the selected continuation at the child node.
    """
    leaf.visit_count += 1
    leaf.value_sum += q_leaf
    ancestors[-1].total_child_visits += 1
    q = q_leaf
    for j in range(len(ancestors) - 1, 0, -1):
        child = ancestors[j]
        parent = ancestors[j - 1]
        q = child.reward - discount * q
        child.visit_count += 1
        child.value_sum += q
        parent.total_child_visits += 1
