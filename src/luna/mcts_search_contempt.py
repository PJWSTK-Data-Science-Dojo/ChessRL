"""Opponent-node sampling for Search-contempt MCTS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from luna.config import MCTSParams
from luna.mcts_gumbel import _interior_best_action
from luna.mcts_tree import _LatentNode


@dataclass(frozen=True, slots=True)
class SearchContemptStats:
    opponent_selections: int = 0
    thompson_selections: int = 0
    frozen_nodes: int = 0


@dataclass(frozen=True, slots=True)
class _FrozenVisitDistribution:
    actions: np.ndarray
    probabilities: np.ndarray

    def sample(self) -> int:
        return int(np.random.choice(self.actions, p=self.probabilities))


class SearchContemptState:
    """Frozen opponent policies owned by one root search."""

    __slots__ = ("_distributions", "_frozen_nodes", "_opponent_selections", "_thompson_selections", "_visit_limit")

    def __init__(self, visit_limit: int | None) -> None:
        self._visit_limit = visit_limit
        self._distributions: dict[_LatentNode, _FrozenVisitDistribution] = {}
        self._frozen_nodes = 0
        self._opponent_selections = 0
        self._thompson_selections = 0

    @property
    def stats(self) -> SearchContemptStats:
        return SearchContemptStats(
            self._opponent_selections,
            self._thompson_selections,
            self._frozen_nodes,
        )

    def select_action(self, node: _LatentNode, depth: int, params: MCTSParams) -> int:
        if self._visit_limit is not None and depth % 2 == 1:
            self._opponent_selections += 1
        distribution = self._opponent_distribution(node, depth)
        if distribution is None:
            return _interior_best_action(node, params)
        self._thompson_selections += 1
        return distribution.sample()

    def _opponent_distribution(
        self,
        node: _LatentNode,
        depth: int,
    ) -> _FrozenVisitDistribution | None:
        if self._visit_limit is None or depth % 2 == 0:
            return None
        frozen = self._distributions.get(node)
        if frozen is not None:
            return frozen
        if node.total_child_visits < self._visit_limit:
            return None
        if node.total_child_visits > self._visit_limit:
            raise RuntimeError("Search-contempt missed its opponent-node freeze threshold")
        frozen = _freeze_visit_distribution(node)
        self._distributions[node] = frozen
        self._frozen_nodes += 1
        return frozen


def _freeze_visit_distribution(node: _LatentNode) -> _FrozenVisitDistribution:
    visited = [(action, child.visit_count) for action, child in node.children.items() if child.visit_count > 0]
    if not visited:
        raise RuntimeError("Search-contempt cannot freeze an unvisited opponent node")
    actions = np.fromiter((action for action, _count in visited), dtype=np.int32)
    counts = np.fromiter((count for _action, count in visited), dtype=np.float64)
    return _FrozenVisitDistribution(actions, counts / float(counts.sum()))
