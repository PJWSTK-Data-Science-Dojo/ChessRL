"""Gumbel MuZero selection and policy-improvement primitives."""

from __future__ import annotations

import math

import numpy as np

from luna.config import MCTSParams
from luna.mcts_tree import EPS, _LatentNode, _puct_best_action


def _get_sequence_of_considered_visits(max_num_considered_actions: int, num_simulations: int) -> tuple[int, ...]:
    """Return the exact Sequential Halving visit schedule used by Gumbel MuZero.

    At simulation ``i``, only root actions with the returned visit count are
    eligible. The first round therefore samples top-m actions without
    replacement; later rounds repeatedly halve the surviving set.
    """
    if max_num_considered_actions <= 0:
        raise ValueError("max_num_considered_actions must be positive")
    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive")
    if max_num_considered_actions == 1:
        return tuple(range(num_simulations))

    log2max = math.ceil(math.log2(max_num_considered_actions))
    sequence: list[int] = []
    visits = [0] * max_num_considered_actions
    num_considered = max_num_considered_actions
    while len(sequence) < num_simulations:
        extra_visits = max(1, int(num_simulations / (log2max * num_considered)))
        for _ in range(extra_visits):
            sequence.extend(visits[:num_considered])
            for i in range(num_considered):
                visits[i] += 1
        num_considered = max(2, num_considered // 2)
    return tuple(sequence[:num_simulations])


def _child_statistics(
    node: _LatentNode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return actions, normalized priors, visits, and mean Q in child order."""
    size = len(node.children)
    actions = np.empty(size, dtype=np.int32)
    priors = np.empty(size, dtype=np.float64)
    visits = np.empty(size, dtype=np.int64)
    qvalues = np.zeros(size, dtype=np.float64)
    for i, (action, child) in enumerate(node.children.items()):
        actions[i] = int(action)
        priors[i] = max(0.0, float(child.prior))
        visits[i] = int(child.visit_count)
        if child.visit_count > 0:
            qvalues[i] = child.value_sum / child.visit_count

    prior_sum = float(priors.sum())
    if size and prior_sum > 0.0:
        priors /= prior_sum
    elif size:
        priors.fill(1.0 / size)
    return actions, priors, visits, qvalues


def _completed_qvalues(node: _LatentNode, params: MCTSParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Complete and transform child Q values as specified by Gumbel MuZero.

    Unvisited actions receive the prior-weighted mixed value. Values are then
    min-max normalized and scaled by ``(maxvisit_init + max_visits) * value_scale``.
    """
    actions, priors, visits, qvalues = _child_statistics(node)
    if not len(actions):
        return actions, priors, visits, qvalues

    visited = visits > 0
    sum_visits = int(visits.sum())
    if np.any(visited):
        safe_priors = np.maximum(priors, np.finfo(np.float64).tiny)
        visited_prior_mass = float(safe_priors[visited].sum())
        weighted_q = float(np.dot(safe_priors[visited], qvalues[visited])) / visited_prior_mass
    else:
        weighted_q = float(node.raw_value)
    mixed_value = (float(node.raw_value) + sum_visits * weighted_q) / (sum_visits + 1)
    completed = np.where(visited, qvalues, mixed_value)

    min_q = float(completed.min())
    max_q = float(completed.max())
    normalized = (completed - min_q) / max(max_q - min_q, EPS)
    visit_scale = params.gumbel_maxvisit_init + float(visits.max())
    transformed = visit_scale * params.gumbel_value_scale * normalized
    return actions, priors, visits, transformed


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    probabilities = np.exp(shifted)
    return probabilities / float(probabilities.sum())


def _gumbel_interior_best_action(node: _LatentNode, params: MCTSParams) -> int:
    """Deterministically match visits to the current improved policy."""
    actions, priors, visits, completed_q = _completed_qvalues(node, params)
    if len(actions) == 0:
        return -1
    if len(actions) == 1:
        return int(actions[0])
    prior_logits = np.log(np.maximum(priors, np.finfo(np.float64).tiny))
    improved_policy = _softmax(prior_logits + completed_q)
    scores = improved_policy - visits / (1.0 + float(visits.sum()))
    return int(actions[int(np.argmax(scores))])


def _interior_best_action(node: _LatentNode, params: MCTSParams) -> int:
    if params.search_mode == "gumbel":
        return _gumbel_interior_best_action(node, params)
    return _puct_best_action(params.cpuct, params.pb_c_base, node)


def _gumbel_improved_policy(root: _LatentNode, action_size: int, params: MCTSParams) -> np.ndarray:
    """Produce the non-degenerate completed-Q policy target from a search tree."""
    policy = np.zeros(action_size, dtype=np.float32)
    actions, priors, _visits, completed_q = _completed_qvalues(root, params)
    if len(actions) == 0:
        return policy
    prior_logits = np.log(np.maximum(priors, np.finfo(np.float64).tiny))
    policy[actions] = _softmax(prior_logits + completed_q).astype(np.float32, copy=False)
    return policy


class _GumbelRootState:
    """Per-root Gumbel sample and Sequential Halving budget schedule."""

    __slots__ = ("actions", "gumbel", "params", "prior_logits", "schedule")

    def __init__(
        self,
        root: _LatentNode,
        num_simulations: int,
        params: MCTSParams,
        *,
        add_noise: bool,
    ) -> None:
        actions, priors, _visits, _qvalues = _child_statistics(root)
        num_considered = min(params.gumbel_max_considered_actions, len(actions))
        if num_considered <= 0:
            raise ValueError("Gumbel root requires at least one legal action")
        self.actions = actions
        self.params = params
        self.prior_logits = np.log(np.maximum(priors, np.finfo(np.float64).tiny))
        if add_noise and params.gumbel_scale > 0.0:
            self.gumbel = params.gumbel_scale * np.random.gumbel(size=len(actions))
        else:
            self.gumbel = np.zeros(len(actions), dtype=np.float64)
        self.schedule = _get_sequence_of_considered_visits(num_considered, num_simulations)

    def select_action(self, root: _LatentNode) -> int:
        """Choose the next root action under the current halving round."""
        actions, _priors, visits, completed_q = _completed_qvalues(root, self.params)
        if not np.array_equal(actions, self.actions):
            raise RuntimeError("root children changed during Sequential Halving")
        simulation_index = int(visits.sum())
        if simulation_index >= len(self.schedule):
            raise RuntimeError("Sequential Halving simulation budget was exceeded")
        considered_visit = self.schedule[simulation_index]
        eligible = visits == considered_visit
        if not np.any(eligible):
            raise RuntimeError("Sequential Halving visit schedule is inconsistent with the tree")
        scores = np.maximum(-1e9, self.gumbel + self.prior_logits + completed_q)
        scores = np.where(eligible, scores, -np.inf)
        return int(actions[int(np.argmax(scores))])

    def proposed_action(self, root: _LatentNode) -> int:
        """Return the Gumbel MuZero action proposed after Sequential Halving."""
        actions, _priors, visits, completed_q = _completed_qvalues(root, self.params)
        if not np.array_equal(actions, self.actions):
            raise RuntimeError("root children changed during Sequential Halving")
        considered_visit = int(visits.max())
        eligible = visits == considered_visit
        scores = np.maximum(-1e9, self.gumbel + self.prior_logits + completed_q)
        scores = np.where(eligible, scores, -np.inf)
        return int(actions[int(np.argmax(scores))])
