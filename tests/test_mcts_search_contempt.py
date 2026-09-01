"""Tests for the opponent-node Search-contempt overlay."""

from typing import Literal

import numpy as np
import pytest

from luna.config import MCTSParams, validate_mcts_params
from luna.mcts_gumbel import _interior_best_action
from luna.mcts_search_contempt import SearchContemptState
from luna.mcts_tree import _LatentNode


def _node_with_visits(counts: list[int]) -> _LatentNode:
    node = _LatentNode(prior=0.0)
    node.expanded = True
    node.raw_value = 0.0
    for action, count in enumerate(counts):
        child = _LatentNode(prior=1.0 / len(counts))
        child.visit_count = count
        child.value_sum = float(count)
        node.children[action] = child
    node.total_child_visits = sum(counts)
    return node


def test_odd_depth_freezes_after_limit_base_visits_and_samples_the_next_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = MCTSParams(search_mode="puct", search_contempt_visit_limit=4)
    node = _node_with_visits([3, 1, 0])
    state = SearchContemptState(visit_limit=4)
    sampled: list[tuple[np.ndarray, np.ndarray]] = []

    def choose(actions: np.ndarray, *, p: np.ndarray) -> np.int32:
        sampled.append((actions.copy(), p.copy()))
        return actions[-1]

    monkeypatch.setattr(np.random, "choice", choose)
    selected = state.select_action(node, depth=1, params=params)

    assert selected == 1
    assert len(sampled) == 1
    np.testing.assert_array_equal(sampled[0][0], [0, 1])
    np.testing.assert_allclose(sampled[0][1], [0.75, 0.25])
    assert state.stats.opponent_selections == 1
    assert state.stats.thompson_selections == 1
    assert state.stats.frozen_nodes == 1


def test_frozen_distribution_does_not_follow_later_visit_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    params = MCTSParams(search_mode="puct", search_contempt_visit_limit=4)
    node = _node_with_visits([3, 1, 0])
    state = SearchContemptState(visit_limit=4)
    captured_probabilities: list[np.ndarray] = []

    def choose(actions: np.ndarray, *, p: np.ndarray) -> np.int32:
        captured_probabilities.append(p.copy())
        return actions[0]

    monkeypatch.setattr(np.random, "choice", choose)
    state.select_action(node, depth=1, params=params)
    node.children[2].visit_count = 100
    node.total_child_visits = 104
    state.select_action(node, depth=1, params=params)

    np.testing.assert_allclose(captured_probabilities, [[0.75, 0.25], [0.75, 0.25]])


@pytest.mark.parametrize("depth", [0, 2, 4])
def test_even_depth_always_uses_base_selector(depth: int) -> None:
    params = MCTSParams(search_mode="puct", search_contempt_visit_limit=1)
    node = _node_with_visits([10, 2])
    state = SearchContemptState(visit_limit=1)

    selected = state.select_action(node, depth=depth, params=params)

    assert selected == _interior_best_action(node, params)
    assert state.stats.opponent_selections == 0
    assert state.stats.thompson_selections == 0
    assert state.stats.frozen_nodes == 0


@pytest.mark.parametrize("search_mode", ["gumbel", "puct"])
def test_disabled_overlay_is_an_exact_rng_free_delegate(search_mode: Literal["gumbel", "puct"]) -> None:
    params = MCTSParams(search_mode=search_mode, search_contempt_visit_limit=None)
    node = _node_with_visits([3, 1])
    state = SearchContemptState(visit_limit=None)
    before = np.random.get_state()

    selected = state.select_action(node, depth=1, params=params)
    after = np.random.get_state()

    assert selected == _interior_best_action(node, params)
    assert all(
        np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
        for left, right in zip(before, after, strict=True)
    )
    assert state.stats.opponent_selections == 0


@pytest.mark.parametrize("visit_limit", [None, 1, 32])
def test_valid_visit_limits(visit_limit: int | None) -> None:
    validate_mcts_params(MCTSParams(search_contempt_visit_limit=visit_limit))


@pytest.mark.parametrize("visit_limit", [0, -1, True])
def test_invalid_visit_limits(visit_limit: int) -> None:
    with pytest.raises(ValueError, match="search_contempt_visit_limit"):
        validate_mcts_params(MCTSParams(search_contempt_visit_limit=visit_limit))
