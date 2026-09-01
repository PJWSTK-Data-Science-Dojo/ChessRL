"""Tests for MCTS search (single-game and batched)."""

import numpy as np

from luna.config import MCTSParams
from luna.mcts import (
    _backup_latent_path,
    _completed_qvalues,
    _get_sequence_of_considered_visits,
    _gumbel_improved_policy,
    _GumbelRootState,
    _LatentNode,
)


def _root_with_priors(priors: list[float], actions: list[int] | None = None) -> _LatentNode:
    root = _LatentNode(prior=0.0)
    root.expanded = True
    root.raw_value = 0.0
    if actions is None:
        actions = list(range(len(priors)))
    for action, prior in zip(actions, priors, strict=True):
        root.children[action] = _LatentNode(prior=prior)
    return root


class TestGumbelMuZero:
    def test_default_mode_is_gumbel(self) -> None:
        assert MCTSParams().search_mode == "gumbel"

    def test_sequential_halving_respects_budget_and_top_m(self) -> None:
        params = MCTSParams(
            search_mode="gumbel",
            gumbel_max_considered_actions=4,
            gumbel_scale=1.0,
        )
        root = _root_with_priors([0.40, 0.30, 0.20, 0.09, 0.005, 0.005])
        state = _GumbelRootState(root, 12, params, add_noise=False)
        quality = [1.0, 0.5, 0.0, -0.5, -1.0, -1.0]

        assert state.schedule == _get_sequence_of_considered_visits(4, 12)
        assert state.schedule == (0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4)
        assert not np.any(state.gumbel)

        for _ in range(12):
            action = state.select_action(root)
            child = root.children[action]
            child.visit_count += 1
            child.value_sum += quality[action]
            root.total_child_visits += 1

        visits = np.array([child.visit_count for child in root.children.values()])
        assert int(visits.sum()) == 12
        np.testing.assert_array_equal(visits, [5, 5, 1, 1, 0, 0])
        assert state.proposed_action(root) == 0

    def test_completed_q_policy_is_soft_legal_and_improved(self) -> None:
        params = MCTSParams(
            gumbel_value_scale=0.1,
            gumbel_maxvisit_init=50.0,
        )
        root = _root_with_priors([0.5, 0.3, 0.2], actions=[2, 5, 7])
        root.raw_value = 0.1
        root.children[2].visit_count = 2
        root.children[2].value_sum = 1.6
        root.children[5].visit_count = 1
        root.children[5].value_sum = -0.2

        _actions, _priors, _visits, transformed_q = _completed_qvalues(root, params)
        policy = _gumbel_improved_policy(root, action_size=10, params=params)

        np.testing.assert_allclose(transformed_q, [5.2, 0.0, 2.8275], atol=1e-10)
        assert abs(float(policy.sum()) - 1.0) < 1e-6
        assert np.all(policy[[2, 5, 7]] > 0)
        assert np.count_nonzero(np.delete(policy, [2, 5, 7])) == 0
        assert policy[2] > 0.5
        assert np.count_nonzero(policy) == 3

    def test_backup_alternates_player_perspective_at_every_depth(self) -> None:
        root = _LatentNode(0.0)
        middle = _LatentNode(1.0)
        parent = _LatentNode(1.0)
        leaf = _LatentNode(1.0)
        middle.reward = 0.1
        parent.reward = 0.2

        _backup_latent_path([root, middle, parent], leaf, q_leaf=0.4, discount=0.5)

        assert leaf.value() == 0.4
        assert parent.value() == 0.0
        assert middle.value() == 0.1
        assert root.total_child_visits == 1
