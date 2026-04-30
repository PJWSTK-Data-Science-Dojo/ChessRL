"""Tests for MCTS search (single-game and batched)."""

from __future__ import annotations

import numpy as np

from luna.config import EzV2LearnerConfig, MCTSParams
from luna.game.chess_game import ChessGame
from luna.mcts import MCTS, BatchedMCTS
from luna.network import LunaNetwork


def _make_mcts(num_sims: int = 5) -> tuple[MCTS, ChessGame, LunaNetwork]:
    game = ChessGame()
    learner = EzV2LearnerConfig(num_channels=32, repr_blocks=2, dyn_blocks=1, proj_dim=64)
    nnet = LunaNetwork(game, learner)
    params = MCTSParams(
        num_mcts_sims=num_sims,
        cpuct=1.25,
        dir_noise=False,
        dir_alpha=0.3,
        discount=0.997,
        recurrent_policy_topk=None,
    )
    return MCTS(game, nnet, params), game, nnet


class TestLatentSearch:
    def test_returns_valid_policy(self) -> None:
        mcts, game, _ = _make_mcts(num_sims=3)
        board = game.get_init_board()
        canonical = game.get_canonical_form(board, 1)
        probs, root_v = mcts.search_latent(canonical, num_sims=3)
        assert len(probs) == game.get_action_size()
        assert abs(sum(probs) - 1.0) < 1e-5
        assert isinstance(root_v, float)

    def test_get_action_prob_uses_latent_search(self) -> None:
        mcts, game, _ = _make_mcts(num_sims=3)
        board = game.get_init_board()
        canonical = game.get_canonical_form(board, 1)
        p_latent, _ = mcts.search_latent(canonical, num_sims=3, temp=1.0)
        p_get = mcts.get_action_prob(canonical, temp=1.0)
        assert len(p_latent) == len(p_get)
        assert abs(sum(p_get) - 1.0) < 1e-5

    def test_deterministic_at_temp_zero(self) -> None:
        mcts, game, _ = _make_mcts(num_sims=3)
        board = game.get_init_board()
        canonical = game.get_canonical_form(board, 1)
        probs = mcts.get_action_prob(canonical, temp=0)
        assert sum(1 for p in probs if p > 0) == 1


class TestBatchedMCTS:
    def test_search_batch_returns_correct_count(self) -> None:
        _, game, nnet = _make_mcts(num_sims=3)
        params = MCTSParams(
            num_mcts_sims=3,
            cpuct=1.25,
            dir_noise=False,
            dir_alpha=0.3,
            discount=0.997,
            recurrent_policy_topk=None,
        )
        bmcts = BatchedMCTS(game, nnet, params)
        boards = [game.get_init_board() for _ in range(4)]
        canonicals = [game.get_canonical_form(b, 1) for b in boards]
        results = bmcts.search_batch(canonicals, num_sims=3)
        assert len(results) == 4
        for probs, root_v, obs, valid in results:
            assert probs.shape == (game.get_action_size(),)
            assert probs.dtype == np.float32
            assert abs(float(probs.sum()) - 1.0) < 1e-5
            assert isinstance(root_v, float)
            assert obs.shape == game.get_board_size()
            assert valid.shape == (game.get_action_size(),)

    def test_batch_deterministic_at_temp_zero(self) -> None:
        _, game, nnet = _make_mcts(num_sims=3)
        params = MCTSParams(
            num_mcts_sims=3,
            cpuct=1.25,
            dir_noise=False,
            dir_alpha=0.3,
            discount=0.997,
            recurrent_policy_topk=None,
        )
        bmcts = BatchedMCTS(game, nnet, params)
        boards = [game.get_init_board(), game.get_init_board()]
        canonicals = [game.get_canonical_form(b, 1) for b in boards]
        results = bmcts.search_batch(canonicals, num_sims=3, temp=0)
        for probs, _, _obs, _valid in results:
            assert int((probs > 0).sum()) == 1
