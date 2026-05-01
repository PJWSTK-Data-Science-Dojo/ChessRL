"""Tests for MCTS search (single-game and batched)."""

import numpy as np

from luna.config import MCTSParams
from luna.mcts import MCTS, BatchedMCTS
from luna.network import LunaNetwork


class TestLatentSearch:
    def test_returns_valid_policy(self, chess_game, small_learner_config):
        nnet = LunaNetwork(chess_game, small_learner_config)
        params = MCTSParams(num_mcts_sims=3, dir_noise=False, recurrent_policy_topk=None)
        mcts = MCTS(chess_game, nnet, params)

        board = chess_game.get_init_board()
        canonical = chess_game.get_canonical_form(board, 1)
        probs, root_v = mcts.search_latent(canonical, num_sims=3)

        assert len(probs) == chess_game.get_action_size()
        assert abs(sum(probs) - 1.0) < 1e-5
        assert isinstance(root_v, float)

    def test_get_action_prob_uses_latent_search(self, chess_game, small_learner_config):
        nnet = LunaNetwork(chess_game, small_learner_config)
        params = MCTSParams(num_mcts_sims=3, dir_noise=False, recurrent_policy_topk=None)
        mcts = MCTS(chess_game, nnet, params)

        board = chess_game.get_init_board()
        canonical = chess_game.get_canonical_form(board, 1)
        p_latent, _ = mcts.search_latent(canonical, num_sims=3, temp=1.0)
        p_get = mcts.get_action_prob(canonical, temp=1.0)

        assert len(p_latent) == len(p_get)
        assert abs(sum(p_get) - 1.0) < 1e-5

    def test_deterministic_at_temp_zero(self, chess_game, small_learner_config):
        nnet = LunaNetwork(chess_game, small_learner_config)
        params = MCTSParams(num_mcts_sims=3, dir_noise=False, recurrent_policy_topk=None)
        mcts = MCTS(chess_game, nnet, params)

        board = chess_game.get_init_board()
        canonical = chess_game.get_canonical_form(board, 1)
        probs = mcts.get_action_prob(canonical, temp=0)

        assert sum(1 for p in probs if p > 0) == 1


class TestBatchedMCTS:
    def test_search_batch_returns_correct_count(self, chess_game, small_learner_config):
        nnet = LunaNetwork(chess_game, small_learner_config)
        params = MCTSParams(num_mcts_sims=3, dir_noise=False, recurrent_policy_topk=None)
        bmcts = BatchedMCTS(chess_game, nnet, params)

        boards = [chess_game.get_init_board() for _ in range(4)]
        canonicals = [chess_game.get_canonical_form(b, 1) for b in boards]
        results = bmcts.search_batch(canonicals, num_sims=3)

        assert len(results) == 4
        for probs, root_v, obs, valid in results:
            assert probs.shape == (chess_game.get_action_size(),)
            assert probs.dtype == np.float32
            assert abs(float(probs.sum()) - 1.0) < 1e-5
            assert isinstance(root_v, float)
            assert obs.shape == chess_game.get_board_size()
            assert valid.shape == (chess_game.get_action_size(),)

    def test_batch_deterministic_at_temp_zero(self, chess_game, small_learner_config):
        nnet = LunaNetwork(chess_game, small_learner_config)
        params = MCTSParams(num_mcts_sims=3, dir_noise=False, recurrent_policy_topk=None)
        bmcts = BatchedMCTS(chess_game, nnet, params)

        boards = [chess_game.get_init_board(), chess_game.get_init_board()]
        canonicals = [chess_game.get_canonical_form(b, 1) for b in boards]
        results = bmcts.search_batch(canonicals, num_sims=3, temp=0)

        for probs, _, _obs, _valid in results:
            assert int((probs > 0).sum()) == 1
