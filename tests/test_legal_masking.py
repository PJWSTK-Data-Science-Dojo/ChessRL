"""Tests for legal move masking in latent MCTS."""

import numpy as np

from luna.config import MCTSParams
from luna.mcts import BatchedMCTS
from luna.network import LunaNetwork


def test_batched_mcts_expansion_with_boards(chess_game, small_learner_config):
    """Batched MCTS should track boards and compute valid masks for all positions."""
    nnet = LunaNetwork(chess_game, small_learner_config)
    params = MCTSParams(num_mcts_sims=3, dir_noise=False, recurrent_policy_topk=None)
    batched_mcts = BatchedMCTS(chess_game, nnet, params)

    board1 = chess_game.get_init_board()
    board2 = chess_game.get_init_board()

    results = batched_mcts.search_batch([board1, board2], num_sims=3)

    assert len(results) == 2
    for policy, root_value, obs, valids in results:
        assert len(policy) == chess_game.get_action_size()
        assert isinstance(root_value, float)
        assert obs.shape[-1] == chess_game.get_board_size()[2]
        assert len(valids) == chess_game.get_action_size()


def test_get_next_state_handles_illegal_action(chess_game):
    """get_next_state should fall back to legal move if action is illegal."""
    board = chess_game.get_init_board()

    illegal_action = 9999

    next_board, next_player = chess_game.get_next_state(board, 1, illegal_action)

    assert next_board.fen() != board.fen()
    assert next_player == -1
