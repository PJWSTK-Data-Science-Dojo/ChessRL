"""Tests for legal move masking in latent MCTS."""

import chess
import pytest

from luna.config import EzV2LearnerConfig, MCTSParams
from luna.game.chess_game import ChessGame
from luna.mcts import BatchedMCTS
from luna.network import LunaNetwork


def test_batched_mcts_expansion_with_boards(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
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


def test_get_next_state_rejects_illegal_action(chess_game: ChessGame) -> None:
    """Illegal actions must fail rather than corrupting action/transition pairs."""
    board = chess_game.get_init_board()

    with pytest.raises(ValueError, match="Illegal action"):
        chess_game.get_next_state(board, 1, 9999)


def test_black_valid_mask_uses_canonical_actions(chess_game: ChessGame) -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    valids = chess_game.get_valid_moves(board, -1)

    # Canonical e7-e5 is mirrored to e2-e4 for the side-to-move representation.
    from luna.game.chess_game import move_to_action

    assert valids[move_to_action(chess.Move.from_uci("e2e4"))] == 1.0
