"""Tests for chess game wrapper -- reward perspective, legal moves, action encoding."""

import chess
import pytest

from luna.game.chess_game import ChessGame, action_to_move, move_to_action, player_from_turn


class TestRewardPerspective:
    """Verify get_game_ended returns correct sign relative to the *player* argument."""

    def test_ongoing_game_returns_zero(self, chess_game):
        board = chess_game.get_init_board()
        assert chess_game.get_game_ended(board, 1) == 0.0
        assert chess_game.get_game_ended(board, -1) == 0.0

    def test_checkmate_perspective(self, chess_game):
        """Scholars mate: white wins. After Qf7# it is black's turn but game is over."""
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]:
            board.push(chess.Move.from_uci(uci))

        assert board.is_checkmate()
        assert chess_game.get_game_ended(board, 1) == 1.0
        assert chess_game.get_game_ended(board, -1) == -1.0


class TestValidMoves:
    def test_initial_position_has_20_moves(self, chess_game):
        board = chess_game.get_init_board()
        valids = chess_game.get_valid_moves(board, 1)
        assert int(valids.sum()) == 20


@pytest.mark.parametrize("uci,expected_piece", [
    ("a7a8", chess.QUEEN),
    ("a7a8n", chess.KNIGHT),
])
def test_action_encoding_with_promotions(uci, expected_piece, chess_game):
    """Action encoding roundtrip preserves from/to/promotion."""
    board = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")

    move = chess.Move.from_uci(uci)
    action = move_to_action(move)
    recovered = action_to_move(action)

    assert recovered.from_square == move.from_square
    assert recovered.to_square == move.to_square
    if move.promotion:
        assert recovered.promotion == move.promotion

    new_board, _player = chess_game.get_next_state(board, 1, action)
    assert new_board.piece_at(chess.A8).piece_type == expected_piece
