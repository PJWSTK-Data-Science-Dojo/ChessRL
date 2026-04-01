"""Tests for chess game wrapper -- reward perspective and legal move consistency."""

from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from luna.game.luna_game import ChessGame, who


class TestRewardPerspective:
    """Verify getGameEnded returns correct sign relative to the *player* argument."""

    def test_ongoing_game_returns_zero(self) -> None:
        g = ChessGame()
        board = g.getInitBoard()
        assert g.getGameEnded(board, 1) == 0.0
        assert g.getGameEnded(board, -1) == 0.0

    def test_checkmate_perspective(self) -> None:
        """Scholars mate: white wins. After Qf7# it is black's turn but game is over."""
        import chess

        g = ChessGame()
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]:
            board.push(chess.Move.from_uci(uci))

        assert board.is_checkmate()
        assert g.getGameEnded(board, 1) == 1.0
        assert g.getGameEnded(board, -1) == -1.0


class TestValidMoves:
    def test_initial_position_has_20_moves(self) -> None:
        g = ChessGame()
        board = g.getInitBoard()
        valids = g.getValidMoves(board, 1)
        assert int(valids.sum()) == 20

    def test_canonical_preserves_move_count(self) -> None:
        g = ChessGame()
        board = g.getInitBoard()
        board.push_uci("e2e4")
        canonical = g.getCanonicalForm(board, who(board.turn))
        valids = g.getValidMoves(canonical, 1)
        assert int(valids.sum()) == 20


class TestActionRoundtrip:
    def test_action_encoding_roundtrip(self) -> None:
        from luna.game.luna_game import from_move, to_move

        import chess

        for move in chess.Board().legal_moves:
            action = from_move(move)
            recovered = to_move(action)
            assert recovered.from_square == move.from_square
            assert recovered.to_square == move.to_square
