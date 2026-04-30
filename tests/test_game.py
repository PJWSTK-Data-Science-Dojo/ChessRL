"""Tests for chess game wrapper -- reward perspective, legal moves, action encoding."""

from __future__ import annotations

from luna.game.chess_game import ChessGame, action_to_move, move_to_action, player_from_turn


class TestRewardPerspective:
    """Verify get_game_ended returns correct sign relative to the *player* argument."""

    def test_ongoing_game_returns_zero(self) -> None:
        g = ChessGame()
        board = g.get_init_board()
        assert g.get_game_ended(board, 1) == 0.0
        assert g.get_game_ended(board, -1) == 0.0

    def test_checkmate_perspective(self) -> None:
        """Scholars mate: white wins. After Qf7# it is black's turn but game is over."""
        import chess

        g = ChessGame()
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]:
            board.push(chess.Move.from_uci(uci))

        assert board.is_checkmate()
        assert g.get_game_ended(board, 1) == 1.0
        assert g.get_game_ended(board, -1) == -1.0


class TestValidMoves:
    def test_initial_position_has_20_moves(self) -> None:
        g = ChessGame()
        board = g.get_init_board()
        valids = g.get_valid_moves(board, 1)
        assert int(valids.sum()) == 20

    def test_canonical_preserves_move_count(self) -> None:
        g = ChessGame()
        board = g.get_init_board()
        board.push_uci("e2e4")
        canonical = g.get_canonical_form(board, player_from_turn(board.turn))
        valids = g.get_valid_moves(canonical, 1)
        assert int(valids.sum()) == 20


class TestActionRoundtrip:
    def test_action_encoding_roundtrip(self) -> None:
        import chess

        for move in chess.Board().legal_moves:
            action = move_to_action(move)
            recovered = action_to_move(action)
            assert recovered.from_square == move.from_square
            assert recovered.to_square == move.to_square

    def test_promotion_handled_by_get_next_state(self) -> None:
        """Queen promotion auto-detected via get_next_state; underpromotions try all types."""
        import chess

        g = ChessGame()
        board = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")
        action = move_to_action(chess.Move.from_uci("a7a8"))
        new_board, _player = g.get_next_state(board, 1, action)
        assert new_board.piece_at(chess.A8).piece_type == chess.QUEEN

    def test_action_size(self) -> None:
        g = ChessGame()
        assert g.get_action_size() == 4288

    def test_underpromotion_roundtrip(self) -> None:
        import chess

        move = chess.Move.from_uci("a7a8n")
        action = move_to_action(move)
        recovered = action_to_move(action)
        assert recovered.from_square == move.from_square
        assert recovered.to_square == move.to_square
        assert recovered.promotion == chess.KNIGHT

    def test_underpromotion_get_next_state(self) -> None:
        import chess

        g = ChessGame()
        board = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")
        knight_move = chess.Move.from_uci("a7a8n")
        action = move_to_action(knight_move)
        new_board, _player = g.get_next_state(board, 1, action)
        assert new_board.piece_at(chess.A8).piece_type == chess.KNIGHT
