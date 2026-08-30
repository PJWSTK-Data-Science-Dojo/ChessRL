"""Tests for chess game wrapper -- reward perspective, legal moves, action encoding."""

import chess
import numpy as np
import pytest

from luna.game.chess_game import (
    CASTLING_PLANES_START,
    EN_PASSANT_PLANE,
    HALFMOVE_CLOCK_PLANE,
    HISTORY_LENGTH,
    OBS_PLANES,
    PIECE_PLANES_PER_POSITION,
    PLANES_PER_POSITION,
    SIDE_TO_MOVE_PLANE,
    ChessGame,
    action_to_move,
    board_to_numpy,
    move_to_action,
)


def _piece_plane(history_index: int, color: chess.Color, piece_type: chess.PieceType) -> int:
    color_offset = 0 if color == chess.WHITE else 6
    return history_index * PLANES_PER_POSITION + color_offset + piece_type - 1


class TestObservationEncoding:
    """Verify the temporal 119-plane representation and auxiliary state."""

    def test_shape_dtype_range_and_initial_position(self, chess_game: ChessGame) -> None:
        board = chess_game.get_init_board()
        observation = board_to_numpy(board)

        assert observation.shape == (8, 8, 119)
        assert observation.shape == chess_game.get_board_size()
        assert OBS_PLANES == 119
        assert observation.dtype == np.float32
        assert np.isfinite(observation).all()
        assert observation.min() >= 0.0
        assert observation.max() <= 1.0

        assert observation[chess.square_rank(chess.E1), chess.square_file(chess.E1), 5] == 1.0
        assert observation[chess.square_rank(chess.E8), chess.square_file(chess.E8), 11] == 1.0
        assert observation[:, :, CASTLING_PLANES_START : CASTLING_PLANES_START + 4].sum() == 256
        assert np.all(observation[:, :, SIDE_TO_MOVE_PLANE] == 1.0)

        # A fresh board has no move stack: unavailable historical slots are empty.
        history_tail = observation[:, :, PLANES_PER_POSITION : HISTORY_LENGTH * PLANES_PER_POSITION]
        assert np.count_nonzero(history_tail) == 0

    def test_current_and_previous_positions_are_distinct(self) -> None:
        board = chess.Board()
        board.push_uci("e2e4")
        observation = board_to_numpy(board)

        current_white_pawn = _piece_plane(0, chess.WHITE, chess.PAWN)
        previous_white_pawn = _piece_plane(1, chess.WHITE, chess.PAWN)
        e2 = (chess.square_rank(chess.E2), chess.square_file(chess.E2))
        e4 = (chess.square_rank(chess.E4), chess.square_file(chess.E4))

        assert observation[*e4, current_white_pawn] == 1.0
        assert observation[*e2, current_white_pawn] == 0.0
        assert observation[*e2, previous_white_pawn] == 1.0
        assert observation[*e4, previous_white_pawn] == 0.0
        assert observation[chess.square_rank(chess.E3), chess.square_file(chess.E3), EN_PASSANT_PLANE] == 1.0
        assert np.all(observation[:, :, SIDE_TO_MOVE_PLANE] == 0.0)

    def test_piece_colors_use_separate_binary_planes(self) -> None:
        board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 17 42")
        observation = board_to_numpy(board)
        white_pawns = _piece_plane(0, chess.WHITE, chess.PAWN)
        black_pawns = _piece_plane(0, chess.BLACK, chess.PAWN)

        assert observation[chess.square_rank(chess.E4), chess.square_file(chess.E4), white_pawns] == 1.0
        assert observation[chess.square_rank(chess.D5), chess.square_file(chess.D5), black_pawns] == 1.0
        assert observation[:, :, white_pawns].sum() == 1.0
        assert observation[:, :, black_pawns].sum() == 1.0
        assert np.allclose(observation[:, :, HALFMOVE_CLOCK_PLANE], 0.17)

    def test_repetition_planes_are_temporal(self) -> None:
        board = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
            board.push_uci(uci)

        observation = board_to_numpy(board)
        current_twice = PIECE_PLANES_PER_POSITION
        current_thrice = PIECE_PLANES_PER_POSITION + 1
        previous_twice = PLANES_PER_POSITION + PIECE_PLANES_PER_POSITION

        assert board.is_repetition(2)
        assert np.all(observation[:, :, current_twice] == 1.0)
        assert np.all(observation[:, :, current_thrice] == 0.0)
        assert np.all(observation[:, :, previous_twice] == 0.0)

    def test_black_canonical_observation_preserves_history(self, chess_game: ChessGame) -> None:
        board = chess.Board()
        board.push_uci("e2e4")
        canonical = chess_game.get_canonical_form(board, -1)
        observation = chess_game.to_array(canonical)

        current_black_pawn = _piece_plane(0, chess.BLACK, chess.PAWN)
        previous_black_pawn = _piece_plane(1, chess.BLACK, chess.PAWN)
        e5 = (chess.square_rank(chess.E5), chess.square_file(chess.E5))
        e7 = (chess.square_rank(chess.E7), chess.square_file(chess.E7))

        assert canonical.turn == chess.WHITE
        assert len(canonical.move_stack) == 1
        assert observation[*e5, current_black_pawn] == 1.0
        assert observation[*e7, current_black_pawn] == 0.0
        assert observation[*e7, previous_black_pawn] == 1.0
        assert observation[*e5, previous_black_pawn] == 0.0
        assert np.all(observation[:, :, SIDE_TO_MOVE_PLANE] == 1.0)


class TestRewardPerspective:
    """Verify exact outcomes relative to the requested player."""

    def test_ongoing_game_returns_none(self, chess_game: ChessGame) -> None:
        board = chess_game.get_init_board()
        assert chess_game.get_game_outcome(board, 1) is None
        assert chess_game.get_game_outcome(board, -1) is None

    def test_checkmate_perspective(self, chess_game: ChessGame) -> None:
        """Scholars mate: white wins. After Qf7# it is black's turn but game is over."""
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]:
            board.push(chess.Move.from_uci(uci))

        assert board.is_checkmate()
        assert chess_game.get_game_outcome(board, 1) == 1.0
        assert chess_game.get_game_outcome(board, -1) == -1.0

    def test_claimable_repetition_is_an_exact_draw(self, chess_game: ChessGame) -> None:
        board = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
            board.push_uci(uci)

        assert board.can_claim_threefold_repetition()
        assert chess_game.get_game_outcome(board, 1) == 0.0

    def test_black_canonical_form_preserves_repetition_history(self, chess_game: ChessGame) -> None:
        board = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
            board.push_uci(uci)

        assert board.turn == chess.BLACK
        canonical = chess_game.get_canonical_form(board, -1)
        assert canonical.turn == chess.WHITE
        assert len(canonical.move_stack) == len(board.move_stack)
        assert canonical.can_claim_threefold_repetition() == board.can_claim_threefold_repetition()


class TestValidMoves:
    def test_initial_position_has_20_moves(self, chess_game: ChessGame) -> None:
        board = chess_game.get_init_board()
        valids = chess_game.get_valid_moves(board, 1)
        assert int(valids.sum()) == 20


def test_player_turn_contract_raises_explicit_error(chess_game: ChessGame) -> None:
    board = chess_game.get_init_board()

    with pytest.raises(ValueError, match="does not match the side to move"):
        chess_game.get_valid_moves(board, -1)
    with pytest.raises(ValueError, match="does not match the side to move"):
        chess_game.get_canonical_form(board, -1)
    with pytest.raises(ValueError, match="does not match the side to move"):
        chess_game.get_next_state(board, -1, move_to_action(chess.Move.from_uci("e2e4")))


@pytest.mark.parametrize(
    "uci,expected_piece",
    [
        ("a7a8", chess.QUEEN),
        ("a7a8n", chess.KNIGHT),
        ("a7a8r", chess.ROOK),
        ("a7a8b", chess.BISHOP),
    ],
)
def test_action_encoding_with_promotions(
    uci: str,
    expected_piece: chess.PieceType,
    chess_game: ChessGame,
) -> None:
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


@pytest.mark.parametrize("suffix", ["q", "n", "r", "b"])
def test_black_promotion_roundtrip_uses_canonical_action(suffix: str, chess_game: ChessGame) -> None:
    board = chess.Board("4k2K/8/8/8/8/8/p7/8 b - - 0 1")
    move = chess.Move.from_uci(f"a2a1{suffix}")
    canonical_move = chess.Move.from_uci(f"a7a8{suffix}")
    action = move_to_action(canonical_move)

    assert chess_game.get_valid_moves(board, -1)[action] == 1.0
    next_board, next_player = chess_game.get_next_state(board, -1, action)
    assert next_player == 1
    assert next_board.peek() == move
    assert next_board.piece_at(chess.A1).piece_type == move.promotion


@pytest.mark.parametrize("action", [-1, 4288])
def test_action_decoder_rejects_out_of_range_indices(action: int) -> None:
    with pytest.raises(ValueError, match="Action index"):
        action_to_move(action)
