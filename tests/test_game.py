"""Tests for chess game wrapper -- reward perspective, legal moves, action encoding."""

import random

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
    mirror_move,
    move_to_action,
    player_from_turn,
)

_LONG_HISTORY_MOVES = (
    "e2e4",
    "e7e5",
    "g1f3",
    "b8c6",
    "f1b5",
    "a7a6",
    "b5a4",
    "g8f6",
    "e1g1",
    "f8e7",
    "f1e1",
    "b7b5",
    "a4b3",
    "d7d6",
    "c2c3",
    "e8g8",
    "h2h3",
    "c6b8",
    "d2d4",
    "b8d7",
    "b1d2",
)


def _piece_plane(history_index: int, color: chess.Color, piece_type: chess.PieceType) -> int:
    color_offset = 0 if color == chess.WHITE else 6
    return history_index * PLANES_PER_POSITION + color_offset + piece_type - 1


class TestObservationEncoding:
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

    def test_repetition_before_recent_zeroing_move_remains_in_history(self) -> None:
        board = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
            board.push_uci(uci)
        board.push_uci("e2e4")

        observation = board_to_numpy(board)
        previous_thrice = PLANES_PER_POSITION + PIECE_PLANES_PER_POSITION + 1

        assert np.all(observation[:, :, previous_thrice] == 1.0)

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

    def test_earliest_repetition_claim_is_not_skipped(self, chess_game: ChessGame) -> None:
        board = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
            board.push_uci(uci)

        assert board.halfmove_clock == 7
        assert board.can_claim_threefold_repetition()
        assert chess_game.get_game_outcome(board, -1) == 0.0

    def test_impossible_early_repetition_claim_avoids_history_replay(
        self,
        chess_game: ChessGame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = 0
        original = chess.Board.can_claim_threefold_repetition

        def count_calls(board: chess.Board) -> bool:
            nonlocal calls
            calls += 1
            return original(board)

        monkeypatch.setattr(chess.Board, "can_claim_threefold_repetition", count_calls)

        assert chess_game.get_game_outcome(chess.Board(), 1) is None
        assert calls == 0

    def test_fifty_move_claim_from_fen_without_history_is_preserved(self, chess_game: ChessGame) -> None:
        board = chess.Board("8/8/8/8/8/8/R6k/K7 w - - 99 50")

        assert board.can_claim_fifty_moves()
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

    def test_black_canonical_form_discards_only_irrelevant_old_history(self, chess_game: ChessGame) -> None:
        board = chess.Board()
        for uci in _LONG_HISTORY_MOVES:
            board.push_uci(uci)

        canonical = chess_game.get_canonical_form(board, -1)
        full_history_mirror = board.root().mirror()
        for move in board.move_stack:
            full_history_mirror.push(mirror_move(move))
        full_history_mirror.fullmove_number = board.fullmove_number

        assert len(canonical.move_stack) < len(board.move_stack)
        assert canonical.fen() == full_history_mirror.fen()
        assert canonical.can_claim_threefold_repetition() == full_history_mirror.can_claim_threefold_repetition()
        np.testing.assert_array_equal(board_to_numpy(canonical), board_to_numpy(full_history_mirror))


class TestSearchState:
    def test_truncated_search_state_matches_full_state_across_random_play(self, chess_game: ChessGame) -> None:
        random_source = random.Random(20260831)
        board = chess.Board()

        for _ in range(96):
            if board.is_game_over(claim_draw=True):
                board = chess.Board()
            move = random_source.choice(list(board.legal_moves))
            player = player_from_turn(board.turn)
            action = move_to_action(move if player == 1 else mirror_move(move))

            full_board, full_player = chess_game.get_next_state(board, player, action)
            search_board, search_player = chess_game.get_next_latent_search_state(board, player, action)

            assert search_player == full_player
            assert search_board.fen() == full_board.fen()
            assert set(search_board.legal_moves) == set(full_board.legal_moves)
            assert search_board.can_claim_threefold_repetition() == full_board.can_claim_threefold_repetition()
            assert search_board.is_fivefold_repetition() == full_board.is_fivefold_repetition()
            assert search_board.outcome(claim_draw=True) == full_board.outcome(claim_draw=True)
            board.push(move)

    def test_truncated_search_state_preserves_rules(self, chess_game: ChessGame) -> None:
        board = chess.Board()
        for uci in _LONG_HISTORY_MOVES:
            board.push_uci(uci)
        original_stack = tuple(board.move_stack)
        player = player_from_turn(board.turn)
        move = chess.Move.from_uci("d7b8")
        assert move in board.legal_moves
        action = move_to_action(move if player == 1 else mirror_move(move))

        full_board, full_player = chess_game.get_next_state(board, player, action)
        search_board, search_player = chess_game.get_next_latent_search_state(board, player, action)

        assert tuple(board.move_stack) == original_stack
        assert len(search_board.move_stack) == board.halfmove_clock + 1
        assert len(search_board.move_stack) < len(full_board.move_stack)
        assert search_player == full_player
        assert search_board.fen() == full_board.fen()
        assert set(search_board.legal_moves) == set(full_board.legal_moves)
        assert chess_game.get_game_outcome(search_board, search_player) == chess_game.get_game_outcome(
            full_board, full_player
        )

    def test_truncated_search_state_preserves_claimable_repetition(self, chess_game: ChessGame) -> None:
        board = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
            board.push_uci(uci)
        move = chess.Move.from_uci("f6g8")
        action = move_to_action(mirror_move(move))

        next_board, next_player = chess_game.get_next_latent_search_state(board, -1, action)

        assert next_board.can_claim_threefold_repetition()
        assert chess_game.get_game_outcome(next_board, next_player) == 0.0

    def test_zeroing_search_move_clears_obsolete_history(self, chess_game: ChessGame) -> None:
        board = chess.Board()
        action = move_to_action(chess.Move.from_uci("e2e4"))

        next_board, next_player = chess_game.get_next_latent_search_state(board, 1, action)

        assert next_player == -1
        assert next_board.fen() == chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").fen()
        assert not next_board.move_stack


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
