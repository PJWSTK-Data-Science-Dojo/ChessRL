"""Exact-tree history retention and draw adjudication tests."""

import chess
import numpy as np

from luna.game.chess_game import HISTORY_LENGTH, ChessGame, board_to_numpy, mirror_move, move_to_action

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


def test_exact_search_root_bounds_history_without_changing_observation_or_draw(chess_game: ChessGame) -> None:
    board = chess.Board()
    for uci in _LONG_HISTORY_MOVES:
        board.push_uci(uci)
    for uci in ("d7b8", "d2b1", "b8d7", "b1d2") * 2:
        board.push_uci(uci)
    assert board.can_claim_threefold_repetition()

    search_root = chess_game.copy_exact_search_root(board)

    assert len(search_root.move_stack) == board.halfmove_clock
    assert len(search_root.move_stack) < len(board.move_stack)
    np.testing.assert_array_equal(board_to_numpy(search_root), board_to_numpy(board))
    assert search_root.can_claim_threefold_repetition()
    assert chess_game.get_game_outcome(search_root, -1) == chess_game.get_game_outcome(board, -1) == 0.0


def test_exact_search_root_keeps_temporal_history_across_zeroing_move(chess_game: ChessGame) -> None:
    board = chess.Board()
    for uci in _LONG_HISTORY_MOVES:
        board.push_uci(uci)
    board.push_uci("c7c5")
    assert board.halfmove_clock == 0

    search_root = chess_game.copy_exact_search_root(board)

    assert len(search_root.move_stack) == HISTORY_LENGTH - 1
    np.testing.assert_array_equal(board_to_numpy(search_root), board_to_numpy(board))


def test_exact_search_transition_preserves_newly_claimable_draw(chess_game: ChessGame) -> None:
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
        board.push_uci(uci)
    action = move_to_action(mirror_move(chess.Move.from_uci("f6g8")))

    next_board, next_player = chess_game.get_next_exact_search_state(board, -1, action)

    assert next_board.can_claim_threefold_repetition()
    assert chess_game.get_game_outcome(next_board, next_player) == 0.0
