"""Versioned opening positions shared by comparable arena evaluations."""

from __future__ import annotations

import chess

OPENING_SUITE_VERSION = 1
MAX_OPENING_PAIRS = 10

_OPENING_LINES: tuple[tuple[str, ...], ...] = (
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"),
    ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"),
    ("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"),
    ("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"),
    ("c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6"),
    ("g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "g7g6"),
    ("e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"),
    ("e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4"),
    ("d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"),
    ("d2d4", "f7f5", "g2g3", "g8f6", "f1g2", "g7g6"),
)


def evaluation_openings(pair_count: int) -> tuple[chess.Board, ...]:
    """Build the requested prefix of the versioned opening suite."""
    if isinstance(pair_count, bool) or not isinstance(pair_count, int) or not 0 <= pair_count <= MAX_OPENING_PAIRS:
        raise ValueError(f"pair_count must be between 0 and {MAX_OPENING_PAIRS}")
    return tuple(_board_after(line) for line in _OPENING_LINES[:pair_count])


def _board_after(line: tuple[str, ...]) -> chess.Board:
    board = chess.Board()
    for move_text in line:
        board.push_uci(move_text)
    return board
