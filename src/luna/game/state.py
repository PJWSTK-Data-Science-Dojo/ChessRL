"""Luna-chess board state."""

from __future__ import annotations

import chess


class LunaState:
    """Lightweight board state wrapper for the engine/web interface."""

    board: chess.Board

    def __init__(self, board: chess.Board | None = None) -> None:
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def key(self) -> tuple:
        return (
            self.board.board_fen(),
            self.board.turn,
            self.board.castling_rights,
            self.board.ep_square,
            self.board.halfmove_clock,
        )

    def edges(self) -> list[chess.Move]:
        return list(self.board.legal_moves)
