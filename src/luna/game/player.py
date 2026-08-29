"""External chess-engine adapters."""

import chess
from loguru import logger

from .chess_game import move_to_action


def move_from_uci(board: chess.Board, uci: str) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        logger.warning("expected an UCI move")
        return None
    if move not in board.legal_moves:
        logger.warning("expected a valid move")
        return None
    return move


class StockfishPlayer:
    """Stockfish wrapper (requires `stockfish` package and binary)."""

    def __init__(
        self,
        elo: int = 1000,
        skill_level: int = 10,
        depth: int = 10,
        think_time: int = 30,
        path: str | None = None,
    ) -> None:
        from stockfish import Stockfish

        bin_path = path if path is not None else "stockfish"
        self.stockfish = Stockfish(path=bin_path, parameters={"Threads": 2, "Minimum Thinking Time": think_time})
        self.stockfish.set_elo_rating(elo)
        self.stockfish.set_skill_level(skill_level)
        self.stockfish.set_depth(depth)

    def play(self, board: chess.Board) -> int:
        self.stockfish.set_fen_position(board.fen())
        uci_move = self.stockfish.get_best_move()
        if uci_move is None:
            raise RuntimeError("Stockfish returned no move for a non-terminal position")
        move = move_from_uci(board, uci_move.strip())
        if move is None:
            raise ValueError(f"Stockfish suggested illegal move {uci_move} in position {board.fen()}")
        return move_to_action(move)
