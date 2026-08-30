"""External chess-engine adapters."""

import importlib
from typing import Protocol, cast

import chess

from luna.game.chess_game import move_to_action


class _StockfishEngine(Protocol):
    def set_elo_rating(self, elo: int) -> None: ...

    def set_depth(self, depth: int) -> None: ...

    def set_fen_position(self, fen: str) -> None: ...

    def get_best_move(self) -> str | None: ...

    def send_quit_command(self) -> None: ...


class _StockfishFactory(Protocol):
    def __call__(self, *, path: str, parameters: dict[str, int]) -> _StockfishEngine: ...


def move_from_uci(board: chess.Board, uci: str) -> chess.Move:
    try:
        move = chess.Move.from_uci(uci)
    except ValueError as exc:
        raise ValueError(f"External engine returned malformed UCI move: {uci!r}") from exc
    if move not in board.legal_moves:
        raise ValueError(f"External engine returned illegal move {uci!r} for position {board.fen()}")
    return move


class StockfishPlayer:
    """Adapter for a locally installed Stockfish binary."""

    stockfish: _StockfishEngine

    def __init__(
        self,
        elo: int = 1320,
        depth: int = 10,
        path: str | None = None,
    ) -> None:
        stockfish_module = importlib.import_module("stockfish")
        stockfish_factory = cast(_StockfishFactory, stockfish_module.Stockfish)
        bin_path = path if path is not None else "stockfish"
        self.stockfish = stockfish_factory(
            path=bin_path,
            parameters={"Threads": 2},
        )
        self.stockfish.set_elo_rating(elo)
        self.stockfish.set_depth(depth)

    def play(self, board: chess.Board) -> int:
        self.stockfish.set_fen_position(board.fen())
        uci_move = self.stockfish.get_best_move()
        if uci_move is None:
            raise RuntimeError("Stockfish returned no move for a non-terminal position")
        move = move_from_uci(board, uci_move.strip())
        return move_to_action(move)

    def close(self) -> None:
        self.stockfish.send_quit_command()
