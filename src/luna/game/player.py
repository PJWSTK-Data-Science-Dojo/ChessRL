"""External chess-engine adapters."""

import chess
import chess.engine

from luna.game.chess_game import move_to_action


def move_from_uci(board: chess.Board, uci: str) -> chess.Move:
    try:
        move = chess.Move.from_uci(uci)
    except ValueError as exc:
        raise ValueError(f"External engine returned malformed UCI move: {uci!r}") from exc
    if move not in board.legal_moves:
        raise ValueError(f"External engine returned illegal move {uci!r} for position {board.fen()}")
    return move


class StockfishPlayer:
    """Adapter for an official or Fairy-Stockfish UCI binary."""

    stockfish: chess.engine.SimpleEngine

    def __init__(
        self,
        elo: int = 1500,
        depth: int = 10,
        path: str | None = None,
        threads: int = 2,
    ) -> None:
        bin_path = path if path is not None else "stockfish"
        self.stockfish = chess.engine.SimpleEngine.popen_uci(bin_path)
        self._depth = depth
        self._game_token = object()
        try:
            self._validate_spin_option("Threads", threads)
            self._validate_spin_option("UCI_Elo", elo)
            if "UCI_LimitStrength" not in self.stockfish.options:
                raise ValueError(f"External engine {self.engine_name!r} does not expose UCI_LimitStrength")
            self.stockfish.configure(
                {
                    "Threads": threads,
                    "UCI_LimitStrength": True,
                    "UCI_Elo": elo,
                }
            )
        except (chess.engine.EngineError, OSError, TimeoutError, ValueError) as exc:
            try:
                self.close()
            except Exception as cleanup_exc:
                exc.add_note(f"External-engine cleanup also failed: {cleanup_exc}")
            raise

    @property
    def engine_name(self) -> str:
        """Return the UCI engine name reported during initialization."""
        return self.stockfish.id.get("name", "unknown UCI engine")

    @property
    def elo_range(self) -> tuple[int, int]:
        """Return the engine-advertised inclusive UCI Elo range."""
        option = self.stockfish.options.get("UCI_Elo")
        if option is None or option.type != "spin" or option.min is None or option.max is None:
            raise ValueError(f"External engine {self.engine_name!r} does not expose bounded UCI_Elo")
        return option.min, option.max

    def _validate_spin_option(self, name: str, value: int) -> None:
        option = self.stockfish.options.get(name)
        if option is None or option.type != "spin" or option.min is None or option.max is None:
            raise ValueError(f"External engine {self.engine_name!r} does not expose bounded {name}")
        if not option.min <= value <= option.max:
            raise ValueError(
                f"External engine {self.engine_name!r} supports {name} from {option.min} through {option.max}, got {value}"
            )

    def play(self, board: chess.Board) -> int:
        result = self.stockfish.play(
            board,
            chess.engine.Limit(depth=self._depth),
            game=self._game_token,
        )
        if result.move is None:
            raise RuntimeError("Stockfish returned no move for a non-terminal position")
        move = move_from_uci(board, result.move.uci())
        return move_to_action(move)

    def new_game(self) -> None:
        self._game_token = object()

    def close(self) -> None:
        quit_failure: Exception | None = None
        try:
            self.stockfish.quit()
        except Exception as exc:
            quit_failure = exc

        try:
            self.stockfish.close()
        except Exception as exc:
            if quit_failure is None:
                raise
            quit_failure.add_note(f"Hard-closing the engine transport also failed: {exc}")

        if quit_failure is not None:
            raise quit_failure
