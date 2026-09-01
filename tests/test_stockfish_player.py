"""Tests for Stockfish benchmark helpers."""

from unittest.mock import patch

import chess
import chess.engine
import pytest

from luna.config import TrainingRunConfig
from luna.game.player import StockfishPlayer
from luna.game.stockfish_eval import (
    validate_ladder_configuration,
    validate_stockfish_configuration,
)


def test_stockfish_player_configures_engine_and_forwards_new_game() -> None:
    calls: list[tuple[str, object]] = []

    class _Engine:
        def __init__(self) -> None:
            self.id = {"name": "Stockfish Test"}
            self.options = {
                "Threads": chess.engine.Option("Threads", "spin", 1, 1, 128, None),
                "UCI_LimitStrength": chess.engine.Option("UCI_LimitStrength", "check", False, None, None, None),
                "UCI_Elo": chess.engine.Option("UCI_Elo", "spin", 1500, 1320, 3190, None),
            }

        def configure(self, options: dict[str, object]) -> None:
            calls.append(("configure", options))

        def play(
            self,
            _board: chess.Board,
            limit: chess.engine.Limit,
            *,
            game: object,
        ) -> chess.engine.PlayResult:
            calls.append(("play", (limit.depth, game)))
            return chess.engine.PlayResult(chess.Move.from_uci("e2e4"), None)

        def quit(self) -> None:
            calls.append(("quit", 0))

        def close(self) -> None:
            calls.append(("close", 0))

    engine = _Engine()
    with patch("luna.game.player.chess.engine.SimpleEngine.popen_uci", return_value=engine) as popen:
        player = StockfishPlayer(path="/opt/stockfish")
        player.play(chess.Board())
        player.new_game()
        player.play(chess.Board())
        player.close()

    popen.assert_called_once_with("/opt/stockfish")
    assert calls[0] == (
        "configure",
        {"Threads": 2, "UCI_LimitStrength": True, "UCI_Elo": 1500},
    )
    first_game = calls[1][1]
    second_game = calls[2][1]
    assert isinstance(first_game, tuple) and isinstance(second_game, tuple)
    assert first_game[0] == second_game[0] == 10
    assert first_game[1] is not second_game[1]
    assert calls[-2:] == [("quit", 0), ("close", 0)]


def test_stockfish_player_hard_closes_transport_after_quit_failure() -> None:
    calls: list[str] = []

    class _Engine:
        def __init__(self) -> None:
            self.id = {"name": "Stockfish Test"}
            self.options = {
                "Threads": chess.engine.Option("Threads", "spin", 1, 1, 128, None),
                "UCI_LimitStrength": chess.engine.Option("UCI_LimitStrength", "check", False, None, None, None),
                "UCI_Elo": chess.engine.Option("UCI_Elo", "spin", 1500, 1320, 3190, None),
            }

        def configure(self, _options: dict[str, object]) -> None:
            pass

        def quit(self) -> None:
            calls.append("quit")
            raise TimeoutError("quit timed out")

        def close(self) -> None:
            calls.append("close")

    with patch("luna.game.player.chess.engine.SimpleEngine.popen_uci", return_value=_Engine()):
        player = StockfishPlayer()

    with pytest.raises(TimeoutError, match="quit timed out"):
        player.close()

    assert calls == ["quit", "close"]


def test_stockfish_player_preserves_quit_failure_when_hard_close_also_fails() -> None:
    calls: list[str] = []

    class _Engine:
        def __init__(self) -> None:
            self.id = {"name": "Stockfish Test"}
            self.options = {
                "Threads": chess.engine.Option("Threads", "spin", 1, 1, 128, None),
                "UCI_LimitStrength": chess.engine.Option("UCI_LimitStrength", "check", False, None, None, None),
                "UCI_Elo": chess.engine.Option("UCI_Elo", "spin", 1500, 1320, 3190, None),
            }

        def configure(self, _options: dict[str, object]) -> None:
            pass

        def quit(self) -> None:
            calls.append("quit")
            raise TimeoutError("quit timed out")

        def close(self) -> None:
            calls.append("close")
            raise OSError("transport close failed")

    with patch("luna.game.player.chess.engine.SimpleEngine.popen_uci", return_value=_Engine()):
        player = StockfishPlayer()

    with pytest.raises(TimeoutError, match="quit timed out") as error:
        player.close()

    assert calls == ["quit", "close"]
    assert error.value.__notes__ == ["Hard-closing the engine transport also failed: transport close failed"]


def test_stockfish_player_initialization_error_is_not_masked_by_cleanup() -> None:
    calls: list[str] = []

    class _Engine:
        def __init__(self) -> None:
            self.id = {"name": "Stockfish Test"}
            self.options = {
                "Threads": chess.engine.Option("Threads", "spin", 1, 1, 128, None),
                "UCI_LimitStrength": chess.engine.Option("UCI_LimitStrength", "check", False, None, None, None),
                "UCI_Elo": chess.engine.Option("UCI_Elo", "spin", 1500, 1320, 3190, None),
            }

        def configure(self, _options: dict[str, object]) -> None:
            raise ValueError("configure failed")

        def quit(self) -> None:
            calls.append("quit")
            raise TimeoutError("quit timed out")

        def close(self) -> None:
            calls.append("close")

    with (
        patch("luna.game.player.chess.engine.SimpleEngine.popen_uci", return_value=_Engine()),
        pytest.raises(ValueError, match="configure failed") as error,
    ):
        StockfishPlayer()

    assert calls == ["quit", "close"]
    assert error.value.__notes__ == ["External-engine cleanup also failed: quit timed out"]


def test_stockfish_preflight_closes_verified_process() -> None:
    closed = False

    class _Player:
        engine_name = "Stockfish Test"
        elo_range = (1320, 3190)

        def play(self, _board: chess.Board) -> int:
            return 0

        def close(self) -> None:
            nonlocal closed
            closed = True

    with patch("luna.game.stockfish_eval._stockfish_player", return_value=_Player()):
        validate_stockfish_configuration(TrainingRunConfig())

    assert closed


@pytest.mark.parametrize("engine_name", ["Fairy-Stockfish 14", "Leela Chess Zero"])
def test_fixed_stockfish_preflight_rejects_non_official_engine_without_masking_identity_error(
    engine_name: str,
) -> None:
    class _Player:
        elo_range = (500, 2850)

        @property
        def engine_name(self) -> str:
            return engine_name

        def play(self, _board: chess.Board) -> int:
            return 0

        def close(self) -> None:
            raise RuntimeError("cleanup failed")

    with (
        patch("luna.game.stockfish_eval._stockfish_player", return_value=_Player()),
        pytest.raises(RuntimeError, match="requires official Stockfish") as error,
    ):
        validate_stockfish_configuration(TrainingRunConfig())

    assert isinstance(error.value.__cause__, ValueError)
    assert error.value.__cause__.__notes__ == ["External-engine preflight cleanup also failed: cleanup failed"]


def test_stockfish_preflight_cleanup_does_not_mask_play_failure() -> None:
    class _Player:
        engine_name = "Stockfish 17.1"
        elo_range = (1320, 3190)

        def play(self, _board: chess.Board) -> int:
            raise RuntimeError("preflight play failed")

        def close(self) -> None:
            raise RuntimeError("cleanup failed")

    with (
        patch("luna.game.stockfish_eval._stockfish_player", return_value=_Player()),
        pytest.raises(RuntimeError, match="preflight play failed") as error,
    ):
        validate_stockfish_configuration(TrainingRunConfig())

    assert error.value.__cause__ is not None
    assert error.value.__cause__.__notes__ == ["External-engine preflight cleanup also failed: cleanup failed"]


def test_fairy_ladder_preflight_accepts_fairy_and_rejects_official_stockfish() -> None:
    class _Player:
        engine_name = "Fairy-Stockfish 14"
        elo_range = (500, 2850)

        def play(self, _board: chess.Board) -> int:
            return 0

        def close(self) -> None:
            pass

    player = _Player()
    with patch("luna.game.stockfish_eval._stockfish_player", return_value=player):
        validate_ladder_configuration(TrainingRunConfig())

    player.engine_name = "Stockfish 17.1"
    with (
        patch("luna.game.stockfish_eval._stockfish_player", return_value=player),
        pytest.raises(RuntimeError, match="requires Fairy-Stockfish"),
    ):
        validate_ladder_configuration(TrainingRunConfig())
