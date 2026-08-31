"""Tests for Stockfish benchmark helpers."""

from unittest.mock import patch

import chess
import chess.engine
import pytest

from luna.config import EzV2LearnerConfig, TrainingRunConfig, evaluation_mcts_params
from luna.game.arena import Arena
from luna.game.chess_game import ChessGame
from luna.game.player import StockfishPlayer
from luna.game.stockfish_eval import (
    StockfishEvalScores,
    StockfishEvalSkipped,
    _score_game_for_model,
    _StockfishException,
    _wandb_metrics,
    retry_stockfish_eval,
    run_stockfish_eval,
    validate_ladder_configuration,
    validate_stockfish_configuration,
)
from luna.network import LunaNetwork


class TestScoreGameForModel:
    def test_model_as_p1(self) -> None:
        assert _score_game_for_model(1.0, model_is_player1=True) == "model"
        assert _score_game_for_model(-1.0, model_is_player1=True) == "sf"
        assert _score_game_for_model(0.0, model_is_player1=True) == "draw"

    def test_model_as_p2(self) -> None:
        assert _score_game_for_model(-1.0, model_is_player1=False) == "model"
        assert _score_game_for_model(1.0, model_is_player1=False) == "sf"
        assert _score_game_for_model(0.0, model_is_player1=False) == "draw"


def test_transient_stockfish_failure_is_retried_until_success() -> None:
    skipped = StockfishEvalSkipped("runtime_error", "engine exited")
    success = StockfishEvalScores(model_wins=1, draws=0, stockfish_wins=1)
    outcomes = iter((skipped, skipped, success))

    with patch("luna.game.stockfish_eval.time.sleep") as sleep:
        outcome = retry_stockfish_eval(lambda: next(outcomes), attempts=3, retry_seconds=2.0)

    assert outcome == success
    assert sleep.call_count == 2


def test_non_retryable_stockfish_configuration_failure_returns_immediately() -> None:
    skipped = StockfishEvalSkipped("too_many_games", "opening suite exhausted")
    calls = 0

    def evaluate() -> StockfishEvalSkipped:
        nonlocal calls
        calls += 1
        return skipped

    assert retry_stockfish_eval(evaluate, attempts=3, retry_seconds=0.0) == skipped
    assert calls == 1


class TestRunStockfishEval:
    def test_returns_skipped_when_too_few_games(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        nnet = LunaNetwork(chess_game, small_learner_config)
        run = TrainingRunConfig(stockfish_eval_games=1, num_mcts_sims=1, dir_noise=False)
        out = run_stockfish_eval(chess_game, nnet, run, iteration=1)
        assert isinstance(out, StockfishEvalSkipped)
        assert out.reason == "too_few_games"

    def test_returns_skipped_when_engine_init_fails(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        nnet = LunaNetwork(chess_game, small_learner_config)
        run = TrainingRunConfig(
            stockfish_eval_games=2,
            num_mcts_sims=1,
            dir_noise=False,
            evaluation_num_mcts_sims=1,
            recurrent_policy_topk=64,
            max_ply=3,
            stockfish_eval_max_ply=3,
        )
        with patch("luna.game.stockfish_eval.StockfishPlayer", side_effect=RuntimeError("no binary")):
            out = run_stockfish_eval(chess_game, nnet, run, iteration=1)
        assert isinstance(out, StockfishEvalSkipped)
        assert out.reason == "no_engine"

    def test_uses_distinct_paired_openings_and_resets_stockfish(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        calls: list[tuple[str, tuple[chess.Move, ...], bool, bool]] = []

        def model_player(_board: chess.Board) -> int:
            raise AssertionError("Patched Arena must not request a move")

        class _Stockfish:
            def __init__(self) -> None:
                self.new_game_calls = 0
                self.close_calls = 0

            def play(self, _board: chess.Board) -> int:
                raise AssertionError("Patched Arena must not request a move")

            def new_game(self) -> None:
                self.new_game_calls += 1

            def close(self) -> None:
                self.close_calls += 1

        stockfish = _Stockfish()

        def record_game(
            arena: Arena,
            verbose: bool = False,
            max_ply: int | None = None,
            initial_board: chess.Board | None = None,
        ) -> float:
            del verbose, max_ply
            assert initial_board is not None
            calls.append(
                (
                    initial_board.fen(),
                    tuple(initial_board.move_stack),
                    arena.player1 is model_player,
                    arena.player2 is model_player,
                )
            )
            return 0.0

        network = LunaNetwork(chess_game, small_learner_config)
        run = TrainingRunConfig(
            stockfish_eval_games=4,
            num_mcts_sims=1,
            evaluation_num_mcts_sims=1,
        )
        with (
            patch("luna.game.stockfish_eval._stockfish_player", return_value=stockfish),
            patch("luna.game.stockfish_eval.ArenaMCTSPlayer", return_value=model_player),
            patch.object(Arena, "play_game", new=record_game),
        ):
            outcome = run_stockfish_eval(chess_game, network, run)

        assert outcome == StockfishEvalScores(model_wins=0, draws=4, stockfish_wins=0)
        assert stockfish.new_game_calls == 4
        assert stockfish.close_calls == 1
        assert len(calls) == 4
        assert calls[0][0] == calls[1][0]
        assert calls[2][0] == calls[3][0]
        assert calls[0][0] != calls[2][0]
        assert [call[2:] for call in calls] == [
            (True, False),
            (False, True),
            (True, False),
            (False, True),
        ]
        assert all(len(call[1]) == 6 for call in calls)

    def test_returns_skipped_when_stockfish_process_exits(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        nnet = LunaNetwork(chess_game, small_learner_config)
        run = TrainingRunConfig(stockfish_eval_games=2, evaluation_num_mcts_sims=1)

        with patch(
            "luna.game.stockfish_eval.StockfishPlayer",
            side_effect=_StockfishException("engine process exited"),
        ):
            out = run_stockfish_eval(chess_game, nnet, run)

        assert isinstance(out, StockfishEvalSkipped)
        assert out.reason == "no_engine"

    def test_close_failure_is_reported_as_runtime_failure(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        class _Stockfish:
            def new_game(self) -> None:
                pass

            def play(self, _board: chess.Board) -> int:
                raise AssertionError("Patched Arena must not request a move")

            def close(self) -> None:
                raise RuntimeError("close failed")

        network = LunaNetwork(chess_game, small_learner_config)
        run = TrainingRunConfig(stockfish_eval_games=2, evaluation_num_mcts_sims=1)
        with (
            patch("luna.game.stockfish_eval._stockfish_player", return_value=_Stockfish()),
            patch("luna.game.stockfish_eval.ArenaMCTSPlayer", return_value=lambda _board: 0),
            patch.object(Arena, "play_game", return_value=0.0),
        ):
            outcome = run_stockfish_eval(chess_game, network, run)

        assert outcome == StockfishEvalSkipped("runtime_error", "failed to close Stockfish: close failed")

    def test_game_failure_is_not_overwritten_by_close_failure(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        class _Stockfish:
            def new_game(self) -> None:
                pass

            def play(self, _board: chess.Board) -> int:
                raise AssertionError("Patched Arena must not request a move")

            def close(self) -> None:
                raise RuntimeError("close failed")

        network = LunaNetwork(chess_game, small_learner_config)
        run = TrainingRunConfig(stockfish_eval_games=2, evaluation_num_mcts_sims=1)
        with (
            patch("luna.game.stockfish_eval._stockfish_player", return_value=_Stockfish()),
            patch("luna.game.stockfish_eval.ArenaMCTSPlayer", return_value=lambda _board: 0),
            patch.object(Arena, "play_game", side_effect=RuntimeError("game failed")),
        ):
            outcome = run_stockfish_eval(chess_game, network, run)

        assert outcome == StockfishEvalSkipped("runtime_error", "game failed")


def test_evaluation_mcts_params_matches_run() -> None:
    run = TrainingRunConfig(num_mcts_sims=40, evaluation_num_mcts_sims=7, dir_noise=True, recurrent_policy_topk=128)
    p = evaluation_mcts_params(run)
    assert p.num_mcts_sims == 7
    assert p.dir_noise is False
    assert p.recurrent_policy_topk == 128


def test_scores_dataclass_fields() -> None:
    s = StockfishEvalScores(model_wins=2, draws=1, stockfish_wins=7)
    assert s.model_wins == 2 and s.draws == 1 and s.stockfish_wins == 7


def test_wandb_metrics_use_named_players_and_draw_aware_score() -> None:
    scores = StockfishEvalScores(model_wins=2, draws=2, stockfish_wins=4)

    metrics = _wandb_metrics(scores, iteration=25)

    assert metrics["benchmark/luna_wins"] == 2
    assert metrics["benchmark/draws"] == 2
    assert metrics["benchmark/stockfish_wins"] == 4
    assert metrics["benchmark/games"] == 8
    assert metrics["benchmark/win_rate"] == 0.25
    assert metrics["benchmark/decisive_win_rate"] == 1 / 3
    assert metrics["benchmark/score"] == 0.375
    assert metrics["benchmark/score_approx_ci95_low"] < 0.375
    assert metrics["benchmark/score_approx_ci95_high"] > 0.375
    assert metrics["benchmark/opening_suite_version"] == 1
    assert metrics["iteration"] == 25


def test_wandb_metrics_use_stable_stockfish_outcome_key_for_ladder() -> None:
    scores = StockfishEvalScores(model_wins=12, draws=4, stockfish_wins=4)

    metrics = _wandb_metrics(scores, iteration=5, prefix="ladder")

    assert metrics["ladder/stockfish_wins"] == 4
    assert "ladder/fairy_wins" not in metrics


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
