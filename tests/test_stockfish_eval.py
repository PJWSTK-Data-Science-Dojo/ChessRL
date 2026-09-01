"""Tests for Stockfish benchmark helpers."""

from unittest.mock import patch

import chess
import chess.engine

from luna.config import EzV2LearnerConfig, TrainingRunConfig, evaluation_mcts_params
from luna.game.arena import Arena
from luna.game.chess_game import ChessGame
from luna.game.stockfish_eval import (
    StockfishEvalScores,
    StockfishEvalSkipped,
    _score_game_for_model,
    _StockfishException,
    _wandb_metrics,
    retry_stockfish_eval,
    run_stockfish_eval,
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
