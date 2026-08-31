"""Tests for Stockfish benchmark helpers."""

from types import SimpleNamespace
from unittest.mock import patch

import chess

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
    run_stockfish_eval,
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

    assert metrics == {
        "stockfish/luna_wins": 2,
        "stockfish/draws": 2,
        "stockfish/stockfish_wins": 4,
        "stockfish/win_rate": 0.25,
        "stockfish/score": 0.375,
        "stockfish/opening_suite_version": 1,
        "iteration": 25,
    }


def test_stockfish_player_configures_engine_and_forwards_new_game() -> None:
    calls: list[tuple[str, int]] = []

    class _Engine:
        def set_elo_rating(self, elo: int) -> None:
            calls.append(("elo", elo))

        def set_depth(self, depth: int) -> None:
            calls.append(("depth", depth))

        def send_ucinewgame_command(self) -> None:
            calls.append(("new_game", 0))

        def send_quit_command(self) -> None:
            calls.append(("close", 0))

    engine = _Engine()

    def create_engine(*, path: str, parameters: dict[str, int]) -> _Engine:
        assert path == "/opt/stockfish"
        assert parameters == {"Threads": 2}
        return engine

    module = SimpleNamespace(Stockfish=create_engine)
    with patch("luna.game.player.importlib.import_module", return_value=module):
        player = StockfishPlayer(path="/opt/stockfish")
        player.new_game()
        player.close()

    assert calls == [("elo", 1320), ("depth", 10), ("new_game", 0), ("close", 0)]


def test_stockfish_preflight_closes_verified_process() -> None:
    closed = False

    class _Player:
        def close(self) -> None:
            nonlocal closed
            closed = True

    with patch("luna.game.stockfish_eval._stockfish_player", return_value=_Player()):
        validate_stockfish_configuration(TrainingRunConfig())

    assert closed
