"""Tests for Stockfish benchmark helpers."""

from unittest.mock import patch

from luna.config import TrainingRunConfig, evaluation_mcts_params
from luna.game.stockfish_eval import (
    StockfishEvalScores,
    StockfishEvalSkipped,
    _score_game_for_model,
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


class TestRunStockfishEval:
    def test_returns_skipped_when_too_few_games(self, chess_game, small_learner_config) -> None:
        nnet = LunaNetwork(chess_game, small_learner_config)
        run = TrainingRunConfig(stockfish_eval_games=1, num_mcts_sims=1, dir_noise=False)
        out = run_stockfish_eval(chess_game, nnet, run, iteration=1)
        assert isinstance(out, StockfishEvalSkipped)
        assert out.reason == "too_few_games"

    def test_returns_skipped_when_engine_init_fails(self, chess_game, small_learner_config) -> None:
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


def test_evaluation_mcts_params_matches_run() -> None:
    run = TrainingRunConfig(num_mcts_sims=40, evaluation_num_mcts_sims=7, dir_noise=True, recurrent_policy_topk=128)
    p = evaluation_mcts_params(run)
    assert p.num_mcts_sims == 7
    assert p.dir_noise is False
    assert p.recurrent_policy_topk == 128


def test_scores_dataclass_fields() -> None:
    s = StockfishEvalScores(model_wins=2, draws=1, stockfish_wins=7)
    assert s.model_wins == 2 and s.draws == 1 and s.stockfish_wins == 7
