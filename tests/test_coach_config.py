"""Tests for Coach self-play (e.g. max ply truncation, batched self-play)."""

import pytest

from luna.coach import (
    Coach,
)
from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.game.chess_game import ChessGame
from luna.network import LunaNetwork


def test_coach_rejects_run_and_learner_discount_mismatch(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.discount = 0.9
    network = LunaNetwork(chess_game, small_learner_config)

    with pytest.raises(ValueError, match=r"run\.discount and learner\.discount must match"):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(discount=1.0, stockfish_eval_every=0),
        )


def test_coach_rejects_invalid_training_schedule(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with pytest.raises(ValueError, match="parallel_games must be a positive integer"):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(parallel_games=0, stockfish_eval_every=0),
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"target_replay_ratio": 0.0}, "target_replay_ratio must be finite"),
        ({"target_replay_ratio": float("inf")}, "target_replay_ratio must be finite"),
        ({"lr_schedule_total_steps": 0}, "lr_schedule_total_steps must be a positive integer"),
        ({"replay_warmup_positions": -1}, "replay_warmup_positions must be a non-negative integer"),
        (
            {"replay_warmup_positions": 101, "replay_capacity": 100},
            "replay_warmup_positions cannot exceed replay_capacity",
        ),
    ],
)
def test_coach_rejects_invalid_dynamic_training_schedule(
    override: dict[str, object],
    message: str,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with pytest.raises(ValueError, match=message):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(stockfish_eval_every=0, **override),
        )


def test_coach_rejects_evaluation_larger_than_opening_suite(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with pytest.raises(ValueError, match="stockfish_eval_games cannot exceed 20"):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(stockfish_eval_games=22),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ladder_start_elo": 499}, "cannot be below Fairy-Stockfish's 500 floor"),
        ({"ladder_max_elo": 2900}, "cannot exceed Fairy-Stockfish's 2850 ceiling"),
        ({"ladder_eval_games": 3}, "ladder_eval_games must be an even integer"),
        ({"ladder_max_elo": 2750}, "reachable.*exact ladder_step_elo increments"),
        ({"checkpoint": ""}, "checkpoint cannot be blank"),
    ],
)
def test_coach_rejects_invalid_fairy_ladder_contract(
    overrides: dict[str, object],
    message: str,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    config = {"ladder_eval_every": 5, **overrides}

    with pytest.raises(ValueError, match=message):
        Coach(chess_game, network, TrainingRunConfig(**config))


def test_fixed_benchmark_defaults_to_1500_and_ladder_is_opt_in() -> None:
    run = TrainingRunConfig()

    assert run.stockfish_elo == 1500
    assert run.ladder_eval_every == 0
    assert run.ladder_start_elo == 500
