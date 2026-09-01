"""Tests for Coach self-play (e.g. max ply truncation, batched self-play)."""

from unittest.mock import call, patch

import pytest

from luna.coach import (
    Coach,
    _optimizer_steps_for_positions,
)
from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.game.chess_game import ChessGame
from luna.network import LunaNetwork
from luna.profiling import IterProfileStats
from luna.replay_buffer import PrioritizedReplayBuffer
from tests.conftest import TrajectoryFactory


def test_selfplay_metrics_report_replay_samples_per_new_position(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    make_trajectory: TrajectoryFactory,
) -> None:
    small_learner_config.batch_size = 8
    coach = Coach(chess_game, LunaNetwork(chess_game, small_learner_config), TrainingRunConfig())

    with patch("luna.coach_metrics.wandb.run", object()), patch("luna.coach_metrics.wandb.log") as wandb_log:
        coach._log_iteration_metrics(
            1,
            [make_trajectory(4), make_trajectory(4)],
            IterProfileStats(iter_index=1),
            optimizer_steps=3,
        )

    metrics = wandb_log.call_args.args[0]
    assert metrics["selfplay/replay_samples_per_new_position"] == 3.0
    assert metrics["replay/optimizer_steps"] == 3
    assert metrics["replay/step_cap_reached"] == 0
    assert metrics["replay/target_samples_per_new_position"] == 0.0
    assert metrics["replay/warmup_positions"] == 8


@pytest.mark.parametrize(
    ("run", "positions", "batch_size", "expected_steps"),
    [
        (TrainingRunConfig(train_steps_per_iter=11), 32, 8, 11),
        (TrainingRunConfig(train_steps_per_iter=11), 0, 8, 0),
        (TrainingRunConfig(train_steps_per_iter=11, target_replay_ratio=2.0), 10, 8, 3),
        (TrainingRunConfig(train_steps_per_iter=11, target_replay_ratio=0.1), 1, 8, 1),
        (TrainingRunConfig(train_steps_per_iter=11, target_replay_ratio=3.0), 100, 8, 11),
    ],
)
def test_optimizer_steps_follow_target_ratio_with_cap_and_legacy_fallback(
    run: TrainingRunConfig,
    positions: int,
    batch_size: int,
    expected_steps: int,
) -> None:
    assert (
        _optimizer_steps_for_positions(
            run,
            positions=positions,
            batch_size=batch_size,
        )
        == expected_steps
    )


def test_replay_warmup_delays_training_until_configured_position_count(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    make_trajectory: TrajectoryFactory,
) -> None:
    small_learner_config.batch_size = 4
    network = LunaNetwork(chess_game, small_learner_config)
    coach = Coach(
        chess_game,
        network,
        TrainingRunConfig(
            num_iters=3,
            num_episodes=1,
            train_steps_per_iter=2,
            replay_warmup_positions=6,
            checkpoint="",
            stockfish_eval_every=0,
        ),
    )

    with (
        patch.object(coach, "execute_episodes_batched", return_value=[make_trajectory(2)]),
        patch.object(network, "train_ezv2", return_value={}) as train,
        patch.object(coach, "_publish_checkpoint") as publish_checkpoint,
        patch.object(coach, "_reconcile_current_evaluations"),
    ):
        coach._learn_iterations(start_iteration=1, actor_pool=None)

    train.assert_called_once()
    assert train.call_args.kwargs["steps"] == 2
    publish_checkpoint.assert_called_once_with(3)


def test_dynamic_steps_use_fixed_lr_horizon_and_reconfigure_per_each_iteration(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    make_trajectory: TrajectoryFactory,
) -> None:
    small_learner_config.batch_size = 4
    network = LunaNetwork(chess_game, small_learner_config)
    coach = Coach(
        chess_game,
        network,
        TrainingRunConfig(
            num_iters=2,
            num_episodes=1,
            train_steps_per_iter=10,
            target_replay_ratio=2.0,
            lr_schedule_total_steps=1234,
            search_contempt_visit_limit=4,
            checkpoint="",
            stockfish_eval_every=0,
        ),
    )

    with (
        patch.object(coach, "execute_episodes_batched", return_value=[make_trajectory(5)]),
        patch.object(network, "train_ezv2", return_value={}) as train,
        patch.object(coach.replay, "configure_beta_annealing") as configure_beta,
        patch.object(coach, "_publish_checkpoint"),
        patch.object(coach, "_reconcile_current_evaluations"),
    ):
        coach._learn_iterations(start_iteration=1, actor_pool=None)

    assert [entry.kwargs["steps"] for entry in train.call_args_list] == [3, 3]
    assert [entry.kwargs["total_train_steps"] for entry in train.call_args_list] == [1234, 1234]
    assert all(entry.kwargs["mcts_for_reanalyze"].search_contempt_visit_limit is None for entry in train.call_args_list)
    assert configure_beta.call_args_list == [call(6), call(3)]


def test_resume_defers_beta_configuration_until_replay_can_train(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    run = TrainingRunConfig(
        num_iters=10,
        num_episodes=1,
        train_steps_per_iter=7,
        checkpoint="",
        stockfish_eval_every=0,
    )
    network = LunaNetwork(chess_game, small_learner_config)
    network._trainer_iteration = 4
    coach = Coach(chess_game, network, run)

    with (
        patch.object(coach.replay, "configure_beta_annealing") as configure_beta,
        patch.object(coach, "_learn_iterations") as learn_iterations,
    ):
        coach.learn()

    configure_beta.assert_not_called()
    learn_iterations.assert_called_once_with(5, actor_pool=None)


def test_resume_reconfigures_beta_over_actual_training_calls_after_skipped_iteration(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    make_trajectory: TrajectoryFactory,
) -> None:
    small_learner_config.batch_size = 4
    run = TrainingRunConfig(
        num_iters=7,
        num_episodes=1,
        train_steps_per_iter=2,
        checkpoint="",
        stockfish_eval_every=0,
    )
    network = LunaNetwork(chess_game, small_learner_config)
    network._trainer_iteration = 4
    coach = Coach(chess_game, network, run)
    trajectory = make_trajectory(2)

    def sample_for_each_optimizer_step(
        replay: PrioritizedReplayBuffer,
        steps: int,
        total_train_steps: int,
        **_: object,
    ) -> dict[str, float]:
        assert total_train_steps == 4
        network._lr_schedule_total_steps = total_train_steps
        for _step in range(steps):
            replay.sample(batch_size=4, unroll_steps=0)
        return {}

    with (
        patch.object(coach, "execute_episodes_batched", return_value=[trajectory]),
        patch.object(
            coach.replay,
            "configure_beta_annealing",
            wraps=coach.replay.configure_beta_annealing,
        ) as configure_beta,
        patch.object(network, "train_ezv2", side_effect=sample_for_each_optimizer_step) as train,
        patch.object(coach, "_publish_checkpoint"),
    ):
        coach._learn_iterations(start_iteration=5, actor_pool=None)

    assert configure_beta.call_args_list == [call(4), call(2)]
    assert train.call_count == 2
    assert coach.replay.beta == pytest.approx(1.0)
