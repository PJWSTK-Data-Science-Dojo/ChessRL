"""Tests for Coach self-play (e.g. max ply truncation, batched self-play)."""

from unittest.mock import patch

import chess
import numpy as np
import pytest

from luna.coach import (
    Coach,
    _self_play_exploration_enabled,
)
from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.game.chess_game import ChessGame
from luna.network import LunaNetwork
from luna.profiling import IterProfileStats
from tests.conftest import TrajectoryFactory


class TestMaxPlyTruncation:
    def test_execute_episode_stops_at_max_ply_with_zero_draw_reward(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        nnet = LunaNetwork(chess_game, small_learner_config)
        run = TrainingRunConfig(
            num_mcts_sims=2,
            max_ply=5,
            dir_noise=False,
            temp_threshold=1,
            recurrent_policy_topk=None,
        )
        coach = Coach(chess_game, nnet, run)
        traj = coach.execute_episode()

        assert len(traj.actions) == 5
        assert len(traj.rewards) == 5
        assert all(r == 0.0 for r in traj.rewards[:-1])
        assert np.isclose(traj.rewards[-1], 0.0)
        assert traj.truncated is True
        assert traj.truncation_bootstrap_value is not None
        assert np.isfinite(traj.truncation_bootstrap_value)


class TestBatchedSelfPlay:
    def test_execute_episodes_batched_returns_trajectories(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        nnet = LunaNetwork(chess_game, small_learner_config)
        run = TrainingRunConfig(
            num_mcts_sims=2,
            max_ply=5,
            dir_noise=False,
            temp_threshold=1,
            parallel_games=2,
            recurrent_policy_topk=None,
        )
        coach = Coach(chess_game, nnet, run)
        trajs = coach.execute_episodes_batched(num_episodes=3)

        assert len(trajs) == 3
        for t in trajs:
            assert t.game_length > 0
            assert t.game_length <= 5
            assert t.observations.shape[0] == t.game_length
            assert t.truncated is True
            assert t.truncation_bootstrap_value is not None
            assert np.isfinite(t.truncation_bootstrap_value)


def test_skipped_training_still_logs_iteration_observability(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    make_trajectory: TrajectoryFactory,
) -> None:
    small_learner_config.batch_size = 8
    coach = Coach(
        chess_game,
        LunaNetwork(chess_game, small_learner_config),
        TrainingRunConfig(
            num_iters=1,
            num_episodes=2,
            checkpoint="",
            stockfish_eval_every=0,
        ),
    )
    trajectories = [make_trajectory(2), make_trajectory(3, truncated=True)]

    with (
        patch.object(coach, "execute_episodes_batched", return_value=trajectories),
        patch("luna.coach_metrics.wandb.run", object()),
        patch("luna.coach_metrics.wandb.log") as wandb_log,
    ):
        coach._learn_iterations(start_iteration=1, actor_pool=None)

    metrics = wandb_log.call_args.args[0]
    assert metrics["iteration"] == 1
    assert metrics["replay_buffer_size"] == 5
    assert metrics["selfplay/games"] == 2
    assert metrics["selfplay/positions"] == 5
    assert metrics["selfplay/avg_ply"] == 2.5
    assert metrics["selfplay/max_ply_fraction"] == 0.5
    assert metrics["selfplay/truncated_fraction"] == 0.5
    assert metrics["selfplay/truncation_bootstrap_mean_abs"] == 0.0
    assert metrics["selfplay/decisive_fraction"] == 0.0
    assert metrics["selfplay/draw_fraction"] == 0.5
    assert metrics["selfplay/white_win_fraction"] == 0.0
    assert metrics["selfplay/black_win_fraction"] == 0.0
    assert metrics["selfplay/policy_entropy"] > 0.0
    assert metrics["performance/self_play_seconds"] > 0.0
    assert metrics["performance/self_play_positions_per_second"] == pytest.approx(
        5 / metrics["performance/self_play_seconds"]
    )
    assert metrics["performance/train_seconds"] == 0.0
    assert metrics["performance/iteration_seconds"] >= metrics["performance/self_play_seconds"]
    assert metrics["replay/size"] == 5
    assert metrics["replay/beta"] == 0.4


def test_completed_iteration_is_logged_before_external_evaluation(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    make_trajectory: TrajectoryFactory,
) -> None:
    small_learner_config.batch_size = 1
    network = LunaNetwork(chess_game, small_learner_config)
    coach = Coach(
        chess_game,
        network,
        TrainingRunConfig(num_iters=1, num_episodes=1, train_steps_per_iter=1, checkpoint="", stockfish_eval_every=0),
    )
    events: list[str] = []

    with (
        patch.object(coach, "execute_episodes_batched", return_value=[make_trajectory(2)]),
        patch.object(network, "train_ezv2", return_value={}),
        patch.object(coach, "_publish_checkpoint", side_effect=lambda _iteration: events.append("checkpoint")),
        patch.object(coach, "_log_iteration_metrics", side_effect=lambda *_args, **_kwargs: events.append("metrics")),
        patch.object(
            coach, "_reconcile_current_evaluations", side_effect=lambda _iteration: events.append("evaluation")
        ),
    ):
        coach._learn_iterations(start_iteration=1, actor_pool=None)

    assert events == ["checkpoint", "metrics", "evaluation"]


def test_selfplay_outcome_metrics_respect_ply_color_and_exclude_truncations(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    make_trajectory: TrajectoryFactory,
) -> None:
    coach = Coach(chess_game, LunaNetwork(chess_game, small_learner_config), TrainingRunConfig())
    white_win = make_trajectory(1, termination=chess.Termination.CHECKMATE)
    white_win.rewards[-1] = 1.0
    black_win = make_trajectory(2, termination=chess.Termination.CHECKMATE)
    black_win.rewards[-1] = 1.0
    draw = make_trajectory(3, termination=chess.Termination.THREEFOLD_REPETITION)
    truncated = make_trajectory(4, truncated=True)

    with (
        patch("luna.coach_metrics.wandb.run", object()),
        patch("luna.coach_metrics.wandb.log") as wandb_log,
    ):
        coach._log_iteration_metrics(
            1,
            [white_win, black_win, draw, truncated],
            IterProfileStats(iter_index=1),
        )

    metrics = wandb_log.call_args.args[0]
    assert metrics["selfplay/white_win_fraction"] == 0.25
    assert metrics["selfplay/black_win_fraction"] == 0.25
    assert metrics["selfplay/draw_fraction"] == 0.25
    assert metrics["selfplay/decisive_fraction"] == 0.5
    assert metrics["selfplay/truncated_fraction"] == 0.25
    assert metrics["selfplay/checkmate_fraction"] == 0.5
    assert metrics["selfplay/threefold_repetition_fraction"] == 0.25
    assert metrics["selfplay/unknown_termination_fraction"] == 0.0


def test_decisive_to_draw_ratio_stays_positive_when_no_games_draw(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    make_trajectory: TrajectoryFactory,
) -> None:
    coach = Coach(chess_game, LunaNetwork(chess_game, small_learner_config), TrainingRunConfig())
    decisive = make_trajectory(1, termination=chess.Termination.CHECKMATE)
    decisive.rewards[-1] = 1.0

    with patch("luna.coach_metrics.wandb.run", object()), patch("luna.coach_metrics.wandb.log") as wandb_log:
        coach._log_iteration_metrics(1, [decisive], IterProfileStats(iter_index=1))

    assert wandb_log.call_args.args[0]["selfplay/decisive_to_draw_ratio"] == 1.0


def test_root_value_terminal_calibration_alternates_perspective_and_excludes_truncations(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    make_trajectory: TrajectoryFactory,
) -> None:
    coach = Coach(chess_game, LunaNetwork(chess_game, small_learner_config), TrainingRunConfig())
    completed = make_trajectory(4, termination=chess.Termination.CHECKMATE)
    completed.rewards[-1] = 1.0
    completed.root_values[:] = [0.0, 0.5, -0.5, 1.0]
    truncated = make_trajectory(4, truncated=True)
    truncated.root_values[:] = 100.0

    with patch("luna.coach_metrics.wandb.run", object()), patch("luna.coach_metrics.wandb.log") as wandb_log:
        coach._log_iteration_metrics(1, [completed, truncated], IterProfileStats(iter_index=1))

    metrics = wandb_log.call_args.args[0]
    assert metrics["selfplay/root_value_terminal_positions"] == 4
    assert metrics["selfplay/root_value_terminal_mae"] == pytest.approx(0.5)
    assert metrics["selfplay/root_value_terminal_bias"] == pytest.approx(0.25)
    assert metrics["selfplay/root_value_terminal_mean"] == pytest.approx(0.25)


def test_gumbel_selfplay_reenables_exploration_for_a_repeated_root() -> None:
    board = chess.Board()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(move)
    assert board.is_repetition(2)

    gumbel = TrainingRunConfig(search_mode="gumbel", temp_threshold=1)
    exact_gumbel = TrainingRunConfig(search_mode="gumbel", tree_state_mode="exact", temp_threshold=1)
    puct = TrainingRunConfig(search_mode="puct", temp_threshold=1)

    assert _self_play_exploration_enabled(board, 5, gumbel) is True
    assert _self_play_exploration_enabled(board, 5, exact_gumbel) is False
    assert _self_play_exploration_enabled(board, 5, puct) is False
