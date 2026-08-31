"""Tests for Coach self-play (e.g. max ply truncation, batched self-play)."""

import json
from collections.abc import Sequence
from hashlib import file_digest
from pathlib import Path
from typing import cast
from unittest.mock import call, patch

import chess
import numpy as np
import pytest

from luna.coach import (
    Coach,
    _optimizer_steps_for_positions,
    _self_play_exploration_enabled,
    validate_fresh_checkpoint_target,
    validate_resume_checkpoint_target,
)
from luna.config import EzV2LearnerConfig, MCTSParams, TrainingRunConfig, WandbResumeMode
from luna.game.arena import Arena
from luna.game.chess_game import ChessGame, move_to_action
from luna.game.stockfish_eval import StockfishEvalScores, StockfishEvalSkipped
from luna.network import LunaNetwork
from luna.profiling import IterProfileStats, SelfPlayMCTSTimings
from luna.replay_buffer import PrioritizedReplayBuffer
from tests.conftest import TrajectoryFactory


def test_wandb_metrics_use_domain_specific_step_axes(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with (
        patch("luna.coach.wandb.init") as wandb_init,
        patch("luna.coach.wandb.define_metric") as define_metric,
    ):
        Coach(chess_game, network, TrainingRunConfig(), wandb_project="ChessRL")

    init_kwargs = wandb_init.call_args.kwargs
    assert init_kwargs["project"] == "ChessRL"
    assert init_kwargs["name"] is None
    assert "id" not in init_kwargs
    assert "resume" not in init_kwargs
    assert init_kwargs["config"]["training_phase_provenance"] is None
    assert define_metric.call_args_list == [
        call("global_step"),
        call("train/*", step_metric="global_step"),
        call("iteration"),
        call("replay_buffer_size", step_metric="iteration"),
        call("selfplay/*", step_metric="iteration"),
        call("performance/*", step_metric="iteration"),
        call("replay/*", step_metric="iteration"),
        call("benchmark/*", step_metric="iteration"),
        call("ladder/evaluation_step"),
        call("ladder/*", step_metric="ladder/evaluation_step"),
    ]


def test_wandb_config_records_training_phase_source_without_private_path(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    source = LunaNetwork(chess_game, small_learner_config)
    source._global_step = 123
    source._trainer_iteration = 17
    source.save_checkpoint(str(tmp_path), "source.pth.tar")
    source_path = tmp_path / "source.pth.tar"
    with source_path.open("rb") as source_file:
        expected_sha256 = file_digest(source_file, "sha256").hexdigest()
    phase = LunaNetwork(chess_game, small_learner_config)
    phase.initialize_training_phase(str(tmp_path), source_path.name)

    with (
        patch("luna.coach.wandb.init") as wandb_init,
        patch("luna.coach.wandb.define_metric"),
    ):
        Coach(chess_game, phase, TrainingRunConfig(), wandb_project="ChessRL")

    provenance_config = wandb_init.call_args.kwargs["config"]["training_phase_provenance"]
    assert provenance_config == {
        "source_checkpoint_sha256": expected_sha256,
        "source_trainer_iteration": 17,
        "source_global_step": 123,
    }
    serialized_config = json.dumps(provenance_config)
    assert str(tmp_path) not in serialized_config
    assert source_path.name not in serialized_config


@pytest.mark.parametrize("resume_mode", ["allow", "never", "must"])
def test_wandb_run_id_uses_requested_resume_policy(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    resume_mode: WandbResumeMode,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with (
        patch("luna.coach.wandb.init") as wandb_init,
        patch("luna.coach.wandb.define_metric"),
    ):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(),
            wandb_project="ChessRL",
            wandb_run_id="luna-throughput-phase-v1",
            wandb_resume=resume_mode,
        )

    init_kwargs = wandb_init.call_args.kwargs
    assert init_kwargs["project"] == "ChessRL"
    assert init_kwargs["id"] == "luna-throughput-phase-v1"
    assert init_kwargs["resume"] == resume_mode


def test_wandb_display_name_is_independent_of_stable_run_id(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with (
        patch("luna.coach.wandb.init") as wandb_init,
        patch("luna.coach.wandb.define_metric"),
    ):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(),
            wandb_project="ChessRL",
            wandb_run_id="luna-strength-1500-v1",
            wandb_run_name="Luna Strength 1500 v1",
            wandb_resume="never",
        )

    init_kwargs = wandb_init.call_args.kwargs
    assert init_kwargs["id"] == "luna-strength-1500-v1"
    assert init_kwargs["name"] == "Luna Strength 1500 v1"


@pytest.mark.parametrize("resume_mode", ["never", "must"])
def test_wandb_resume_policy_is_not_forwarded_without_run_id(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    resume_mode: WandbResumeMode,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with (
        patch("luna.coach.wandb.init") as wandb_init,
        patch("luna.coach.wandb.define_metric"),
    ):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(),
            wandb_project="ChessRL",
            wandb_resume=resume_mode,
        )

    init_kwargs = wandb_init.call_args.kwargs
    assert "id" not in init_kwargs
    assert "resume" not in init_kwargs


def test_coach_rejects_invalid_wandb_resume_policy(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with pytest.raises(ValueError, match="wandb_resume"):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(),
            wandb_resume=cast(WandbResumeMode, "sometimes"),
        )


@pytest.mark.parametrize("run_name", ["", "   ", " leading", "trailing "])
def test_coach_rejects_invalid_wandb_display_name(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    run_name: str,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with pytest.raises(ValueError, match="wandb_run_name"):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(),
            wandb_run_name=run_name,
        )


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
        patch("luna.coach.wandb.run", object()),
        patch("luna.coach.wandb.log") as wandb_log,
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
        patch("luna.coach.wandb.run", object()),
        patch("luna.coach.wandb.log") as wandb_log,
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


def test_gumbel_selfplay_reenables_exploration_for_a_repeated_root() -> None:
    board = chess.Board()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(move)
    assert board.is_repetition(2)

    gumbel = TrainingRunConfig(search_mode="gumbel", temp_threshold=1)
    puct = TrainingRunConfig(search_mode="puct", temp_threshold=1)

    assert _self_play_exploration_enabled(board, 5, gumbel) is True
    assert _self_play_exploration_enabled(board, 5, puct) is False


def _repetition_threat_board() -> chess.Board:
    board = chess.Board()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"):
        board.push_uci(move)
    assert board.outcome(claim_draw=True) is None
    return board


def test_single_selfplay_retries_a_move_that_enables_threefold_and_logs_guard_counters(
    monkeypatch: pytest.MonkeyPatch,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    board = _repetition_threat_board()
    repetition_action = move_to_action(chess.Move.from_uci("f3g1"))
    safe_action = move_to_action(chess.Move.from_uci("b1c3"))
    restrictions: list[set[int] | None] = []

    class _Search:
        def __init__(self, _game: ChessGame, _network: LunaNetwork, _params: MCTSParams) -> None:
            self.last_action: int | None = None

        def search_latent(
            self,
            _board: chess.Board,
            temp: float,
            *,
            add_exploration_noise: bool | None,
            allowed_root_actions: Sequence[int] | None = None,
        ) -> tuple[np.ndarray, float]:
            assert temp == 1.0
            assert add_exploration_noise is True
            restriction = None if allowed_root_actions is None else set(allowed_root_actions)
            restrictions.append(restriction)
            selected_action = repetition_action if restriction is None else safe_action
            if restriction is not None:
                assert repetition_action not in restriction
                assert safe_action in restriction
            self.last_action = selected_action
            policy = np.zeros(chess_game.get_action_size(), dtype=np.float32)
            policy[selected_action] = 1.0
            return policy, 0.0

    monkeypatch.setattr("luna.coach.MCTS", _Search)
    monkeypatch.setattr(chess_game, "get_init_board", lambda: board.copy(stack=True))
    coach = Coach(
        chess_game,
        LunaNetwork(chess_game, small_learner_config),
        TrainingRunConfig(
            num_mcts_sims=1,
            max_ply=1,
            temp_threshold=100,
            self_play_repetition_guard=True,
        ),
    )

    trajectory = coach.execute_episode()

    assert trajectory.actions.tolist() == [safe_action]
    assert trajectory.valids[0, repetition_action]
    assert trajectory.root_policies[0, repetition_action] == 0.0
    assert trajectory.repetition_guard_attempts == 1
    assert trajectory.repetition_guard_interventions == 1
    assert trajectory.repetition_guard_forced_fallbacks == 0
    assert trajectory.repetition_guard_excluded_actions >= 1
    assert restrictions[0] is None
    assert restrictions[1] is not None

    with patch("luna.coach.wandb.run", object()), patch("luna.coach.wandb.log") as wandb_log:
        coach._log_iteration_metrics(
            1,
            [trajectory],
            IterProfileStats(iter_index=1),
        )

    metrics = wandb_log.call_args.args[0]
    assert metrics["selfplay/repetition_guard_attempts"] == 1
    assert metrics["selfplay/repetition_guard_interventions"] == 1
    assert metrics["selfplay/repetition_guard_forced_fallbacks"] == 0
    assert metrics["selfplay/repetition_guard_excluded_actions"] == trajectory.repetition_guard_excluded_actions
    assert metrics["selfplay/repetition_guard_attempt_fraction"] == 1.0
    assert metrics["selfplay/repetition_guard_intervention_fraction"] == 1.0


def test_batched_selfplay_retries_only_with_non_repetition_root_actions(
    monkeypatch: pytest.MonkeyPatch,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    board = _repetition_threat_board()
    repetition_action = move_to_action(chess.Move.from_uci("f3g1"))
    safe_action = move_to_action(chess.Move.from_uci("b1c3"))
    restrictions: list[list[set[int] | None]] = []

    class _BatchedSearch:
        def __init__(
            self,
            game: ChessGame,
            _network: LunaNetwork,
            _params: MCTSParams,
            timings: SelfPlayMCTSTimings | None = None,
        ) -> None:
            del timings
            self.game = game
            self.last_actions: list[int] = []

        def search_batch(
            self,
            boards: list[chess.Board],
            temp: float,
            *,
            add_exploration_noise: bool | Sequence[bool] | None,
            allowed_root_actions: Sequence[Sequence[int] | None] | None = None,
        ) -> list[tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
            assert temp == 1.0
            assert add_exploration_noise == [True]
            normalized: list[set[int] | None] = (
                [None] * len(boards)
                if allowed_root_actions is None
                else [None if actions is None else set(actions) for actions in allowed_root_actions]
            )
            restrictions.append(normalized)
            selected_actions: list[int] = []
            for restriction in normalized:
                selected_action = repetition_action if restriction is None else safe_action
                if restriction is not None:
                    assert repetition_action not in restriction
                    assert safe_action in restriction
                selected_actions.append(selected_action)
            self.last_actions = selected_actions
            outputs = []
            for root, selected_action in zip(boards, selected_actions):
                policy = np.zeros(self.game.get_action_size(), dtype=np.float32)
                policy[selected_action] = 1.0
                outputs.append(
                    (
                        policy,
                        0.0,
                        self.game.to_array(root),
                        self.game.get_valid_moves(root, 1),
                    )
                )
            return outputs

    monkeypatch.setattr("luna.coach.BatchedMCTS", _BatchedSearch)
    monkeypatch.setattr(chess_game, "get_init_board", lambda: board.copy(stack=True))
    coach = Coach(
        chess_game,
        LunaNetwork(chess_game, small_learner_config),
        TrainingRunConfig(
            num_mcts_sims=1,
            max_ply=1,
            temp_threshold=100,
            parallel_games=1,
            self_play_repetition_guard=True,
        ),
    )

    trajectory = coach.execute_episodes_batched(1, progress=False)[0]

    assert trajectory.actions.tolist() == [safe_action]
    assert trajectory.valids[0, repetition_action]
    assert trajectory.root_policies[0, repetition_action] == 0.0
    assert trajectory.repetition_guard_attempts == 1
    assert trajectory.repetition_guard_interventions == 1
    assert trajectory.repetition_guard_forced_fallbacks == 0
    assert trajectory.repetition_guard_excluded_actions >= 1
    assert restrictions[0] == [None]
    assert restrictions[1][0] is not None


def test_selfplay_metrics_report_replay_samples_per_new_position(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    make_trajectory: TrajectoryFactory,
) -> None:
    small_learner_config.batch_size = 8
    coach = Coach(chess_game, LunaNetwork(chess_game, small_learner_config), TrainingRunConfig())

    with patch("luna.coach.wandb.run", object()), patch("luna.coach.wandb.log") as wandb_log:
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


def test_gumbel_selfplay_executes_proposal_but_stores_improved_target(
    monkeypatch: pytest.MonkeyPatch,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    target_action = move_to_action(chess.Move.from_uci("d2d4"))
    proposed_action = move_to_action(chess.Move.from_uci("e2e4"))

    class _Search:
        def __init__(self, _game: ChessGame, _network: LunaNetwork, _params: MCTSParams) -> None:
            self.last_action = proposed_action

        def search_latent(
            self,
            _board: chess.Board,
            temp: float,
            *,
            add_exploration_noise: bool | None,
        ) -> tuple[np.ndarray, float]:
            assert temp == 1.0
            assert add_exploration_noise is True
            policy = np.zeros(chess_game.get_action_size(), dtype=np.float32)
            policy[target_action] = 1.0
            return policy, 0.0

    monkeypatch.setattr("luna.coach.MCTS", _Search)
    network = LunaNetwork(chess_game, small_learner_config)
    coach = Coach(
        chess_game,
        network,
        TrainingRunConfig(num_mcts_sims=1, max_ply=1, temp_threshold=2),
    )

    trajectory = coach.execute_episode()

    assert trajectory.actions.tolist() == [proposed_action]
    assert int(np.argmax(trajectory.root_policies[0])) == target_action


def test_batched_gumbel_selfplay_routes_per_root_exploration_and_proposals(
    monkeypatch: pytest.MonkeyPatch,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    target_action = move_to_action(chess.Move.from_uci("d2d4"))
    proposed_action = move_to_action(chess.Move.from_uci("e2e4"))

    class _BatchedSearch:
        def __init__(
            self,
            game: ChessGame,
            _network: LunaNetwork,
            _params: MCTSParams,
            timings: SelfPlayMCTSTimings | None = None,
        ) -> None:
            del timings
            self.game = game
            self.last_actions: list[int | None] = []

        def search_batch(
            self,
            boards: list[chess.Board],
            temp: float,
            *,
            add_exploration_noise: bool | Sequence[bool] | None,
        ) -> list[tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
            assert temp == 1.0
            assert add_exploration_noise == [True]
            self.last_actions = [proposed_action]
            outputs = []
            for board in boards:
                policy = np.zeros(self.game.get_action_size(), dtype=np.float32)
                policy[target_action] = 1.0
                outputs.append(
                    (
                        policy,
                        0.0,
                        self.game.to_array(board),
                        self.game.get_valid_moves(board, 1),
                    )
                )
            return outputs

    monkeypatch.setattr("luna.coach.BatchedMCTS", _BatchedSearch)
    network = LunaNetwork(chess_game, small_learner_config)
    coach = Coach(
        chess_game,
        network,
        TrainingRunConfig(
            num_mcts_sims=1,
            max_ply=1,
            temp_threshold=2,
            parallel_games=1,
        ),
    )

    trajectory = coach.execute_episodes_batched(1)[0]

    assert trajectory.actions.tolist() == [proposed_action]
    assert int(np.argmax(trajectory.root_policies[0])) == target_action


class TestArenaMaxPly:
    def test_play_game_returns_draw_when_max_ply_reached(self, chess_game: ChessGame) -> None:
        def pick_first(canonical_board: chess.Board) -> int:
            valids = chess_game.get_valid_moves(canonical_board, 1)
            return int(np.argmax(valids))

        arena = Arena(pick_first, pick_first, chess_game)
        result = arena.play_game(verbose=False, max_ply=3)
        assert result == 0.0

    def test_initial_board_selects_its_side_to_move_without_mutating_caller(self, chess_game: ChessGame) -> None:
        initial_board = chess.Board()
        initial_board.push_uci("e2e4")
        initial_fen = initial_board.fen()
        called_players: list[int] = []

        def player_one(canonical_board: chess.Board) -> int:
            called_players.append(1)
            return int(np.argmax(chess_game.get_valid_moves(canonical_board, 1)))

        def player_two(canonical_board: chess.Board) -> int:
            called_players.append(-1)
            assert canonical_board.turn == chess.WHITE
            assert len(canonical_board.move_stack) == 1
            return int(np.argmax(chess_game.get_valid_moves(canonical_board, 1)))

        result = Arena(player_one, player_two, chess_game).play_game(
            max_ply=1,
            initial_board=initial_board,
        )

        assert result == 0.0
        assert called_players == [-1]
        assert initial_board.fen() == initial_fen
        assert len(initial_board.move_stack) == 1


def test_checkpoint_retention_keeps_top_k(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    tmp_path: Path,
) -> None:
    run = TrainingRunConfig(
        num_mcts_sims=2,
        dir_noise=False,
        checkpoint=str(tmp_path.resolve()),
        checkpoint_top_k=2,
        recurrent_policy_topk=None,
    )
    nnet = LunaNetwork(chess_game, small_learner_config)
    coach = Coach(chess_game, nnet, run)
    coach._publish_checkpoint(1)
    coach._publish_checkpoint(2)
    coach._publish_checkpoint(3)

    assert not (tmp_path / "checkpoint_1.pth.tar").is_file()
    assert (tmp_path / "checkpoint_2.pth.tar").is_file()
    assert (tmp_path / "checkpoint_3.pth.tar").is_file()


def test_orphaned_best_evaluation_metadata_fails_loudly(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    run = TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0)
    coach = Coach(chess_game, LunaNetwork(chess_game, small_learner_config), run)
    coach._publish_checkpoint(1)
    (tmp_path / "best_eval.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata exists without its best checkpoint"):
        coach._update_best_from_stockfish(1, StockfishEvalScores(model_wins=1, draws=1, stockfish_wins=0))
    assert (tmp_path / "latest.pth.tar").is_file()
    assert not (tmp_path / "best.pth.tar").exists()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_best_checkpoint_record_rejects_invalid_score(
    tmp_path: Path,
    score: float,
) -> None:
    with pytest.raises(RuntimeError, match="finite and between zero and one"):
        Coach._validate_best_record(
            {
                "schema_version": 1,
                "iteration": 1,
                "score": score,
                "protocol": {},
                "source_checkpoint_sha256": "a" * 64,
            },
            protocol={},
            best_path=tmp_path / "best.pth.tar",
            trainer_iteration=1,
        )


def test_best_checkpoint_record_repairs_metadata_and_is_bound_to_protocol(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    run = TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0)
    coach = Coach(chess_game, LunaNetwork(chess_game, small_learner_config), run)
    coach._publish_checkpoint(1)
    score = StockfishEvalScores(model_wins=1, draws=1, stockfish_wins=0)
    coach._update_best_from_stockfish(1, score)
    metadata_path = tmp_path / "best_eval.json"
    metadata: dict[str, object] = json.loads(metadata_path.read_text(encoding="utf-8"))

    protocol = cast(dict[str, object], metadata["protocol"])
    assert protocol["opening_suite_version"] == 1
    metadata_path.write_text("not-json", encoding="utf-8")
    assert Coach._previous_best_score(tmp_path, protocol) == 0.75
    repaired = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert repaired["protocol"] == protocol

    changed_protocol = dict(protocol)
    changed_protocol["opening_suite_version"] = 2
    with pytest.raises(RuntimeError, match="protocol differs"):
        Coach._previous_best_score(tmp_path, changed_protocol)


def test_configured_external_evaluation_failure_stops_promotion(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    run = TrainingRunConfig(checkpoint=str(tmp_path))
    coach = Coach(chess_game, LunaNetwork(chess_game, small_learner_config), run)

    with pytest.raises(RuntimeError, match=r"External evaluation did not complete.*no_engine"):
        coach._update_best_from_stockfish(1, StockfishEvalSkipped("no_engine", "binary not found"))

    assert not (tmp_path / "best.pth.tar").exists()


def test_fresh_training_refuses_managed_checkpoint_without_clobbering_it(tmp_path: Path) -> None:
    latest_path = tmp_path / "latest.pth.tar"
    original = b"existing checkpoint"
    latest_path.write_bytes(original)
    run = TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0)

    with pytest.raises(FileExistsError, match="Fresh training would overwrite managed files"):
        validate_fresh_checkpoint_target(run)

    assert latest_path.read_bytes() == original


@pytest.mark.parametrize(
    "managed_name",
    [
        "checkpoint_2.pth.tar",
        "latest.pth.tar",
        "best.pth.tar",
        "best_eval.json",
        "benchmark_state.json",
        "fairy_ladder.json",
    ],
)
def test_resume_refuses_managed_state_from_another_directory(tmp_path: Path, managed_name: str) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    managed_path = target / managed_name
    managed_path.write_bytes(b"another run")
    run = TrainingRunConfig(checkpoint=str(target), stockfish_eval_every=0)

    with pytest.raises(FileExistsError, match="another checkpoint lineage"):
        validate_resume_checkpoint_target(run, source / "latest.pth.tar")

    assert managed_path.read_bytes() == b"another run"


def test_resume_allows_source_directory_or_empty_new_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    empty_target = tmp_path / "empty-target"
    source.mkdir()
    (source / "latest.pth.tar").write_bytes(b"resume checkpoint")

    validate_resume_checkpoint_target(
        TrainingRunConfig(checkpoint=str(source), stockfish_eval_every=0),
        source / "latest.pth.tar",
    )
    validate_resume_checkpoint_target(
        TrainingRunConfig(checkpoint=str(empty_target), stockfish_eval_every=0),
        source / "latest.pth.tar",
    )


def test_explicit_evaluation_migration_allows_only_sidecars_in_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    for name in ("benchmark_state.json", "fairy_ladder.json"):
        (target / name).write_text("{}", encoding="utf-8")
    run = TrainingRunConfig(checkpoint=str(target), stockfish_eval_every=0)

    validate_resume_checkpoint_target(
        run,
        source / "latest.pth.tar",
        allow_evaluation_artifacts_only=True,
    )
    (target / "latest.pth.tar").write_bytes(b"different lineage")

    with pytest.raises(FileExistsError, match="another checkpoint lineage"):
        validate_resume_checkpoint_target(
            run,
            source / "latest.pth.tar",
            allow_evaluation_artifacts_only=True,
        )


def test_resume_resolves_traversal_before_comparing_lineages(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "latest.pth.tar").write_bytes(b"source run")
    target_latest = target / "latest.pth.tar"
    target_latest.write_bytes(b"target run")
    traversing_source = target / ".." / "source" / "latest.pth.tar"

    with pytest.raises(FileExistsError, match="another checkpoint lineage"):
        validate_resume_checkpoint_target(
            TrainingRunConfig(checkpoint=str(target), stockfish_eval_every=0),
            traversing_source,
        )

    assert target_latest.read_bytes() == b"target run"


def test_zero_counter_checkpoint_is_recognized_as_resume(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    initial = LunaNetwork(chess_game, small_learner_config)
    initial.save_checkpoint(str(tmp_path), "latest.pth.tar")
    resumed = LunaNetwork(chess_game, small_learner_config)
    resumed.load_checkpoint(str(tmp_path), "latest.pth.tar")
    coach = Coach(
        chess_game,
        resumed,
        TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0),
    )

    coach._assert_checkpoint_target()
    coach._assert_checkpoint_target()

    assert resumed._global_step == 0
    assert resumed._trainer_iteration == 0
    assert (tmp_path / "latest.pth.tar").is_file()


def test_zero_counter_resume_rejects_newer_numbered_lineage(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    initial = LunaNetwork(chess_game, small_learner_config)
    initial.save_checkpoint(str(tmp_path), "latest.pth.tar")
    newer = LunaNetwork(chess_game, small_learner_config)
    newer._trainer_iteration = 5
    newer.save_checkpoint(str(tmp_path), "checkpoint_5.pth.tar")
    resumed = LunaNetwork(chess_game, small_learner_config)
    resumed.load_checkpoint(str(tmp_path), "latest.pth.tar")
    coach = Coach(
        chess_game,
        resumed,
        TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0),
    )

    coach._assert_checkpoint_target()
    with pytest.raises(RuntimeError, match="newer training state"):
        coach._assert_checkpoint_lineage()


def test_publish_checkpoint_refuses_to_replace_numbered_snapshot(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    checkpoint_path = tmp_path / "checkpoint_1.pth.tar"
    original = b"existing checkpoint"
    checkpoint_path.write_bytes(original)
    network = LunaNetwork(chess_game, small_learner_config)
    coach = Coach(
        chess_game,
        network,
        TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0),
    )

    with pytest.raises(FileExistsError, match="immutable numbered checkpoint"):
        coach._publish_checkpoint(1)

    assert checkpoint_path.read_bytes() == original
    assert network._trainer_iteration == 0


def test_publish_checkpoint_restores_iteration_when_numbered_save_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    network._trainer_iteration = 4
    coach = Coach(
        chess_game,
        network,
        TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0),
    )

    def fail_save(folder: str, filename: str) -> None:
        del folder, filename
        raise OSError("storage unavailable")

    monkeypatch.setattr(network, "save_checkpoint", fail_save)

    with pytest.raises(OSError, match="storage unavailable"):
        coach._publish_checkpoint(5)

    assert network._trainer_iteration == 4
    assert not (tmp_path / "checkpoint_5.pth.tar").exists()


def test_publish_checkpoint_refuses_to_roll_back_a_newer_lineage(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    newer_path = tmp_path / "checkpoint_5.pth.tar"
    newer_path.write_bytes(b"newer checkpoint")
    network = LunaNetwork(chess_game, small_learner_config)
    network._trainer_iteration = 1
    coach = Coach(
        chess_game,
        network,
        TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0),
    )

    with pytest.raises(FileExistsError, match="Refusing non-monotonic checkpoint"):
        coach._publish_checkpoint(2)

    assert newer_path.read_bytes() == b"newer checkpoint"
    assert not (tmp_path / "latest.pth.tar").exists()


def test_publish_checkpoint_refuses_to_roll_back_latest_only_lineage(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    newer = LunaNetwork(chess_game, small_learner_config)
    newer._trainer_iteration = 5
    newer.save_checkpoint(str(tmp_path), "latest.pth.tar")
    resumed = LunaNetwork(chess_game, small_learner_config)
    resumed._trainer_iteration = 1
    coach = Coach(
        chess_game,
        resumed,
        TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0),
    )

    with pytest.raises(FileExistsError, match="Refusing non-monotonic checkpoint"):
        coach._publish_checkpoint(2)

    assert LunaNetwork.checkpoint_trainer_iteration(tmp_path / "latest.pth.tar") == 5


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
