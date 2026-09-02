"""Playout Cap Randomization self-play tests."""

from collections.abc import Sequence
from dataclasses import replace

import chess
import numpy as np
import pytest

from luna.coach import Coach
from luna.coach_metrics import _self_play_metrics, _summarize_trajectories
from luna.coach_self_play import select_self_play_search_plan
from luna.config import EzV2LearnerConfig, MCTSParams, TrainingRunConfig, validate_training_configuration
from luna.game.chess_game import ChessGame, move_to_action
from luna.mcts_search_contempt import SearchContemptStats
from luna.network import LunaNetwork
from luna.profiling import SelfPlayMCTSTimings
from luna.targets import build_unroll_targets
from tests.conftest import TrajectoryFactory


def _pcr_run(*, parallel_games: int = 1) -> TrainingRunConfig:
    return TrainingRunConfig(
        num_mcts_sims=1,
        max_ply=1,
        temp_threshold=10,
        parallel_games=parallel_games,
        playout_cap_full_sims=8,
        playout_cap_fast_sims=2,
        playout_cap_full_probability=0.25,
    )


def _pcr_learner(config: EzV2LearnerConfig) -> EzV2LearnerConfig:
    return replace(config, reanalyze_policy=False)


def test_search_plan_draws_full_and_fast_cohorts(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _pcr_run()
    monkeypatch.setattr("luna.coach_self_play.np.random.random", lambda: 0.1)
    assert select_self_play_search_plan(run).simulations == 8
    assert select_self_play_search_plan(run).train_policy is True

    monkeypatch.setattr("luna.coach_self_play.np.random.random", lambda: 0.9)
    assert select_self_play_search_plan(run).simulations == 2
    assert select_self_play_search_plan(run).train_policy is False


def test_disabled_pcr_keeps_the_configured_search_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("luna.coach_self_play.np.random.random", lambda: pytest.fail("unexpected PCR draw"))

    plan = select_self_play_search_plan(TrainingRunConfig(num_mcts_sims=7))

    assert plan.simulations == 7
    assert plan.train_policy is True


def test_policy_entropy_reports_full_and_fast_search_separately(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    make_trajectory: TrajectoryFactory,
) -> None:
    trajectory = make_trajectory(2)
    trajectory.policy_train_mask = np.asarray([True, False])
    trajectory.root_policies[1] = 0.0
    trajectory.root_policies[1, 0] = 1.0
    coach = Coach(chess_game, LunaNetwork(chess_game, _pcr_learner(small_learner_config)), _pcr_run())

    metrics = _self_play_metrics(coach, _summarize_trajectories([trajectory]), optimizer_steps=0)

    assert metrics["selfplay/policy_entropy"] > 0.0
    assert metrics["selfplay/full_search_policy_entropy"] == metrics["selfplay/policy_entropy"]
    assert metrics["selfplay/fast_search_policy_entropy"] == 0.0


def test_serial_fast_search_disables_noise_and_policy_training(
    monkeypatch: pytest.MonkeyPatch,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    action = move_to_action(chess.Move.from_uci("e2e4"))

    class _Search:
        def __init__(self, _game: ChessGame, _network: LunaNetwork, _params: MCTSParams) -> None:
            self.last_action = action
            self.last_search_contempt_stats = SearchContemptStats()

        def search_latent(
            self,
            _board: chess.Board,
            num_sims: int | None,
            temp: float,
            *,
            add_exploration_noise: bool | None,
        ) -> tuple[np.ndarray, float]:
            assert num_sims == 2
            assert temp == 1.0
            assert add_exploration_noise is False
            policy = np.zeros(chess_game.get_action_size(), dtype=np.float32)
            policy[action] = 1.0
            return policy, 0.0

    coach = Coach(chess_game, LunaNetwork(chess_game, _pcr_learner(small_learner_config)), _pcr_run())
    monkeypatch.setattr("luna.coach_self_play.MCTS", _Search)
    monkeypatch.setattr("luna.coach_self_play.np.random.random", lambda: 0.9)

    trajectory = coach.execute_episode()
    targets = build_unroll_targets(trajectory, 0, unroll_steps=0, td_steps=1)

    assert trajectory.policy_train_mask.tolist() == [False]
    assert targets["policy_mask"] == [0.0]
    metrics = _self_play_metrics(coach, _summarize_trajectories([trajectory]), optimizer_steps=0)
    assert metrics["selfplay/playout_cap_full_fraction"] == 0.0
    assert metrics["selfplay/avg_mcts_sims"] == 2.0


def test_batched_fast_search_uses_one_cohort_for_the_pool_step(
    monkeypatch: pytest.MonkeyPatch,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    action = move_to_action(chess.Move.from_uci("e2e4"))
    calls: list[tuple[int | None, list[bool]]] = []

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
            self.last_search_contempt_stats: list[SearchContemptStats] = []

        def search_batch(
            self,
            boards: list[chess.Board],
            num_sims: int | None,
            temp: float,
            *,
            add_exploration_noise: bool | Sequence[bool] | None,
        ) -> list[tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
            assert temp == 1.0
            noise = list(add_exploration_noise) if isinstance(add_exploration_noise, Sequence) else []
            calls.append((num_sims, noise))
            self.last_actions = [action] * len(boards)
            self.last_search_contempt_stats = [SearchContemptStats() for _ in boards]
            policy = np.zeros(self.game.get_action_size(), dtype=np.float32)
            policy[action] = 1.0
            return [
                (policy.copy(), 0.0, self.game.to_array(board), self.game.get_valid_moves(board, 1)) for board in boards
            ]

    coach = Coach(
        chess_game,
        LunaNetwork(chess_game, _pcr_learner(small_learner_config)),
        _pcr_run(parallel_games=2),
    )
    monkeypatch.setattr("luna.coach_batched_self_play.BatchedMCTS", _BatchedSearch)
    monkeypatch.setattr("luna.coach_self_play.np.random.random", lambda: 0.9)

    trajectories = coach.execute_episodes_batched(2, progress=False)

    assert calls == [(2, [False, False])]
    assert [trajectory.policy_train_mask.tolist() for trajectory in trajectories] == [[False], [False]]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"playout_cap_full_sims": 8}, "PCR requires positive"),
        (
            {
                "playout_cap_full_sims": 8,
                "playout_cap_fast_sims": 8,
                "playout_cap_full_probability": 0.25,
            },
            "must be smaller",
        ),
        ({"playout_cap_full_probability": 1.1}, "must be between 0 and 1"),
    ],
)
def test_invalid_pcr_configuration_fails_closed(
    overrides: dict[str, int | float],
    message: str,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_training_configuration(TrainingRunConfig(**overrides), small_learner_config)


def test_pcr_rejects_policy_reanalysis(small_learner_config: EzV2LearnerConfig) -> None:
    small_learner_config.reanalyze_policy = True

    with pytest.raises(ValueError, match="policy reanalysis"):
        validate_training_configuration(_pcr_run(), small_learner_config)
