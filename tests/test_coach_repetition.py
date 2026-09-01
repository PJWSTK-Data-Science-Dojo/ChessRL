"""Tests for Coach self-play (e.g. max ply truncation, batched self-play)."""

from collections.abc import Sequence
from unittest.mock import patch

import chess
import numpy as np
import pytest

from luna.coach import (
    Coach,
)
from luna.config import EzV2LearnerConfig, MCTSParams, TrainingRunConfig
from luna.game.chess_game import ChessGame, move_to_action
from luna.network import LunaNetwork
from luna.profiling import IterProfileStats, SelfPlayMCTSTimings


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

    monkeypatch.setattr("luna.coach_self_play.MCTS", _Search)
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

    with patch("luna.coach_metrics.wandb.run", object()), patch("luna.coach_metrics.wandb.log") as wandb_log:
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
            for root, selected_action in zip(boards, selected_actions, strict=True):
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

    monkeypatch.setattr("luna.coach_batched_self_play.BatchedMCTS", _BatchedSearch)
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
