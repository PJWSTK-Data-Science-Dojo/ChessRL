"""Shared test fixtures for ChessRL test suite."""

from typing import Protocol

import chess
import numpy as np
import pytest

from luna.config import EzV2LearnerConfig
from luna.game.chess_game import ChessGame
from luna.replay_buffer import Trajectory


class TrajectoryFactory(Protocol):
    def __call__(
        self,
        length: int = 10,
        *,
        truncated: bool = False,
        termination: chess.Termination | None = None,
    ) -> Trajectory: ...


@pytest.fixture
def small_learner_config() -> EzV2LearnerConfig:
    return EzV2LearnerConfig(
        device="cpu",
        num_channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        proj_dim=32,
        compile_inference=False,
    )


@pytest.fixture
def chess_game() -> ChessGame:
    return ChessGame()


@pytest.fixture
def make_trajectory() -> TrajectoryFactory:
    def _make(
        length: int = 10,
        *,
        truncated: bool = False,
        termination: chess.Termination | None = None,
    ) -> Trajectory:
        game = ChessGame()
        action_size = game.get_action_size()
        observations = [np.random.randn(*game.get_board_size()).astype(np.float32) for _ in range(length)]
        policies = [np.full(action_size, 1.0 / action_size, dtype=np.float32) for _ in range(length)]
        rewards = [0.0] * length
        return Trajectory(
            observations=observations,
            actions=np.zeros(length, dtype=np.int64),
            rewards=rewards,
            root_policies=policies,
            root_values=np.zeros(length, dtype=np.float32),
            valids=np.ones((length, action_size), dtype=np.float32),
            truncated=truncated,
            termination=termination,
        )

    return _make
