"""Shared test fixtures for ChessRL test suite."""

import numpy as np
import pytest

from luna.config import EzV2LearnerConfig
from luna.game.chess_game import ChessGame
from luna.replay_buffer import Trajectory


@pytest.fixture
def small_learner_config():
    """Minimal model config for fast CPU tests."""
    return EzV2LearnerConfig(
        device="cpu",
        num_channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        proj_dim=32,
        compile_inference=False,
    )


@pytest.fixture
def chess_game():
    """ChessGame instance."""
    return ChessGame()


@pytest.fixture
def make_trajectory():
    """Factory for creating test trajectories."""
    def _make(length: int = 10) -> Trajectory:
        game = ChessGame()
        observations = [np.random.randn(*game.get_board_size()) for _ in range(length)]
        policies = [np.random.rand(game.get_action_size()) for _ in range(length)]
        rewards = [0.0] * length
        return Trajectory(observations, policies, rewards, 1.0)
    return _make
