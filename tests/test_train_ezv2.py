"""Regression tests for EfficientZeroV2 training loop."""

import numpy as np
import pytest
import torch

from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.game.chess_game import ACTION_SIZE, OBS_PLANES, ChessGame
from luna.network import LunaNetwork
from luna.replay_buffer import PrioritizedReplayBuffer, Trajectory


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_train_ezv2_increments_global_step_per_optimizer_step() -> None:
    game = ChessGame()
    learner = EzV2LearnerConfig(
        batch_size=2,
        grad_accum_steps=2,
        num_channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        proj_dim=32,
        lr=3e-3,
    )
    nnet = LunaNetwork(game, learner)
    replay = PrioritizedReplayBuffer(capacity=500)

    def _traj(length: int = 12) -> Trajectory:
        return Trajectory(
            observations=[np.random.randn(8, 8, OBS_PLANES).astype(np.float32) for _ in range(length)],
            actions=[np.random.randint(0, min(256, ACTION_SIZE)) for _ in range(length)],
            rewards=np.zeros(length, dtype=np.float32),
            root_policies=[np.ones(ACTION_SIZE, dtype=np.float32) / ACTION_SIZE for _ in range(length)],
            root_values=np.zeros(length, dtype=np.float32),
            valids=[np.ones(ACTION_SIZE, dtype=np.float32) for _ in range(length)],
        )

    for _ in range(8):
        replay.save_trajectory(_traj())

    g0 = int(nnet._global_step)
    run_params = TrainingRunConfig(num_mcts_sims=2)
    nnet.train_ezv2(
        replay,
        steps=4,
        start_step=0,
        discount=0.997,
        mcts_for_reanalyze=run_params,
    )
    assert nnet._global_step == g0 + 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reanalyze_disables_async_prefetch_paths() -> None:
    game = ChessGame()
    learner = EzV2LearnerConfig(
        reanalyze_mcts_sims=2,
        reanalyze_prob=1.0,
        mixed_value_td_until_step=0,
        batch_size=2,
        num_channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        proj_dim=32,
        lr=1e-3,
    )
    nnet = LunaNetwork(game, learner)
    assert not nnet._async_batch_prefetch()
