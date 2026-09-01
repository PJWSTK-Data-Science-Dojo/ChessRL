"""Regression tests for EfficientZeroV2 training loop."""

import math
from collections.abc import Sequence

import chess
import numpy as np
import pytest
import torch
from torch.amp import GradScaler

from luna.config import EzV2LearnerConfig, MCTSParams, TrainingRunConfig
from luna.game.chess_game import ACTION_SIZE, OBS_PLANES, ChessGame
from luna.network import (
    LunaNetwork,
)
from luna.replay_buffer import PrioritizedReplayBuffer, Trajectory


def _make_trajectory(length: int = 4) -> Trajectory:
    return Trajectory(
        observations=[np.random.randn(8, 8, OBS_PLANES).astype(np.float32) for _ in range(length)],
        actions=[np.random.randint(0, min(256, ACTION_SIZE)) for _ in range(length)],
        rewards=np.zeros(length, dtype=np.float32),
        root_policies=[np.full(ACTION_SIZE, 1.0 / ACTION_SIZE, dtype=np.float32) for _ in range(length)],
        root_values=np.zeros(length, dtype=np.float32),
        valids=np.ones((length, ACTION_SIZE), dtype=np.float32),
    )


def test_non_finite_gradient_fails_before_parameter_update(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.batch_size = 1
    small_learner_config.unroll_steps = 1
    small_learner_config.td_steps = 1
    small_learner_config.mixed_precision = False
    small_learner_config.dataloader_workers = 0
    network = LunaNetwork(ChessGame(), small_learner_config)
    replay = PrioritizedReplayBuffer(capacity=2)
    replay.save_trajectory(_make_trajectory(length=1))
    parameter = next(network.nnet.parameters())
    original = parameter.detach().clone()

    def inject_non_finite(gradient: torch.Tensor) -> torch.Tensor:
        return torch.full_like(gradient, float("inf"))

    hook = parameter.register_hook(inject_non_finite)
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            network.train_ezv2(replay, steps=1)
    finally:
        hook.remove()

    torch.testing.assert_close(parameter, original)


def test_grad_scaler_retries_transient_overflow_without_counting_skipped_update(
    monkeypatch: pytest.MonkeyPatch,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.batch_size = 1
    small_learner_config.unroll_steps = 1
    small_learner_config.td_steps = 1
    small_learner_config.mixed_precision = False
    small_learner_config.dataloader_workers = 1
    small_learner_config.lr = 1e-3
    small_learner_config.lr_min = 1e-3
    network = LunaNetwork(ChessGame(), small_learner_config)
    network.scaler = GradScaler("cpu", init_scale=8.0, growth_interval=1_000, enabled=True)
    replay = PrioritizedReplayBuffer(capacity=2)
    replay.save_trajectory(_make_trajectory(length=1))
    sample_calls = 0
    priority_update_calls = 0
    original_sample = replay.sample
    original_update_priorities = replay.update_priorities

    def record_sample(
        batch_size: int,
        unroll_steps: int,
    ) -> tuple[list[tuple[Trajectory, int]], np.ndarray, list[int]]:
        nonlocal sample_calls
        sample_calls += 1
        return original_sample(batch_size, unroll_steps)

    def record_priority_update(indices: list[int], td_errors: np.ndarray) -> None:
        nonlocal priority_update_calls
        priority_update_calls += 1
        original_update_priorities(indices, td_errors)

    monkeypatch.setattr(replay, "sample", record_sample)
    monkeypatch.setattr(replay, "update_priorities", record_priority_update)
    parameter = next(network.nnet.parameters())
    gradient_calls = 0

    def inject_one_overflow(gradient: torch.Tensor) -> torch.Tensor:
        nonlocal gradient_calls
        gradient_calls += 1
        if gradient_calls == 1:
            return torch.full_like(gradient, float("inf"))
        return gradient

    hook = parameter.register_hook(inject_one_overflow)
    try:
        metrics = network.train_ezv2(replay, steps=1)
    finally:
        hook.remove()

    assert gradient_calls == 2
    assert sample_calls == 1
    assert priority_update_calls == 1
    assert network._global_step == 1
    assert network.scaler.get_scale() == pytest.approx(4.0)
    assert all(math.isfinite(value) for value in metrics.values())


def test_grad_scaler_stops_after_bounded_consecutive_overflows(
    monkeypatch: pytest.MonkeyPatch,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.batch_size = 1
    small_learner_config.unroll_steps = 1
    small_learner_config.td_steps = 1
    small_learner_config.mixed_precision = False
    small_learner_config.dataloader_workers = 0
    network = LunaNetwork(ChessGame(), small_learner_config)
    network.scaler = GradScaler("cpu", init_scale=8.0, growth_interval=1_000, enabled=True)
    replay = PrioritizedReplayBuffer(capacity=2)
    replay.save_trajectory(_make_trajectory(length=1))
    parameter = next(network.nnet.parameters())
    original = parameter.detach().clone()

    def inject_overflow(gradient: torch.Tensor) -> torch.Tensor:
        return torch.full_like(gradient, float("inf"))

    def reject_priority_update(_indices: list[int], _td_errors: np.ndarray) -> None:
        raise AssertionError("A skipped optimizer update must not change replay priorities")

    monkeypatch.setattr("luna.network._MAX_CONSECUTIVE_AMP_SKIPS", 2)
    monkeypatch.setattr(replay, "update_priorities", reject_priority_update)
    hook = parameter.register_hook(inject_overflow)
    try:
        with pytest.raises(RuntimeError, match="2 consecutive non-finite gradient updates"):
            network.train_ezv2(replay, steps=1)
    finally:
        hook.remove()

    assert network._global_step == 0
    torch.testing.assert_close(parameter, original)


def test_finite_gradient_norm_overflow_fails_before_optimizer_mutation(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.batch_size = 1
    small_learner_config.unroll_steps = 1
    small_learner_config.td_steps = 1
    small_learner_config.mixed_precision = False
    small_learner_config.dataloader_workers = 0
    network = LunaNetwork(ChessGame(), small_learner_config)
    network.scaler = GradScaler("cpu", init_scale=8.0, enabled=True)
    replay = PrioritizedReplayBuffer(capacity=2)
    replay.save_trajectory(_make_trajectory(length=1))
    parameter = next(parameter for parameter in network.nnet.parameters() if parameter.numel() >= 1_000)
    original_parameters = [item.detach().clone() for item in network.nnet.parameters()]
    original_lr = network.optimizer.param_groups[0]["lr"]

    def inject_large_finite_gradient(gradient: torch.Tensor) -> torch.Tensor:
        return torch.full_like(gradient, 1e30)

    hook = parameter.register_hook(inject_large_finite_gradient)
    try:
        with pytest.raises(RuntimeError, match="Gradient norm overflowed despite finite gradient elements"):
            network.train_ezv2(replay, steps=1)
    finally:
        hook.remove()

    assert network._global_step == 0
    assert not network.optimizer.state
    assert network.optimizer.param_groups[0]["lr"] == original_lr
    for current, original_item in zip(network.nnet.parameters(), original_parameters, strict=True):
        torch.testing.assert_close(current, original_item)


def test_reanalysis_restores_training_mode_and_uses_direct_sve(
    monkeypatch: pytest.MonkeyPatch,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    game = ChessGame()
    small_learner_config.batch_size = 1
    small_learner_config.reanalyze_mcts_sims = 2
    small_learner_config.reanalyze_prob = 1.0
    small_learner_config.reanalyze_policy = True
    small_learner_config.reanalyze_start_step = 0
    nnet = LunaNetwork(game, small_learner_config)
    replay = PrioritizedReplayBuffer(capacity=4)
    replay.save_trajectory(_make_trajectory(length=1))

    fresh_value = 0.375

    class _FakeBatchedMCTS:
        def __init__(self, _game: ChessGame, network: LunaNetwork, _params: MCTSParams) -> None:
            self.network = network

        def search_batch(
            self,
            boards: list[chess.Board],
            temp: float,
            *,
            add_exploration_noise: bool | Sequence[bool] | None,
        ) -> list[tuple[np.ndarray, float, None, None]]:
            assert temp == 1.0
            assert add_exploration_noise is False
            self.network.nnet.eval()
            policy = np.full(ACTION_SIZE, 1.0 / ACTION_SIZE, dtype=np.float32)
            return [(policy, fresh_value, None, None) for _ in boards]

    monkeypatch.setattr("luna.network.BatchedMCTS", _FakeBatchedMCTS)
    nnet.nnet.train()

    collated, _weights, _indices, _reanalysis = nnet._prepare_batch(
        replay,
        bs=1,
        unroll=0,
        td=5,
        discount=1.0,
        training_step=0,
        mcts_for_reanalyze=TrainingRunConfig(num_mcts_sims=2),
    )

    assert nnet.nnet.training
    assert collated["target_values"][0, 0] == pytest.approx(fresh_value)
