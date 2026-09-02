"""Regression tests for EfficientZeroV2 training loop."""

import math
from collections.abc import Sequence
from dataclasses import replace
from unittest.mock import patch

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


def _parameter_gradients(network: LunaNetwork) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in network.nnet.named_parameters()
        if parameter.grad is not None
    }


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


def test_pcr_policy_mask_has_identical_gradients_and_update_with_accumulation(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    config = replace(
        small_learner_config,
        batch_size=4,
        grad_accum_steps=1,
        unroll_steps=1,
        td_steps=1,
        lr=1e-3,
        lr_min=1e-3,
        lr_warmup_steps=0,
        weight_decay=0.0,
        grad_clip_norm=1e6,
        mixed_precision=False,
        dataloader_workers=0,
        policy_loss_weight=1.0,
        value_loss_weight=0.0,
        reward_loss_weight=0.0,
        consistency_loss_weight=0.0,
        reconstruction_loss_weight=0.0,
    )
    full_batch = LunaNetwork(ChessGame(), config)
    accumulated = LunaNetwork(ChessGame(), replace(config, grad_accum_steps=2))
    accumulated.nnet.load_state_dict(full_batch.nnet.state_dict())
    replay = PrioritizedReplayBuffer(capacity=1)
    replay.save_trajectory(_make_trajectory(length=2))
    sampled = full_batch._prepare_batch(replay, 4, 1, 1, 1.0, 1, None)
    collated = {name: values.copy() for name, values in sampled.collated.items()}
    collated["policy_mask"][:] = 0.0
    collated["policy_mask"][:2, 0] = 1.0
    collated["value_mask"][:] = 0.0
    collated["unroll_mask"][:] = 0.0
    collated["consistency_mask"][:] = 0.0
    collated["target_policies"][:] = 0.0
    for row, action in enumerate((0, 17, 65, 130)):
        collated["target_policies"][row, :, action] = 1.0
        collated["observations"][row, :, :, row] = float(row + 1)
    prepared = sampled._replace(
        collated=collated,
        is_weights=np.asarray((1.0, 0.5, 0.25, 0.25), dtype=np.float32),
    )

    def train_once(network: LunaNetwork) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        gradients: dict[str, torch.Tensor] = {}
        original_step = network.optimizer.step

        def capture_step(*args: object, **kwargs: object) -> object:
            gradients.update(_parameter_gradients(network))
            return original_step(*args, **kwargs)

        with (
            patch.object(network, "_prepare_batch", return_value=prepared),
            patch.object(network.optimizer, "step", side_effect=capture_step),
        ):
            network.train_ezv2(replay, steps=1, total_train_steps=1)
        return gradients, network.nnet.state_dict()

    full_gradients, full_state = train_once(full_batch)
    accumulated_gradients, accumulated_state = train_once(accumulated)

    assert full_gradients.keys() == accumulated_gradients.keys()
    for name in full_gradients:
        torch.testing.assert_close(accumulated_gradients[name], full_gradients[name], rtol=1e-5, atol=1e-7)
    for name in full_state:
        torch.testing.assert_close(accumulated_state[name], full_state[name], rtol=1e-5, atol=1e-7)


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

    prepared = nnet._prepare_batch(
        replay,
        bs=1,
        unroll=0,
        td=5,
        discount=1.0,
        training_step=0,
        mcts_for_reanalyze=TrainingRunConfig(num_mcts_sims=2),
    )
    collated = prepared.collated

    assert nnet.nnet.training
    assert collated["target_values"][0, 0] == pytest.approx(fresh_value)
