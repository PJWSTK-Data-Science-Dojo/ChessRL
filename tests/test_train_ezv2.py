"""Regression tests for EfficientZeroV2 training loop."""

import math
from collections.abc import Sequence
from dataclasses import replace
from hashlib import file_digest
from pathlib import Path
from unittest.mock import patch

import chess
import numpy as np
import pytest
import torch
from torch._inductor import config as torch_inductor_config
from torch.amp import GradScaler

from luna.balanced_networks import BalancedNetworks
from luna.config import EzV2LearnerConfig, MCTSParams, TrainingRunConfig
from luna.ezv2_networks import SimSiamProjector
from luna.game.chess_game import ACTION_SIZE, OBS_PLANES, ChessGame
from luna.network import (
    LunaNetwork,
    PreparedBatch,
    TrainingPhaseProvenance,
    _configure_dynamic_cudagraphs,
    _latent_health_metrics,
    _scale_gradient,
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


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.0])
def test_scale_gradient_preserves_forward_and_scales_backward(scale: float) -> None:
    source = torch.tensor([1.5, -2.0], requires_grad=True)
    upstream = torch.tensor([3.0, -4.0])

    scaled = _scale_gradient(source, scale)
    torch.testing.assert_close(scaled, source)
    (scaled * upstream).sum().backward()

    assert source.grad is not None
    torch.testing.assert_close(source.grad, upstream * scale)


def test_dynamic_cudagraphs_are_skipped_for_compiled_mcts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_inductor_config.triton, "cudagraph_skip_dynamic_graphs", False)

    _configure_dynamic_cudagraphs()

    assert torch_inductor_config.triton.cudagraph_skip_dynamic_graphs is True


def test_latent_health_metrics_detect_collapsed_batch() -> None:
    projector = SimSiamProjector(in_dim=2, proj_dim=2)
    projector.projection = torch.nn.Identity()
    collapsed = torch.tensor([1.0, 0.0]).reshape(1, 2, 1, 1).repeat(4, 1, 1, 1)

    metrics = _latent_health_metrics(projector, collapsed, collapsed)

    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["train/latent_predicted_batch_feature_std"] == pytest.approx(0.0)
    assert metrics["train/latent_target_batch_feature_std"] == pytest.approx(0.0)
    assert metrics["train/projector_target_batch_std"] == pytest.approx(0.0)
    assert metrics["train/projector_target_offdiag_cosine"] == pytest.approx(1.0)
    assert metrics["train/consistency_cosine_alignment"] == pytest.approx(1.0)


def test_latent_health_metrics_distinguish_diverse_batch() -> None:
    projector = SimSiamProjector(in_dim=2, proj_dim=2)
    projector.projection = torch.nn.Identity()
    diverse = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    ).reshape(4, 2, 1, 1)

    metrics = _latent_health_metrics(projector, diverse, diverse)

    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["train/latent_target_batch_feature_std"] > 0.5
    assert metrics["train/projector_target_batch_std"] > 0.5
    assert metrics["train/projector_target_offdiag_cosine"] == pytest.approx(1.0 / 3.0)
    assert metrics["train/consistency_cosine_alignment"] == pytest.approx(1.0)


def test_latent_health_metrics_are_finite_for_single_sample() -> None:
    projector = SimSiamProjector(in_dim=2, proj_dim=2)
    projector.projection = torch.nn.Identity()
    sample = torch.tensor([1.0, -1.0]).reshape(1, 2, 1, 1)

    metrics = _latent_health_metrics(projector, sample, sample)

    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["train/latent_predicted_batch_feature_std"] == pytest.approx(0.0)
    assert metrics["train/latent_target_batch_feature_std"] == pytest.approx(0.0)
    assert metrics["train/projector_target_batch_std"] == pytest.approx(0.0)
    assert metrics["train/projector_target_offdiag_cosine"] == pytest.approx(0.0)


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

    for _ in range(8):
        replay.save_trajectory(_make_trajectory(length=12))

    g0 = int(nnet._global_step)
    run_params = TrainingRunConfig(num_mcts_sims=2)
    nnet.train_ezv2(
        replay,
        steps=4,
        discount=0.997,
        mcts_for_reanalyze=run_params,
    )
    assert nnet._global_step == g0 + 4


def test_balanced_model_completes_unrolled_optimizer_step(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    config = replace(
        small_learner_config,
        model_name="balanced",
        batch_size=1,
        dataloader_workers=0,
        mixed_precision=False,
        td_steps=1,
        unroll_steps=1,
    )
    network = LunaNetwork(ChessGame(), config)
    replay = PrioritizedReplayBuffer(capacity=8)
    replay.save_trajectory(_make_trajectory(length=4))

    metrics = network.train_ezv2(replay, steps=1, total_train_steps=1)

    assert network._global_step == 1
    assert math.isfinite(metrics["total"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reanalyze_disables_async_prefetch_paths() -> None:
    game = ChessGame()
    learner = EzV2LearnerConfig(
        reanalyze_mcts_sims=2,
        reanalyze_prob=1.0,
        reanalyze_start_step=0,
        batch_size=2,
        num_channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        proj_dim=32,
        lr=1e-3,
    )
    nnet = LunaNetwork(game, learner)
    assert not nnet._async_batch_prefetch()


def test_reanalysis_warmup_keeps_plain_replay_prefetch_enabled(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.reanalyze_mcts_sims = 2
    small_learner_config.reanalyze_prob = 1.0
    small_learner_config.reanalyze_start_step = 10
    network = LunaNetwork(ChessGame(), small_learner_config)

    assert network._async_batch_prefetch(upcoming_steps=9)
    assert not network._async_batch_prefetch(upcoming_steps=10)


def test_zero_replay_workers_disable_background_prefetch(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.dataloader_workers = 0
    network = LunaNetwork(ChessGame(), small_learner_config)

    assert network._prefetch_executor is None
    assert not network._async_batch_prefetch()


def test_recurrent_inference_copies_only_legal_policy_candidates(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(ChessGame(), small_learner_config)
    latent = torch.zeros(1, small_learner_config.num_channels, 8, 8, device=network.device)
    legal_mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    legal_mask[[12, 34]] = 1.0

    result = network.batched_recurrent_inference(
        latent,
        [12],
        valid_masks=[legal_mask],
        policy_topk=256,
    )

    assert result.policy_full is None
    assert result.topk_indices is not None
    assert result.topk_probs is not None
    assert result.topk_indices.shape == (1, 2)
    assert set(result.topk_indices[0]) == {12, 34}
    assert float(result.topk_probs.sum()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("model_name", "unknown", "model_name must be 'baseline' or 'balanced'"),
        ("grad_accum_steps", 0, "grad_accum_steps must be a positive integer"),
        ("dataloader_workers", -1, "dataloader_workers must be a non-negative integer"),
        ("support_size", 0, "support_size must be a positive integer"),
        ("grad_clip_norm", 0.0, "grad_clip_norm must be positive and finite"),
        ("grad_clip_norm", float("inf"), "grad_clip_norm must be positive and finite"),
        ("lr", float("nan"), "lr must be finite"),
        ("reanalyze_prob", 1.1, "reanalyze_prob must be between 0 and 1"),
        ("consistency_loss_weight", float("inf"), "consistency_loss_weight must be finite"),
    ],
)
def test_invalid_execution_settings_fail_loudly(
    small_learner_config: EzV2LearnerConfig,
    field_name: str,
    value: object,
    message: str,
) -> None:
    setattr(small_learner_config, field_name, value)

    with pytest.raises(ValueError, match=message):
        LunaNetwork(ChessGame(), small_learner_config)


def test_learner_rejects_zero_unroll_horizon(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.unroll_steps = 0

    with pytest.raises(ValueError, match="unroll_steps must be a positive integer"):
        LunaNetwork(ChessGame(), small_learner_config)


def test_learner_requires_equal_accumulation_microbatches(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.batch_size = 3
    small_learner_config.grad_accum_steps = 2

    with pytest.raises(ValueError, match="batch_size must be divisible by grad_accum_steps"):
        LunaNetwork(ChessGame(), small_learner_config)


def test_reported_total_loss_is_invariant_to_identical_gradient_accumulation() -> None:
    np.random.seed(23)
    trajectory = _make_trajectory(length=1)

    def train_once(grad_accum_steps: int) -> float:
        torch.manual_seed(7)
        learner = EzV2LearnerConfig(
            device="cpu",
            batch_size=2,
            unroll_steps=1,
            td_steps=1,
            num_channels=16,
            repr_blocks=1,
            dyn_blocks=1,
            proj_dim=32,
            lr=0.0,
            lr_min=0.0,
            mixed_precision=False,
            grad_accum_steps=grad_accum_steps,
            dataloader_workers=0,
        )
        network = LunaNetwork(ChessGame(), learner)
        replay = PrioritizedReplayBuffer(capacity=2)
        replay.save_trajectory(trajectory)
        return network.train_ezv2(replay, steps=1)["total"]

    single_microbatch = train_once(1)
    accumulated_microbatches = train_once(2)

    assert accumulated_microbatches == pytest.approx(single_microbatch, rel=1e-5)


def test_reward_and_consistency_losses_use_their_own_active_masks(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    config = replace(
        small_learner_config,
        batch_size=2,
        unroll_steps=3,
        td_steps=1,
        mixed_precision=False,
        dataloader_workers=0,
        lr=0.0,
        lr_min=0.0,
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        reward_loss_weight=1.0,
        consistency_loss_weight=1.0,
    )
    network = LunaNetwork(ChessGame(), config)
    replay = PrioritizedReplayBuffer(capacity=8)
    replay.save_trajectory(_make_trajectory(length=4))
    prepared = network._prepare_batch(
        replay,
        bs=2,
        unroll=3,
        td=1,
        discount=1.0,
        training_step=1,
        mcts_for_reanalyze=None,
    )
    prepared.collated["unroll_mask"][:] = np.array(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    prepared.collated["consistency_mask"][:] = np.array(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    controlled = PreparedBatch(
        prepared.collated,
        np.ones(2, dtype=np.float32),
        prepared.tree_indices,
        prepared.reanalysis,
    )

    def constant_support_loss(logits: torch.Tensor, _targets: torch.Tensor) -> torch.Tensor:
        return logits.sum(dim=1) * 0.0 + 2.0

    def constant_consistency_loss(
        _projector: SimSiamProjector,
        predicted: torch.Tensor,
        _target: torch.Tensor,
    ) -> torch.Tensor:
        return predicted.flatten(1).sum(dim=1) * 0.0 + 3.0

    with (
        patch.object(network, "_prepare_batch", return_value=controlled),
        patch("luna.network._soft_ce_with_support", side_effect=constant_support_loss),
        patch("luna.network._simsiam_loss", side_effect=constant_consistency_loss),
    ):
        metrics = network.train_ezv2(replay, steps=1)

    assert metrics["reward"] == pytest.approx(2.0)
    assert metrics["consistency"] == pytest.approx(3.0)
    assert metrics["total"] == pytest.approx(5.0)


def test_wandb_reports_gradient_clipping_health_metrics(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    config = replace(
        small_learner_config,
        batch_size=1,
        unroll_steps=1,
        td_steps=1,
        mixed_precision=False,
        dataloader_workers=0,
        lr=0.0,
        lr_min=0.0,
        grad_clip_norm=2.0,
    )
    network = LunaNetwork(ChessGame(), config)
    replay = PrioritizedReplayBuffer(capacity=2)
    replay.save_trajectory(_make_trajectory(length=2))

    with (
        patch("luna.network.torch.nn.utils.clip_grad_norm_", return_value=torch.tensor(10.0)),
        patch("luna.network.wandb.run", object()),
        patch("luna.network.wandb.log") as wandb_log,
    ):
        network.train_ezv2(replay, steps=1)

    wandb_log.assert_called_once()
    metrics = wandb_log.call_args.args[0]
    assert metrics["train/grad_norm"] == pytest.approx(10.0)
    assert metrics["train/grad_norm_preclip"] == pytest.approx(10.0)
    assert metrics["train/grad_norm_postclip"] == pytest.approx(2.0)
    assert metrics["train/grad_clip_coefficient"] == pytest.approx(0.2)
    assert metrics["train/grad_clip_fraction"] == pytest.approx(1.0)


def test_accumulation_samples_one_configured_batch_per_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.batch_size = 4
    small_learner_config.grad_accum_steps = 2
    small_learner_config.unroll_steps = 1
    small_learner_config.td_steps = 1
    small_learner_config.mixed_precision = False
    small_learner_config.dataloader_workers = 0
    small_learner_config.lr = 0.0
    small_learner_config.lr_min = 0.0
    network = LunaNetwork(ChessGame(), small_learner_config)
    replay = PrioritizedReplayBuffer(capacity=4)
    replay.save_trajectory(_make_trajectory(length=1))
    requested_batch_sizes: list[int] = []
    original_sample = replay.sample

    def record_sample(
        batch_size: int,
        unroll_steps: int,
    ) -> tuple[list[tuple[Trajectory, int]], np.ndarray, list[int]]:
        requested_batch_sizes.append(batch_size)
        return original_sample(batch_size, unroll_steps)

    monkeypatch.setattr(replay, "sample", record_sample)

    network.train_ezv2(replay, steps=2)

    assert requested_batch_sizes == [4, 4]


def test_accumulation_updates_duplicate_replay_index_atomically_with_largest_error(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.batch_size = 2
    small_learner_config.grad_accum_steps = 2
    small_learner_config.unroll_steps = 1
    small_learner_config.td_steps = 1
    small_learner_config.mixed_precision = False
    small_learner_config.dataloader_workers = 0
    small_learner_config.lr = 0.0
    small_learner_config.lr_min = 0.0
    network = LunaNetwork(ChessGame(), small_learner_config)
    replay = PrioritizedReplayBuffer(capacity=1)
    replay.save_trajectory(_make_trajectory(length=1))
    collated, weights, _indices, reanalysis = network._prepare_batch(
        replay,
        bs=2,
        unroll=1,
        td=1,
        discount=1.0,
        training_step=1,
        mcts_for_reanalyze=None,
    )
    collated["target_values"][:, 0] = np.array([-1.0, 1.0], dtype=np.float32)
    prepared = PreparedBatch(collated, weights, [0, 0], reanalysis)

    with (
        patch.object(network, "_prepare_batch", return_value=prepared),
        patch.object(replay, "update_priorities", wraps=replay.update_priorities) as update_priorities,
    ):
        network.train_ezv2(replay, steps=1)

    update_priorities.assert_called_once()
    indices, errors = update_priorities.call_args.args
    assert indices == [0, 0]
    expected_raw_priority = float(np.max(errors)) + 1e-6
    assert replay._tree.tree[replay.capacity] == pytest.approx(expected_raw_priority**replay.alpha)


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
    for current, original_item in zip(network.nnet.parameters(), original_parameters):
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


def test_checkpoint_contains_architecture_metadata(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.recurrent_gradient_scale = 0.4
    nnet = LunaNetwork(chess_game, small_learner_config)
    nnet._global_step = 17
    nnet._trainer_iteration = 6
    nnet._lr_schedule_total_steps = 80
    nnet.scaler = GradScaler("cpu", init_scale=1024.0, growth_interval=123, enabled=True)
    expected_scaler_state = nnet.scaler.state_dict()
    nnet.save_checkpoint(str(tmp_path), "metadata.pth.tar")

    checkpoint = torch.load(
        tmp_path / "metadata.pth.tar",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["format_version"] == 2
    assert checkpoint["global_step"] == 17
    assert checkpoint["trainer_iteration"] == 6
    assert checkpoint["lr_schedule_total_steps"] == 80
    assert checkpoint["scaler"] == expected_scaler_state
    assert checkpoint["model_spec"]["action_size"] == chess_game.get_action_size()
    assert checkpoint["model_spec"]["observation_shape"] == list(chess_game.get_board_size())
    assert checkpoint["learner_config"]["num_channels"] == small_learner_config.num_channels
    assert checkpoint["model_spec"]["model_name"] == small_learner_config.model_name
    assert checkpoint["learner_config"]["recurrent_gradient_scale"] == pytest.approx(0.4)

    restored = LunaNetwork.from_checkpoint(
        chess_game,
        tmp_path / "metadata.pth.tar",
        device="cpu",
    )
    assert restored._global_step == 17
    assert restored._trainer_iteration == 6
    assert restored._lr_schedule_total_steps == 80
    assert restored._learner.num_channels == small_learner_config.num_channels
    assert restored._learner.recurrent_gradient_scale == pytest.approx(0.4)

    resumed = LunaNetwork(chess_game, small_learner_config)
    resumed.scaler = GradScaler("cpu", init_scale=32.0, growth_interval=7, enabled=True)
    resumed.load_checkpoint(str(tmp_path), "metadata.pth.tar", load_optimizer=True)
    assert resumed.scaler.state_dict() == expected_scaler_state
    assert resumed._lr_schedule_total_steps == 80


def test_checkpoint_reconstructs_balanced_model_from_factory_metadata(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    config = replace(small_learner_config, model_name="balanced")
    network = LunaNetwork(chess_game, config)
    network.save_checkpoint(str(tmp_path), "balanced.pth.tar")

    restored = LunaNetwork.from_checkpoint(chess_game, tmp_path / "balanced.pth.tar", device="cpu")

    assert restored._learner.model_name == "balanced"
    assert isinstance(restored.nnet, BalancedNetworks)
    for name, tensor in network.nnet.state_dict().items():
        torch.testing.assert_close(tensor, restored.nnet.state_dict()[name])


def test_checkpoint_without_model_factory_metadata_defaults_to_baseline(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    network.save_checkpoint(str(tmp_path), "current.pth.tar")
    checkpoint = torch.load(tmp_path / "current.pth.tar", map_location="cpu", weights_only=True)
    del checkpoint["model_spec"]["model_name"]
    torch.save(checkpoint, tmp_path / "pre-factory.pth.tar")

    restored = LunaNetwork.from_checkpoint(chess_game, tmp_path / "pre-factory.pth.tar", device="cpu")

    assert restored._learner.model_name == "baseline"
    assert type(restored.nnet) is type(network.nnet)


def test_learning_rate_continues_from_checkpoint_global_step(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.batch_size = 2
    small_learner_config.unroll_steps = 1
    small_learner_config.td_steps = 1
    small_learner_config.mixed_precision = False
    small_learner_config.lr = 1e-3
    small_learner_config.lr_min = 1e-5
    small_learner_config.lr_warmup_steps = 4

    nnet = LunaNetwork(chess_game, small_learner_config)
    nnet._global_step = 7
    nnet._lr_schedule_total_steps = 20
    nnet.save_checkpoint(str(tmp_path), "resume.pth.tar")

    restored = LunaNetwork(chess_game, small_learner_config)
    restored.load_checkpoint(str(tmp_path), "resume.pth.tar", load_optimizer=False)
    assert restored._lr_schedule_total_steps == 20
    replay = PrioritizedReplayBuffer(capacity=8)
    replay.save_trajectory(_make_trajectory(length=4))
    expected_lr = restored._lr_schedule(step_in_run=8, total_steps=20)

    restored.train_ezv2(replay, steps=1, total_train_steps=40)

    assert restored._global_step == 8
    assert restored._lr_schedule_total_steps == 20
    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)


def test_new_training_phase_loads_only_weights_and_resets_progress(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    source = LunaNetwork(chess_game, small_learner_config)
    source_parameter = next(source.nnet.parameters())
    with torch.no_grad():
        source_parameter.add_(0.25)
    source_parameter.grad = torch.ones_like(source_parameter)
    source.optimizer.step()
    source.optimizer.zero_grad(set_to_none=True)
    source._global_step = 123
    source._trainer_iteration = 17
    source._lr_schedule_total_steps = 500
    source.save_checkpoint(str(tmp_path), "source.pth.tar")
    source_path = tmp_path / "source.pth.tar"
    with source_path.open("rb") as source_file:
        source_sha256 = file_digest(source_file, "sha256").hexdigest()

    phase_config = replace(
        small_learner_config,
        batch_size=64,
        grad_accum_steps=2,
        lr=7e-4,
        lr_min=2e-5,
        weight_decay=3e-4,
    )
    phase = LunaNetwork(chess_game, phase_config)
    phase_parameter = next(phase.nnet.parameters())
    phase_parameter.grad = torch.ones_like(phase_parameter)
    phase.optimizer.step()
    phase.optimizer.zero_grad(set_to_none=True)
    phase._global_step = 9
    phase._trainer_iteration = 4
    phase._lr_schedule_total_steps = 80

    phase.initialize_training_phase(str(tmp_path), "source.pth.tar")

    for name, tensor in phase.nnet.state_dict().items():
        torch.testing.assert_close(tensor, source.nnet.state_dict()[name])
    assert phase.optimizer.state_dict()["state"] == {}
    assert phase.optimizer.param_groups[0]["lr"] == pytest.approx(phase_config.lr)
    assert phase.optimizer.param_groups[0]["weight_decay"] == pytest.approx(phase_config.weight_decay)
    assert phase._global_step == 0
    assert phase._trainer_iteration == 0
    assert phase._lr_schedule_total_steps == 0
    assert not phase._lr_schedule_mismatch_warned
    assert phase._loaded_checkpoint_path is None
    provenance = phase.training_phase_provenance
    assert provenance == TrainingPhaseProvenance(
        source_checkpoint_sha256=source_sha256,
        source_trainer_iteration=17,
        source_global_step=123,
    )
    assert provenance is not None

    phase.save_checkpoint(str(tmp_path), "phase.pth.tar")
    phase_checkpoint = torch.load(tmp_path / "phase.pth.tar", map_location="cpu", weights_only=True)
    assert phase_checkpoint["training_phase_provenance"] == provenance.as_config()
    assert set(phase_checkpoint["training_phase_provenance"]) == {
        "source_checkpoint_sha256",
        "source_trainer_iteration",
        "source_global_step",
    }

    resumed = LunaNetwork(chess_game, phase_config)
    resumed.load_checkpoint(str(tmp_path), "phase.pth.tar", load_optimizer=False)
    assert resumed.training_phase_provenance == provenance


def test_new_training_phase_rejects_architecture_change_before_mutation(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    source = LunaNetwork(chess_game, small_learner_config)
    source.save_checkpoint(str(tmp_path), "source.pth.tar")
    phase = LunaNetwork(chess_game, replace(small_learner_config, num_channels=24))
    original = {name: tensor.detach().clone() for name, tensor in phase.nnet.state_dict().items()}

    with pytest.raises(ValueError, match=r"model configuration differs.*num_channels"):
        phase.initialize_training_phase(str(tmp_path), "source.pth.tar")

    for name, tensor in phase.nnet.state_dict().items():
        torch.testing.assert_close(tensor, original[name])


def test_new_training_phase_validates_ignored_optimizer_state_before_mutation(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    source = LunaNetwork(chess_game, small_learner_config)
    source.save_checkpoint(str(tmp_path), "source.pth.tar")
    checkpoint = torch.load(tmp_path / "source.pth.tar", map_location="cpu", weights_only=True)
    checkpoint["optimizer"]["param_groups"][0]["lr"] = float("nan")
    torch.save(checkpoint, tmp_path / "corrupt-source.pth.tar")

    phase = LunaNetwork(chess_game, replace(small_learner_config, lr=5e-4))
    original = {name: tensor.detach().clone() for name, tensor in phase.nnet.state_dict().items()}
    with pytest.raises(ValueError, match=r"non-finite value at checkpoint\.optimizer"):
        phase.initialize_training_phase(str(tmp_path), "corrupt-source.pth.tar")

    for name, tensor in phase.nnet.state_dict().items():
        torch.testing.assert_close(tensor, original[name])


def test_new_training_phase_requires_exact_tensor_contract(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    source = LunaNetwork(chess_game, small_learner_config)
    source.save_checkpoint(str(tmp_path), "source.pth.tar")
    checkpoint = torch.load(tmp_path / "source.pth.tar", map_location="cpu", weights_only=True)
    first_name = next(iter(checkpoint["state_dict"]))
    checkpoint["state_dict"][first_name] = checkpoint["state_dict"][first_name].double()
    torch.save(checkpoint, tmp_path / "wrong-dtype.pth.tar")

    phase = LunaNetwork(chess_game, small_learner_config)
    with pytest.raises(ValueError, match=r"does not strictly match.*incompatible"):
        phase.initialize_training_phase(str(tmp_path), "wrong-dtype.pth.tar")


def test_checkpoint_without_phase_provenance_restores_none(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    source = LunaNetwork(chess_game, small_learner_config)
    source.save_checkpoint(str(tmp_path), "source.pth.tar")
    phase = LunaNetwork(chess_game, small_learner_config)
    phase.initialize_training_phase(str(tmp_path), "source.pth.tar")
    assert phase.training_phase_provenance is not None

    checkpoint = torch.load(tmp_path / "source.pth.tar", map_location="cpu", weights_only=True)
    assert "training_phase_provenance" not in checkpoint
    torch.save(checkpoint, tmp_path / "old-format-v2.pth.tar")

    phase.load_checkpoint(str(tmp_path), "old-format-v2.pth.tar", load_optimizer=False)

    assert phase.training_phase_provenance is None


@pytest.mark.parametrize(
    ("raw_provenance", "message"),
    [
        pytest.param("invalid", "must be a string-keyed mapping", id="not-mapping"),
        pytest.param(
            {
                "source_checkpoint_sha256": "0" * 64,
                "source_trainer_iteration": 3,
            },
            "fields are invalid",
            id="missing-field",
        ),
        pytest.param(
            {
                "source_checkpoint_sha256": "g" * 64,
                "source_trainer_iteration": 3,
                "source_global_step": 12,
            },
            "64 lowercase hexadecimal characters",
            id="non-hex-digest",
        ),
        pytest.param(
            {
                "source_checkpoint_sha256": "A" * 64,
                "source_trainer_iteration": 3,
                "source_global_step": 12,
            },
            "64 lowercase hexadecimal characters",
            id="uppercase-digest",
        ),
        pytest.param(
            {
                "source_checkpoint_sha256": "0" * 64,
                "source_trainer_iteration": -1,
                "source_global_step": 12,
            },
            "source_trainer_iteration.*non-negative integer",
            id="negative-iteration",
        ),
        pytest.param(
            {
                "source_checkpoint_sha256": "0" * 64,
                "source_trainer_iteration": 3,
                "source_global_step": True,
            },
            "source_global_step.*non-negative integer",
            id="boolean-global-step",
        ),
    ],
)
def test_checkpoint_rejects_invalid_phase_provenance_before_mutation(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    raw_provenance: object,
    message: str,
) -> None:
    source = LunaNetwork(chess_game, small_learner_config)
    source.save_checkpoint(str(tmp_path), "valid-provenance-source.pth.tar")
    checkpoint = torch.load(
        tmp_path / "valid-provenance-source.pth.tar",
        map_location="cpu",
        weights_only=True,
    )
    first_name = next(iter(checkpoint["state_dict"]))
    checkpoint["state_dict"][first_name] = checkpoint["state_dict"][first_name] + 1
    checkpoint["training_phase_provenance"] = raw_provenance
    torch.save(checkpoint, tmp_path / "invalid-provenance.pth.tar")

    target = LunaNetwork(chess_game, small_learner_config)
    target._global_step = 9
    target._trainer_iteration = 4
    original = {name: tensor.detach().clone() for name, tensor in target.nnet.state_dict().items()}

    with pytest.raises(ValueError, match=message):
        target.load_checkpoint(str(tmp_path), "invalid-provenance.pth.tar", load_optimizer=False)

    for name, tensor in target.nnet.state_dict().items():
        torch.testing.assert_close(tensor, original[name])
    assert target._global_step == 9
    assert target._trainer_iteration == 4
    assert target.training_phase_provenance is None


def test_extra_checkpoint_state_cannot_replace_phase_provenance(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with pytest.raises(ValueError, match=r"reserved checkpoint fields.*training_phase_provenance"):
        network.save_checkpoint(
            str(tmp_path),
            "invalid-extra-state.pth.tar",
            extra_state={"training_phase_provenance": None},
        )

    assert not (tmp_path / "invalid-extra-state.pth.tar").exists()


def test_changed_lr_horizon_warning_is_emitted_once(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    network._lr_schedule_total_steps = 20

    with patch("luna.network.logger.warning") as warning:
        assert network._resolve_lr_schedule_total(40, 1) == 20
        assert network._resolve_lr_schedule_total(40, 1) == 20

    warning.assert_called_once()


def test_checkpoint_loader_rejects_legacy_and_mismatched_model_specs(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    legacy_path = tmp_path / "legacy.pth.tar"
    torch.save({"state_dict": network.nnet.state_dict()}, legacy_path)

    with pytest.raises(ValueError, match="only format version 2"):
        network.load_checkpoint(str(tmp_path), legacy_path.name)

    valid_path = tmp_path / "valid.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    mismatched = torch.load(valid_path, map_location="cpu", weights_only=True)
    mismatched["model_spec"]["action_size"] += 1
    mismatch_path = tmp_path / "mismatch.pth.tar"
    torch.save(mismatched, mismatch_path)

    with pytest.raises(ValueError, match="model specification"):
        network.load_checkpoint(str(tmp_path), mismatch_path.name)


@pytest.mark.parametrize(
    "missing_field",
    ["optimizer", "scaler", "global_step", "trainer_iteration", "lr_schedule_total_steps"],
)
def test_checkpoint_loader_rejects_incomplete_v2_state(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    missing_field: str,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    valid_path = tmp_path / "complete.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    del checkpoint[missing_field]
    incomplete_path = tmp_path / f"missing-{missing_field}.pth.tar"
    torch.save(checkpoint, incomplete_path)

    with pytest.raises(ValueError, match="missing required fields"):
        network.load_checkpoint(str(tmp_path), incomplete_path.name, load_optimizer=False)


def test_checkpoint_loader_rejects_incompatible_training_state(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    original = {name: tensor.detach().clone() for name, tensor in network.nnet.state_dict().items()}
    valid_path = tmp_path / "valid-training-state.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    first_name = next(iter(checkpoint["state_dict"]))
    checkpoint["state_dict"][first_name] = checkpoint["state_dict"][first_name] + 1
    checkpoint["optimizer"] = {"invalid": True}
    corrupt_path = tmp_path / "invalid-training-state.pth.tar"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(RuntimeError, match="training state is incompatible"):
        network.load_checkpoint(str(tmp_path), corrupt_path.name, load_optimizer=True)

    for name, tensor in network.nnet.state_dict().items():
        torch.testing.assert_close(tensor, original[name])


def test_checkpoint_loader_rejects_mismatched_resume_semantics(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    valid_path = tmp_path / "valid-learner-config.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    checkpoint["learner_config"]["unroll_steps"] += 1
    mismatch_path = tmp_path / "mismatched-learner-config.pth.tar"
    torch.save(checkpoint, mismatch_path)

    with pytest.raises(ValueError, match=r"differs in fields.*unroll_steps"):
        network.load_checkpoint(str(tmp_path), mismatch_path.name, load_optimizer=False)


def test_checkpoint_loader_rejects_corrupt_learner_metadata(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    valid_path = tmp_path / "valid-metadata.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    checkpoint["learner_config"] = "corrupt"
    corrupt_path = tmp_path / "corrupt-metadata.pth.tar"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(ValueError, match="string-keyed mapping"):
        network.load_checkpoint(str(tmp_path), corrupt_path.name, load_optimizer=False)


@pytest.mark.parametrize("field_name", ["optimizer", "scaler"])
def test_checkpoint_loader_rejects_invalid_training_state_containers_for_inference(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    field_name: str,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    valid_path = tmp_path / "valid-containers.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    checkpoint[field_name] = None
    corrupt_path = tmp_path / f"invalid-{field_name}.pth.tar"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(ValueError, match="optimizer and scaler states must be mappings"):
        network.load_checkpoint(str(tmp_path), corrupt_path.name, load_optimizer=False)


@pytest.mark.parametrize("counter_name", ["global_step", "lr_schedule_total_steps"])
def test_checkpoint_counter_validation_precedes_model_mutation(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    counter_name: str,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    valid_path = tmp_path / "valid-counter.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    original = {name: tensor.detach().clone() for name, tensor in network.nnet.state_dict().items()}
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    first_name = next(iter(checkpoint["state_dict"]))
    checkpoint["state_dict"][first_name] = checkpoint["state_dict"][first_name] + 1
    checkpoint[counter_name] = -1
    corrupt_path = tmp_path / f"negative-{counter_name}.pth.tar"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(ValueError, match="non-negative integer"):
        network.load_checkpoint(str(tmp_path), corrupt_path.name, load_optimizer=False)

    for name, tensor in network.nnet.state_dict().items():
        torch.testing.assert_close(tensor, original[name])


def test_checkpoint_save_rejects_non_finite_model_state_without_creating_file(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    parameter = next(network.nnet.parameters())
    with torch.no_grad():
        parameter.view(-1)[0] = float("nan")
    checkpoint_path = tmp_path / "non-finite-save.pth.tar"

    with pytest.raises(ValueError, match="non-finite value"):
        network.save_checkpoint(str(tmp_path), checkpoint_path.name)

    assert not checkpoint_path.exists()


def test_checkpoint_save_rejects_non_finite_numpy_extra_state(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    checkpoint_path = tmp_path / "non-finite-extra.pth.tar"

    with pytest.raises(ValueError, match=r"non-finite value at checkpoint\.diagnostics"):
        network.save_checkpoint(
            str(tmp_path),
            checkpoint_path.name,
            extra_state={"diagnostics": np.array([0.0, np.nan])},
        )

    assert not checkpoint_path.exists()


def test_checkpoint_loader_rejects_non_finite_model_state_before_mutation(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    original = {name: tensor.detach().clone() for name, tensor in network.nnet.state_dict().items()}
    valid_path = tmp_path / "finite-model.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    first_name = next(iter(checkpoint["state_dict"]))
    corrupt_tensor = checkpoint["state_dict"][first_name].clone()
    corrupt_tensor.view(-1)[0] = float("nan")
    checkpoint["state_dict"][first_name] = corrupt_tensor
    corrupt_path = tmp_path / "non-finite-model.pth.tar"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(ValueError, match=r"non-finite value at checkpoint\.state_dict"):
        network.load_checkpoint(str(tmp_path), corrupt_path.name, load_optimizer=False)

    for name, tensor in network.nnet.state_dict().items():
        torch.testing.assert_close(tensor, original[name])


def test_checkpoint_loader_validates_optimizer_finiteness_for_inference_before_mutation(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    original = {name: tensor.detach().clone() for name, tensor in network.nnet.state_dict().items()}
    valid_path = tmp_path / "finite-training-state.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    first_name = next(iter(checkpoint["state_dict"]))
    checkpoint["state_dict"][first_name] = checkpoint["state_dict"][first_name] + 1
    checkpoint["optimizer"]["param_groups"][0]["lr"] = float("nan")
    corrupt_path = tmp_path / "non-finite-optimizer.pth.tar"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(ValueError, match=r"non-finite value at checkpoint\.optimizer\.param_groups\[0\]\.lr"):
        network.load_checkpoint(str(tmp_path), corrupt_path.name, load_optimizer=False)

    for name, tensor in network.nnet.state_dict().items():
        torch.testing.assert_close(tensor, original[name])


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("scale", 0.0, "scale must be a positive normal float32"),
        ("scale", math.ulp(0.0), "scale must be a positive normal float32"),
        ("growth_factor", 1.0, "growth_factor must be greater than 1"),
        ("growth_factor", 1.00000001, "growth_factor must be greater than 1"),
        ("backoff_factor", 1.0, "backoff_factor must be between 0 and 1"),
        ("backoff_factor", 0.999999999, "backoff_factor must be between 0 and 1"),
        ("growth_interval", 0, "growth_interval must be positive"),
        ("growth_interval", 2**31, "growth_interval must fit int32"),
        ("_growth_tracker", -1, "_growth_tracker must be non-negative"),
        ("_growth_tracker", 2_000, "_growth_tracker must be less than growth_interval"),
    ],
)
def test_checkpoint_loader_rejects_invalid_scaler_semantics_before_mutation(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    field_name: str,
    value: float | int,
    message: str,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    original = {name: tensor.detach().clone() for name, tensor in network.nnet.state_dict().items()}
    valid_path = tmp_path / "finite-scaler-state.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    first_name = next(iter(checkpoint["state_dict"]))
    checkpoint["state_dict"][first_name] = checkpoint["state_dict"][first_name] + 1
    checkpoint["scaler"] = GradScaler("cpu", enabled=True).state_dict()
    checkpoint["scaler"][field_name] = value
    corrupt_path = tmp_path / f"invalid-scaler-{field_name}.pth.tar"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(ValueError, match=message):
        network.load_checkpoint(str(tmp_path), corrupt_path.name, load_optimizer=False)

    for name, tensor in network.nnet.state_dict().items():
        torch.testing.assert_close(tensor, original[name])


def test_checkpoint_loader_rejects_non_tensor_model_state(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    valid_path = tmp_path / "valid-model-state.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    checkpoint["state_dict"] = {"invalid": None}
    corrupt_path = tmp_path / "non-tensor-state.pth.tar"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(ValueError, match="state_dict must map string names to tensors"):
        network.load_checkpoint(str(tmp_path), corrupt_path.name, load_optimizer=False)
