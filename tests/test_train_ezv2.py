"""Regression tests for EfficientZeroV2 training loop."""

import math
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch._inductor import config as torch_inductor_config

from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.ezv2_networks import SimSiamProjector
from luna.game.chess_game import ACTION_SIZE, OBS_PLANES, ChessGame
from luna.network import (
    LunaNetwork,
    RepresentationCollapseError,
    _configure_dynamic_cudagraphs,
    _latent_health_metrics,
    _piece_class_targets,
    _scale_gradient,
)
from luna.replay_buffer import PrioritizedReplayBuffer, Trajectory


def _make_trajectory(length: int = 4, *, truncated: bool = False) -> Trajectory:
    return Trajectory(
        observations=[np.random.randn(8, 8, OBS_PLANES).astype(np.float32) for _ in range(length)],
        actions=[np.random.randint(0, min(256, ACTION_SIZE)) for _ in range(length)],
        rewards=np.zeros(length, dtype=np.float32),
        root_policies=[np.full(ACTION_SIZE, 1.0 / ACTION_SIZE, dtype=np.float32) for _ in range(length)],
        root_values=np.zeros(length, dtype=np.float32),
        valids=np.ones((length, ACTION_SIZE), dtype=np.float32),
        truncated=truncated,
        truncation_bootstrap_value=0.0 if truncated else None,
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


def test_state_anchor_reconstructs_empty_and_piece_classes() -> None:
    observation = torch.zeros(1, 8, 8, OBS_PLANES)
    observation[0, 1, 2, 0] = 1.0
    observation[0, 6, 5, 11] = 1.0

    target = _piece_class_targets(observation)

    assert target.shape == (1, 8, 8)
    assert target[0, 0, 0].item() == 0
    assert target[0, 1, 2].item() == 1
    assert target[0, 6, 5].item() == 12


def test_reconstruction_model_trains_without_simsiam_target_branch(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    config = replace(
        small_learner_config,
        model_name="balanced_reconstruction",
        batch_size=1,
        dataloader_workers=0,
        mixed_precision=False,
        td_steps=4,
        unroll_steps=1,
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        reward_loss_weight=0.0,
        consistency_loss_weight=0.0,
        reconstruction_loss_weight=0.5,
    )
    network = LunaNetwork(ChessGame(), config)
    replay = PrioritizedReplayBuffer(capacity=8)
    replay.save_trajectory(_make_trajectory(length=4))
    np.random.seed(0)
    representation_before = network.nnet.representation.conv_in.weight.detach().clone()
    dynamics_before = network.nnet.dynamics.conv_in.weight.detach().clone()
    assert network.nnet.piece_reconstruction is not None
    reconstruction_before = network.nnet.piece_reconstruction.classifier.weight.detach().clone()

    with (
        patch.object(
            network,
            "_training_representation",
            side_effect=AssertionError("disabled SimSiam target branch was evaluated"),
        ),
        patch("luna.network._simsiam_loss", side_effect=AssertionError("SimSiam loss was evaluated")),
    ):
        metrics = network.train_ezv2(replay, steps=1, total_train_steps=1)

    assert network._global_step == 1
    assert metrics["consistency"] == pytest.approx(0.0)
    assert metrics["reconstruction"] > 0.0
    assert not torch.equal(representation_before, network.nnet.representation.conv_in.weight)
    assert not torch.equal(dynamics_before, network.nnet.dynamics.conv_in.weight)
    assert not torch.equal(
        reconstruction_before,
        network.nnet.piece_reconstruction.classifier.weight,
    )


def test_root_only_training_skips_dynamics() -> None:
    config = EzV2LearnerConfig(
        device="cpu",
        model_name="balanced_reconstruction",
        num_channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        proj_dim=32,
        batch_size=1,
        grad_accum_steps=1,
        mixed_precision=False,
        dataloader_workers=0,
        unroll_steps=0,
        td_steps=4,
        reward_loss_weight=0.0,
        consistency_loss_weight=0.0,
        reconstruction_loss_weight=0.25,
        train_value_on_truncated=False,
    )
    network = LunaNetwork(ChessGame(), config)
    replay = PrioritizedReplayBuffer(capacity=8)
    replay.save_trajectory(_make_trajectory(length=4, truncated=True))
    representation_before = network.nnet.representation.conv_in.weight.detach().clone()
    dynamics_before = network.nnet.dynamics.conv_in.weight.detach().clone()

    metrics = network.train_ezv2(replay, steps=1, total_train_steps=1)

    assert math.isfinite(metrics["total"])
    assert metrics["policy"] > 0.0
    assert metrics["value"] == pytest.approx(0.0)
    assert metrics["reward"] == pytest.approx(0.0)
    assert metrics["consistency"] == pytest.approx(0.0)
    assert not torch.equal(representation_before, network.nnet.representation.conv_in.weight)
    assert torch.equal(dynamics_before, network.nnet.dynamics.conv_in.weight)


def test_state_anchor_collapse_guard_requires_three_consecutive_reports(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    config = replace(
        small_learner_config,
        model_name="balanced_reconstruction",
        reconstruction_loss_weight=0.5,
    )
    network = LunaNetwork(ChessGame(), config)

    network._check_representation_diversity(0.001, 100)
    network._check_representation_diversity(0.001, 150)
    with pytest.raises(RepresentationCollapseError, match="consecutive collapsed-latent reports"):
        network._check_representation_diversity(0.001, 200)


def test_state_anchor_collapse_guard_resets_after_healthy_report(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    config = replace(
        small_learner_config,
        model_name="balanced_reconstruction",
        reconstruction_loss_weight=0.5,
    )
    network = LunaNetwork(ChessGame(), config)

    network._check_representation_diversity(0.001, 100)
    network._check_representation_diversity(0.5, 150)
    network._check_representation_diversity(0.001, 200)

    assert network._low_diversity_reports == 1


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
