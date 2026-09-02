"""Regression tests for EfficientZeroV2 training loop."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from luna.config import EzV2LearnerConfig
from luna.ezv2_networks import SimSiamProjector
from luna.game.chess_game import ACTION_SIZE, OBS_PLANES, ChessGame
from luna.network import (
    LunaNetwork,
    PreparedBatch,
)
from luna.network_training_forward import _policy_loss_scale, _priority_errors
from luna.network_training_types import Microbatch, RootState
from luna.network_types import ReanalysisBatchStats
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


def test_masked_value_priority_uses_policy_kl() -> None:
    policy_target = torch.tensor(
        [
            [[0.75, 0.25, 0.0]],
            [[0.50, 0.50, 0.0]],
        ]
    )
    policy_entropy = -torch.xlogy(policy_target[:, 0], policy_target[:, 0]).sum(dim=1)
    root = RootState(
        latent=torch.empty(0),
        value_prediction=torch.tensor([-0.5, 0.95]),
        policy_loss=policy_entropy + torch.tensor([0.4, 0.25]),
        value_loss=torch.empty(0),
        reconstruction_loss=torch.empty(0),
        reconstruction_logits=None,
        reconstruction_target=None,
        target_latents=None,
    )
    unused = torch.empty(0)
    batch = Microbatch(
        observations=unused,
        valid_moves=unused,
        target_values=torch.tensor([[-1.0], [0.0]]),
        target_rewards=unused,
        target_policies=policy_target,
        unroll_observations=unused,
        actions=unused,
        importance_weights=unused,
        unroll_mask=unused,
        consistency_mask=unused,
        policy_mask=torch.ones(2, 1),
        policy_loss_scale=1.0,
        value_mask=torch.tensor([[1.0], [0.0]]),
        unroll_valid_moves=unused,
        tree_indices=[],
    )

    errors = _priority_errors(root, batch)

    np.testing.assert_allclose(errors, [0.5, 0.25], atol=1e-6)


def test_policy_mask_scale_preserves_is_weighted_gradient_mass() -> None:
    importance_weights = np.array([1.0, 0.5, 0.25, 0.25], dtype=np.float32)
    policy_mask = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    prepared = PreparedBatch(
        {"policy_mask": policy_mask},
        importance_weights,
        [],
        ReanalysisBatchStats(0, 0, 0.0),
    )

    scale = _policy_loss_scale(prepared)
    active_rows = policy_mask.any(axis=1)

    assert scale == pytest.approx(4.0 / 3.0)
    assert np.mean(importance_weights * active_rows * scale) == pytest.approx(np.mean(importance_weights))


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


def test_reconstruction_loss_ignores_padded_unroll_states(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    config = replace(
        small_learner_config,
        model_name="balanced_reconstruction",
        batch_size=2,
        unroll_steps=3,
        td_steps=4,
        mixed_precision=False,
        dataloader_workers=0,
        lr=0.0,
        lr_min=0.0,
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        reward_loss_weight=0.0,
        consistency_loss_weight=0.0,
        reconstruction_loss_weight=1.0,
    )
    network = LunaNetwork(ChessGame(), config)
    replay = PrioritizedReplayBuffer(capacity=8)
    replay.save_trajectory(_make_trajectory(length=4))
    prepared = network._prepare_batch(
        replay,
        bs=2,
        unroll=3,
        td=4,
        discount=1.0,
        training_step=1,
        mcts_for_reanalyze=None,
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

    def constant_reconstruction_loss(logits: torch.Tensor, _target: torch.Tensor) -> torch.Tensor:
        return logits.flatten(1).sum(dim=1) * 0.0 + 4.0

    with (
        patch.object(network, "_prepare_batch", return_value=controlled),
        patch("luna.network._piece_reconstruction_loss", side_effect=constant_reconstruction_loss),
    ):
        metrics = network.train_ezv2(replay, steps=1)

    assert metrics["reconstruction"] == pytest.approx(4.0)
    assert metrics["total"] == pytest.approx(4.0)


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
    sampled = network._prepare_batch(
        replay,
        bs=2,
        unroll=1,
        td=1,
        discount=1.0,
        training_step=1,
        mcts_for_reanalyze=None,
    )
    collated = sampled.collated
    collated["target_values"][:, 0] = np.array([-1.0, 1.0], dtype=np.float32)
    prepared = PreparedBatch(collated, sampled.is_weights, [0, 0], sampled.reanalysis)

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
