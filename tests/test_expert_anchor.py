"""Tests for LC0 expert anchoring during online self-play training."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from luna.config import EzV2LearnerConfig, TrainingRunConfig, validate_training_configuration
from luna.expert_anchor import ExpertAnchorBatchSource, expert_anchor_fingerprint
from luna.expert_anchor_loss import expert_anchor_forward_and_backward
from luna.game.chess_game import ACTION_SIZE, OBS_PLANES, ChessGame
from luna.lc0_corpus import dataset_fingerprint
from luna.lc0_dataset import Lc0Batch, Lc0DatasetConfig
from luna.network import LunaNetwork
from luna.network_training_types import OptimizerOutcome, StepAccumulation
from luna.replay_buffer import PrioritizedReplayBuffer, Trajectory
from luna.self_play_actors import _actor_learner_config


class _StubAnchor:
    def __init__(self, batch: Lc0Batch) -> None:
        self.batch = batch
        self.calls = 0

    def next_batch(self) -> Lc0Batch:
        self.calls += 1
        return self.batch


def _anchor_batch(size: int = 1, marker: float = 0.0) -> Lc0Batch:
    policies = np.full((size, ACTION_SIZE), 1.0 / ACTION_SIZE, dtype=np.float32)
    values = np.tile(np.asarray((0.2, 0.5, 0.3), dtype=np.float32), (size, 1))
    return Lc0Batch(
        observations=np.full((size, 8, 8, OBS_PLANES), marker, dtype=np.float32),
        policies=policies,
        value_targets=values,
        valid_moves=np.ones((size, ACTION_SIZE), dtype=np.bool_),
        visits=np.full(size, 32, dtype=np.int64),
    )


def _trajectory() -> Trajectory:
    return Trajectory(
        observations=np.zeros((1, 8, 8, OBS_PLANES), dtype=np.float32),
        actions=[0],
        rewards=[0.0],
        root_policies=np.full((1, ACTION_SIZE), 1.0 / ACTION_SIZE, dtype=np.float32),
        root_values=[0.0],
        valids=np.ones((1, ACTION_SIZE), dtype=np.bool_),
    )


def _active_learner(path: Path, fingerprint: str, **changes: object) -> EzV2LearnerConfig:
    learner = EzV2LearnerConfig(
        device="cpu",
        batch_size=1,
        num_channels=8,
        repr_blocks=0,
        dyn_blocks=0,
        proj_dim=16,
        mixed_precision=False,
        dataloader_workers=0,
        unroll_steps=1,
        td_steps=1,
        lr=0.0,
        lr_min=0.0,
        expert_anchor_path=str(path),
        expert_anchor_fingerprint=fingerprint,
        expert_anchor_fraction=1.0,
        expert_anchor_loss_weight=0.25,
    )
    return replace(learner, **changes)


def test_directory_fingerprint_covers_shard_names_and_contents(tmp_path: Path) -> None:
    directory = tmp_path / "anchor"
    directory.mkdir()
    first = directory / "a.tar"
    second = directory / "b.tar"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    original = expert_anchor_fingerprint(directory)
    first.rename(directory / "c.tar")

    assert expert_anchor_fingerprint(directory) != original
    assert expert_anchor_fingerprint(second) == dataset_fingerprint(second)


def test_directory_source_advances_deterministic_member_windows(tmp_path: Path) -> None:
    directory = tmp_path / "anchor"
    directory.mkdir()
    (directory / "b.tar").write_bytes(b"second")
    (directory / "a.tar").write_bytes(b"first")
    learner = _active_learner(
        directory,
        expert_anchor_fingerprint(directory),
        batch_size=4,
        expert_anchor_fraction=0.5,
    )
    calls: list[tuple[int, str, int, int, int, int | None]] = []

    def batches(
        _path: Path,
        config: Lc0DatasetConfig,
        _game: ChessGame,
        *,
        archive_offset: int,
        member_window_index: int,
        member_window_count: int,
    ) -> Iterator[Lc0Batch]:
        calls.append(
            (
                config.epoch,
                config.value_source,
                archive_offset,
                member_window_index,
                member_window_count,
                config.max_samples,
            )
        )
        return iter(_anchor_batch(config.batch_size, float(config.epoch)) for _ in range(1_000))

    with patch("luna.expert_anchor.iter_lc0_corpus_batches", side_effect=batches):
        source = ExpertAnchorBatchSource(learner, ChessGame(), seed=7, starting_step=0)
        markers = [float(source.next_batch().observations[0, 0, 0, 0]) for _ in range(1_001)]

    assert markers[:1] == [0.0]
    assert markers[-1:] == [1.0]
    assert calls == [
        (0, "root", 0, 0, 10, 2_000),
        (1, "root", 1, 1, 10, 2_000),
    ]


def test_resumed_source_continues_the_uninterrupted_batch_sequence(tmp_path: Path) -> None:
    directory = tmp_path / "anchor"
    directory.mkdir()
    (directory / "b.tar").write_bytes(b"second")
    (directory / "a.tar").write_bytes(b"first")
    learner = _active_learner(directory, expert_anchor_fingerprint(directory))

    produced: list[int] = []

    def batches(
        _path: Path,
        config: Lc0DatasetConfig,
        _game: ChessGame,
        **_window: int,
    ) -> Iterator[Lc0Batch]:
        for index in range(1_000):
            marker = 1_000 * config.epoch + index
            produced.append(marker)
            yield _anchor_batch(marker=marker)

    with patch("luna.expert_anchor.iter_lc0_corpus_batches", side_effect=batches):
        resumed = ExpertAnchorBatchSource(learner, ChessGame(), seed=7, starting_step=9_009)
        resumed_sequence = [float(resumed.next_batch().observations[0, 0, 0, 0]) for _ in range(5)]

    assert resumed_sequence == [9_009.0, 9_010.0, 9_011.0, 9_012.0, 9_013.0]
    assert produced == list(range(9_000, 9_014))


def test_source_rejects_a_fingerprint_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "anchor.tar"
    archive.write_bytes(b"anchor")
    learner = _active_learner(archive, "0" * 64)

    with pytest.raises(ValueError, match="expert anchor fingerprint mismatch"):
        ExpertAnchorBatchSource(learner, ChessGame(), seed=0, starting_step=0)


@pytest.mark.parametrize(
    "changes",
    [
        {"expert_anchor_fingerprint": ""},
        {"expert_anchor_fraction": 0.0},
        {"expert_anchor_loss_weight": 0.0},
        {"support_size": 2},
        {"policy_loss_weight": 0.0, "value_loss_weight": 0.0},
    ],
)
def test_incomplete_or_incompatible_anchor_objective_is_rejected(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    learner = _active_learner(tmp_path / "anchor.tar", "0" * 64, **changes)

    with pytest.raises(ValueError, match="expert anchor"):
        LunaNetwork(ChessGame(), learner)


def test_training_configuration_rejects_a_missing_anchor_path(tmp_path: Path) -> None:
    learner = _active_learner(tmp_path / "missing.tar", "0" * 64)

    with pytest.raises(ValueError, match="expert_anchor_path does not exist"):
        validate_training_configuration(TrainingRunConfig(), learner)


def test_self_play_actor_does_not_open_the_expert_dataset(tmp_path: Path) -> None:
    learner = _active_learner(tmp_path / "anchor.tar", "0" * 64)

    actor = _actor_learner_config(learner)

    assert actor.expert_anchor_path == ""
    assert actor.expert_anchor_fingerprint == ""
    assert actor.expert_anchor_fraction == 0.0
    assert actor.expert_anchor_loss_weight == 0.0


def test_online_training_reports_soft_wdl_anchor_without_expanding_per(tmp_path: Path) -> None:
    archive = tmp_path / "anchor.tar"
    archive.write_bytes(b"unused")
    learner = _active_learner(archive, dataset_fingerprint(archive))
    network = LunaNetwork(ChessGame(), learner)
    replay = PrioritizedReplayBuffer(capacity=1)
    replay.save_trajectory(_trajectory())
    expert_batch = _anchor_batch()
    anchor = _StubAnchor(expert_batch)
    observations = torch.as_tensor(expert_batch.observations, dtype=torch.float32)
    valid_moves = torch.as_tensor(expert_batch.valid_moves, dtype=torch.float32)
    values = torch.as_tensor(expert_batch.value_targets, dtype=torch.float32)
    with torch.no_grad():
        _latent, log_policy, value_logits = network._training_initial_inference(observations, valid_moves)
        expected_policy_ce = float((-(torch.as_tensor(expert_batch.policies) * log_policy).sum(dim=1)).mean())
        expected_value_ce = float((-(values * torch.log_softmax(value_logits, dim=1)).sum(dim=1)).mean())
        support = torch.tensor((-1.0, 0.0, 1.0))
        expected_q_mae = float(((torch.softmax(value_logits, dim=1) @ support) - (values @ support)).abs().mean())

    with (
        patch.object(replay, "update_priorities", wraps=replay.update_priorities) as update_priorities,
        patch("luna.network_training_metrics.wandb.run", object()),
        patch("luna.network_training_metrics.wandb.log") as wandb_log,
    ):
        metrics = network.train_ezv2(
            replay,
            steps=1,
            total_train_steps=1,
            expert_anchor=cast(ExpertAnchorBatchSource, anchor),
        )

    expected_weighted = learner.expert_anchor_loss_weight * (
        learner.policy_loss_weight * expected_policy_ce + learner.value_loss_weight * expected_value_ce
    )
    assert metrics["expert_anchor"] == pytest.approx(expected_weighted, rel=1e-5)
    assert anchor.calls == 1
    update_priorities.assert_called_once()
    assert len(update_priorities.call_args.args[0]) == learner.batch_size
    reported = wandb_log.call_args.args[0]
    assert reported["train/expert_anchor_policy_ce"] == pytest.approx(expected_policy_ce, rel=1e-5)
    assert reported["train/expert_anchor_value_wdl_ce"] == pytest.approx(expected_value_ce, rel=1e-5)
    assert reported["train/expert_anchor_q_mae"] == pytest.approx(expected_q_mae, rel=1e-5)
    assert reported["train/expert_anchor_positions"] == 1


def test_partial_expert_microbatch_matches_full_batch_gradient_and_update(tmp_path: Path) -> None:
    config = _active_learner(
        tmp_path / "anchor.tar",
        "0" * 64,
        batch_size=4,
        grad_accum_steps=1,
        lr=1e-3,
        lr_min=1e-3,
        weight_decay=0.0,
    )
    full_batch = LunaNetwork(ChessGame(), config)
    accumulated = LunaNetwork(ChessGame(), replace(config, grad_accum_steps=2))
    accumulated.nnet.load_state_dict(full_batch.nnet.state_dict())
    batch = _anchor_batch(size=3)
    batch.observations[:] = 0.0
    batch.policies[:] = 0.0
    batch.value_targets[:] = np.asarray(
        ((0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8)),
        dtype=np.float32,
    )
    for row, action in enumerate((0, 17, 65)):
        batch.observations[row, :, :, row] = float(row + 1)
        batch.policies[row, action] = 1.0

    def backward_and_step(
        network: LunaNetwork,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        network.optimizer.zero_grad(set_to_none=True)
        expert_anchor_forward_and_backward(network, batch)
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in network.nnet.named_parameters()
            if parameter.grad is not None
        }
        network.optimizer.step()
        return gradients, network.nnet.state_dict()

    full_gradients, full_state = backward_and_step(full_batch)
    accumulated_gradients, accumulated_state = backward_and_step(accumulated)

    assert full_gradients.keys() == accumulated_gradients.keys()
    for name in full_gradients:
        torch.testing.assert_close(accumulated_gradients[name], full_gradients[name], rtol=1e-5, atol=5e-6)
    compared_updates = 0
    for name, full_gradient in full_gradients.items():
        accumulated_gradient = accumulated_gradients[name]
        material = torch.maximum(full_gradient.abs(), accumulated_gradient.abs()) >= 1e-5
        if material.any():
            compared_updates += int(material.sum())
            torch.testing.assert_close(
                accumulated_state[name][material],
                full_state[name][material],
                rtol=1e-5,
                atol=1e-7,
            )
    assert compared_updates > 0


def test_amp_retry_reuses_the_same_prepared_expert_batch(tmp_path: Path) -> None:
    learner = _active_learner(tmp_path / "unused.tar", "0" * 64, expert_anchor_loss_weight=0.0)
    learner = replace(
        learner,
        expert_anchor_path="",
        expert_anchor_fingerprint="",
        expert_anchor_fraction=0.0,
    )
    network = LunaNetwork(ChessGame(), learner)
    replay = PrioritizedReplayBuffer(capacity=1)
    replay.save_trajectory(_trajectory())
    anchor = _StubAnchor(_anchor_batch())
    prepared_ids: list[int] = []
    outcomes = [OptimizerOutcome(True, 1.0, 2.0, 1.0), OptimizerOutcome(False, 1.0, 1.0, 1.0)]

    def capture_batch(
        _network: object,
        prepared: object,
        _step: int,
        _settings: object,
        _functions: object,
    ) -> StepAccumulation:
        prepared_ids.append(id(prepared))
        return StepAccumulation.empty(network.device)

    with (
        patch("luna.network_training.run_microbatches", side_effect=capture_batch),
        patch("luna.network_training.apply_optimizer_update", side_effect=outcomes),
        patch("luna.network_training.record_successful_step"),
        patch("luna.network_training.report_training"),
    ):
        network.train_ezv2(
            replay,
            steps=1,
            total_train_steps=1,
            expert_anchor=cast(ExpertAnchorBatchSource, anchor),
        )

    assert prepared_ids[0] == prepared_ids[1]
    assert anchor.calls == 1


def test_legacy_checkpoint_defaults_anchor_configuration(
    tmp_path: Path,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(ChessGame(), small_learner_config)
    network.save_checkpoint(str(tmp_path), "current.pth.tar")
    checkpoint = torch.load(tmp_path / "current.pth.tar", map_location="cpu", weights_only=True)
    for field in (
        "expert_anchor_path",
        "expert_anchor_fingerprint",
        "expert_anchor_fraction",
        "expert_anchor_loss_weight",
    ):
        del checkpoint["learner_config"][field]
    legacy = tmp_path / "legacy.pth.tar"
    torch.save(checkpoint, legacy)

    restored = LunaNetwork.from_checkpoint(ChessGame(), legacy, device="cpu")

    assert restored._learner.expert_anchor_path == ""
    assert restored._learner.expert_anchor_fingerprint == ""
    assert restored._learner.expert_anchor_fraction == 0.0
    assert restored._learner.expert_anchor_loss_weight == 0.0


def test_resume_allows_relocating_the_same_fingerprinted_anchor(tmp_path: Path) -> None:
    original = tmp_path / "original.tar"
    relocated = tmp_path / "relocated.tar"
    original.write_bytes(b"same anchor")
    relocated.write_bytes(b"same anchor")
    fingerprint = dataset_fingerprint(original)
    source = LunaNetwork(ChessGame(), _active_learner(original, fingerprint))
    source.save_checkpoint(str(tmp_path), "anchored.pth.tar")
    restored = LunaNetwork(ChessGame(), _active_learner(relocated, fingerprint))

    restored.load_checkpoint(str(tmp_path), "anchored.pth.tar")

    assert restored._learner.expert_anchor_path == str(relocated)
