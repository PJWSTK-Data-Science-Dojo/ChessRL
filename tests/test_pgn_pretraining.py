"""Offline PGN pretraining orchestration tests."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

import pretrain_pgn
from luna.config import EzV2LearnerConfig
from luna.network import LunaNetwork
from luna.pgn_dataset import PgnDataset, PgnDatasetConfig, PgnDatasetStats
from luna.pgn_pretraining import (
    PgnPretrainingConfig,
    _resume_seed,
    evaluate_validation,
    run_pgn_pretraining,
    validate_pretraining_config,
)
from luna.pgn_pretraining_checkpoints import (
    CHECKPOINT_METADATA_KEY,
)
from luna.pgn_pretraining_validation import ValidationPlan
from luna.replay_buffer import Trajectory


def _offline_learner() -> EzV2LearnerConfig:
    return EzV2LearnerConfig(
        device="cpu",
        model_name="balanced_reconstruction",
        num_channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        proj_dim=32,
        batch_size=2,
        unroll_steps=1,
        td_steps=0,
        reward_loss_weight=0.0,
        consistency_loss_weight=0.0,
        reconstruction_loss_weight=0.1,
        reanalyze_mcts_sims=0,
        reanalyze_prob=0.0,
        reanalyze_policy=False,
        dataloader_workers=0,
    )


def _config(tmp_path: Path) -> PgnPretrainingConfig:
    dataset_path = tmp_path / "expert.pgn.zst"
    source = tmp_path / "source.pth.tar"
    dataset_path.write_bytes(b"expert games")
    source.write_bytes(b"source checkpoint")
    return PgnPretrainingConfig(
        dataset_path=dataset_path,
        output_dir=tmp_path / "output",
        source_checkpoint=source,
        total_steps=5,
        chunk_steps=2,
        validation_batch_size=4,
        validation_positions=10,
        dataset=PgnDatasetConfig(max_positions=100),
        learner=_offline_learner(),
    )


def _stats(trajectory: Trajectory) -> PgnDatasetStats:
    positions = trajectory.game_length
    return PgnDatasetStats(2, 2, 0, 0, 0, 1, 1, positions, positions, 2 * positions, 0, False)


def _dataset(trajectory: Trajectory) -> PgnDataset:
    return PgnDataset((trajectory,), (trajectory,), _stats(trajectory))


class _ValidationNetwork:
    def __init__(self, policies: np.ndarray, values: np.ndarray) -> None:
        self._policies = policies
        self._values = values
        self._offset = 0

    def batched_initial_inference(
        self,
        obs_batch: np.ndarray,
        valid_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, object]:
        del valid_batch
        end = self._offset + len(obs_batch)
        policies = self._policies[self._offset : end]
        values = self._values[self._offset : end]
        self._offset = end
        return policies, values, object()


def test_validation_reports_expert_policy_and_value_metrics(
    make_trajectory: Callable[[int], Trajectory],
) -> None:
    trajectory = make_trajectory(2)
    trajectory.actions[:] = [0, 1]
    trajectory.root_values[:] = [0.5, -0.5]
    policies = np.zeros((2, trajectory.root_policies.shape[1]), dtype=np.float32)
    policies[0, [0, 2]] = [0.8, 0.2]
    policies[1, [1, 2, 3]] = [0.3, 0.6, 0.1]
    network = _ValidationNetwork(policies, np.asarray([0.2, -0.1], dtype=np.float32))

    metrics = evaluate_validation(network, [trajectory], ValidationPlan(batch_size=1, maximum_positions=2, seed=0))

    assert metrics.policy_top1 == pytest.approx(0.5)
    assert metrics.policy_top5 == pytest.approx(1.0)
    assert metrics.policy_nll == pytest.approx((-np.log(0.8) - np.log(0.3)) / 2)
    assert metrics.value_mae == pytest.approx(0.35)


@pytest.mark.parametrize(
    ("learner", "message"),
    [
        (replace(_offline_learner(), td_steps=1), "td_steps=0"),
        (replace(_offline_learner(), reward_loss_weight=0.1), "reward and consistency"),
        (replace(_offline_learner(), consistency_loss_weight=0.1), "reward and consistency"),
        (replace(_offline_learner(), reanalyze_prob=0.1), "disable replay reanalysis"),
        (replace(_offline_learner(), reanalyze_policy=True), "disable replay reanalysis"),
    ],
)
def test_pretraining_rejects_online_only_objectives(
    tmp_path: Path,
    learner: EzV2LearnerConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_pretraining_config(replace(_config(tmp_path), learner=learner))


def test_fresh_and_resumed_phases_require_explicit_wandb_semantics(tmp_path: Path) -> None:
    fresh = _config(tmp_path)
    with pytest.raises(ValueError, match=r"fresh.*existing W&B"):
        validate_pretraining_config(replace(fresh, wandb_resume="must"))

    fresh.output_dir.mkdir()
    resume = fresh.output_dir / "latest.pth.tar"
    resume.write_bytes(b"checkpoint")
    resumed = replace(fresh, source_checkpoint=None, resume_checkpoint=resume)
    with pytest.raises(ValueError, match=r"resumed.*allow.*must"):
        validate_pretraining_config(resumed)


def test_wandb_pretraining_requires_explicit_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit wandb_run_id"):
        validate_pretraining_config(replace(_config(tmp_path), wandb_project="ChessRL"))


class _FakeNetwork:
    def __init__(self, *, interrupt: bool = False, resume_step: int = 0) -> None:
        self.global_step = 0
        self.trainer_iteration = 0
        self.training_phase_provenance = None
        self.interrupt = interrupt
        self.initialized_from: tuple[str, str] | None = None
        self.train_requests: list[tuple[int, int]] = []
        self.saved: list[tuple[str, dict[str, object] | None]] = []
        self.resume_step = resume_step

    def initialize_training_phase(self, folder: str, filename: str) -> None:
        self.initialized_from = (folder, filename)

    def load_checkpoint(self, folder: str, filename: str, *, load_optimizer: bool) -> None:
        del folder, filename, load_optimizer
        self.global_step = self.resume_step

    def train_ezv2(
        self,
        replay: object,
        steps: int,
        total_train_steps: int,
    ) -> dict[str, float]:
        del replay
        self.train_requests.append((steps, total_train_steps))
        if self.interrupt:
            self.global_step += 1
            raise KeyboardInterrupt
        self.global_step += steps
        return {"total": 0.0}

    def batched_initial_inference(
        self,
        obs_batch: np.ndarray,
        valid_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, object]:
        del valid_batch
        policies = np.zeros((len(obs_batch), 4288), dtype=np.float32)
        policies[:, 0] = 1.0
        return policies, np.zeros(len(obs_batch), dtype=np.float32), object()

    def save_checkpoint(
        self,
        folder: str,
        filename: str,
        *,
        extra_state: dict[str, object] | None = None,
    ) -> None:
        del folder
        self.saved.append((filename, extra_state))


def test_pretraining_uses_uniform_replay_and_chunked_atomic_checkpoints(
    tmp_path: Path,
    make_trajectory: Callable[[int], Trajectory],
) -> None:
    config = _config(tmp_path)
    dataset = _dataset(make_trajectory(2))
    network = _FakeNetwork()
    digest = hashlib.sha256(config.dataset_path.read_bytes()).hexdigest()

    with (
        patch("luna.pgn_pretraining.load_pgn_dataset", return_value=dataset),
        patch("luna.pgn_pretraining.LunaNetwork", return_value=network),
        patch("luna.pgn_pretraining.wandb.run", None),
    ):
        result = run_pgn_pretraining(config)

    assert result.global_step == 5
    assert network.train_requests == [(2, 5), (2, 5), (1, 5)]
    assert network.initialized_from == (str(config.source_checkpoint.parent.resolve()), config.source_checkpoint.name)
    assert [name for name, _metadata in network.saved] == [
        "pretrain_step_00000002.pth.tar",
        "latest.pth.tar",
        "pretrain_step_00000004.pth.tar",
        "latest.pth.tar",
        "pretrain_step_00000005.pth.tar",
        "latest.pth.tar",
    ]
    metadata = network.saved[-1][1]
    assert metadata is not None
    provenance = metadata[CHECKPOINT_METADATA_KEY]
    assert isinstance(provenance, dict)
    assert provenance["dataset_sha256"] == digest
    assert provenance["dataset_license"] == "CC BY-SA 4.0"
    assert provenance["seed"] == 0
    assert provenance["wandb_run_id"] is None


def test_keyboard_interrupt_publishes_resumable_state(
    tmp_path: Path,
    make_trajectory: Callable[[int], Trajectory],
) -> None:
    config = _config(tmp_path)
    network = _FakeNetwork(interrupt=True)
    with (
        patch("luna.pgn_pretraining.load_pgn_dataset", return_value=_dataset(make_trajectory(2))),
        patch("luna.pgn_pretraining.LunaNetwork", return_value=network),
        patch("luna.pgn_pretraining.wandb.run", None),
        pytest.raises(KeyboardInterrupt),
    ):
        run_pgn_pretraining(config)

    assert [name for name, _metadata in network.saved] == [
        "pretrain_step_00000001.pth.tar",
        "latest.pth.tar",
    ]


def _write_checkpoint(path: Path, step: int, metadata: dict[str, object] | None = None) -> None:
    payload: dict[str, object] = {"format_version": 2, "global_step": step}
    if metadata is not None:
        payload[CHECKPOINT_METADATA_KEY] = metadata
    torch.save(payload, path)


def test_completed_numbered_resume_republishes_latest_alias(
    tmp_path: Path,
    make_trajectory: Callable[[int], Trajectory],
) -> None:
    config = _config(tmp_path)
    output = config.output_dir
    output.mkdir()
    numbered = output / "pretrain_step_00000005.pth.tar"
    _write_checkpoint(numbered, 5, {})
    resumed = replace(
        config,
        source_checkpoint=None,
        resume_checkpoint=output / "latest.pth.tar",
        wandb_resume="allow",
    )
    network = _FakeNetwork(resume_step=5)

    with (
        patch("luna.pgn_pretraining.load_pgn_dataset", return_value=_dataset(make_trajectory(2))),
        patch("luna.pgn_pretraining.LunaNetwork", return_value=network),
        patch("luna.pgn_pretraining.validate_resume_contract"),
        patch("luna.pgn_pretraining.wandb.run", None),
    ):
        result = run_pgn_pretraining(resumed)

    assert result.global_step == 5
    assert [name for name, _metadata in network.saved] == ["latest.pth.tar"]


def test_resume_seed_changes_with_training_progress() -> None:
    assert _resume_seed(0, 1_000) == _resume_seed(0, 1_000)
    assert _resume_seed(0, 1_000) != _resume_seed(0, 2_000)


def test_network_exposes_read_only_training_progress(
    chess_game: object, small_learner_config: EzV2LearnerConfig
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    assert network.global_step == 0
    assert network.trainer_iteration == 0
    with pytest.raises(AttributeError):
        network.global_step = 1


def test_cli_reports_keyboard_interrupt_as_safe_stop() -> None:
    with (
        patch.object(pretrain_pgn.tyro, "cli", return_value=PgnPretrainingConfig()),
        patch.object(pretrain_pgn, "run_pgn_pretraining", side_effect=KeyboardInterrupt),
    ):
        assert pretrain_pgn.main() == 130


def test_makefile_pins_restart_safe_pgn_experiments() -> None:
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")

    expected_contract = (
        "714d0eb99f99fca8d791142038b6c59b5ca6a51b3339bd3891a92f4bdffcbf0c",
        "pretrain-pgn:",
        "eval-pgn-warmstart:",
        "train-pgn-warmstart:",
        "--checkpoint-top-k 10",
        "PGN_SELECTED_CHECKPOINT",
        "--dataset.max-positions 300000",
        "--learner.dataloader-workers 4",
        "luna-balanced-pgn-pretrain-v1",
        "luna-balanced-ezv2-pgn-warmstart-v1",
        "wandb_resume=never",
        "wandb_resume=must",
        "--wandb-resume must",
        "--wandb-resume never",
    )
    assert all(fragment in makefile for fragment in expected_contract)
