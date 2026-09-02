from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

import pretrain_lc0
from luna.config import EzV2LearnerConfig
from luna.game.chess_game import ChessGame
from luna.lc0_dataset import Lc0Batch, Lc0DatasetConfig
from luna.lc0_pretraining import (
    _freeze_for_root_supervision,
    _frozen_parameter_digest,
    _microbatch_losses,
    _train_batch,
    run_lc0_pretraining,
)
from luna.lc0_pretraining_config import (
    LC0_CHECKPOINT_METADATA_KEY,
    Lc0PretrainingConfig,
    validate_lc0_online_source,
    validate_lc0_pretraining_config,
)
from luna.lc0_pretraining_validation import (
    Lc0TrainingMetrics,
    Lc0ValidationMetrics,
    _capture_modes,
    _restore_modes,
)
from luna.network import LunaNetwork
from luna.network_losses import soft_ce_with_support


def _learner() -> EzV2LearnerConfig:
    return EzV2LearnerConfig(
        device="cpu",
        model_name="balanced_reconstruction",
        num_channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        proj_dim=32,
        lr=1e-2,
        lr_min=1e-2,
        lr_warmup_steps=0,
        weight_decay=0.1,
        batch_size=2,
        grad_accum_steps=1,
        grad_clip_norm=100.0,
        unroll_steps=1,
        td_steps=0,
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        reward_loss_weight=0.0,
        consistency_loss_weight=0.0,
        reconstruction_loss_weight=0.0,
        reanalyze_mcts_sims=0,
        reanalyze_prob=0.0,
        reanalyze_policy=False,
        dataloader_workers=0,
    )


def _config(tmp_path: Path) -> Lc0PretrainingConfig:
    dataset = tmp_path / "lc0.tar"
    source = tmp_path / "source.pth.tar"
    dataset.write_bytes(b"lc0 archive")
    source.write_bytes(b"source checkpoint")
    return Lc0PretrainingConfig(
        dataset_path=dataset,
        output_dir=tmp_path / "output",
        source_checkpoint=source,
        total_steps=5,
        chunk_steps=2,
        validation_batch_size=2,
        validation_positions=2,
        dataset=Lc0DatasetConfig(batch_size=2, max_samples=10),
        learner=_learner(),
    )


def _batch(game: ChessGame) -> Lc0Batch:
    board = game.get_init_board()
    observation = game.to_array(game.get_canonical_form(board, 1))
    valid = game.get_valid_moves(board, 1).astype(np.bool_)
    legal = np.flatnonzero(valid)[:2]
    policies = np.zeros((2, game.get_action_size()), dtype=np.float32)
    policies[np.arange(2), legal] = 1.0
    return Lc0Batch(
        observations=np.stack([observation, observation]).astype(np.float32),
        policies=policies,
        value_targets=np.asarray([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]], dtype=np.float32),
        valid_moves=np.stack([valid, valid]),
        visits=np.asarray([100, 100], dtype=np.int64),
    )


@pytest.mark.parametrize(
    ("learner", "message"),
    [
        (replace(_learner(), support_size=2), "support_size=1"),
        (replace(_learner(), policy_loss_weight=0.0), "positive policy and value"),
        (replace(_learner(), value_loss_weight=0.0), "positive policy and value"),
        (replace(_learner(), reward_loss_weight=0.1), "root-only"),
        (replace(_learner(), consistency_loss_weight=0.1), "root-only"),
        (replace(_learner(), reconstruction_loss_weight=0.1), "root-only"),
        (replace(_learner(), reanalyze_prob=0.1), "disable replay reanalysis"),
    ],
)
def test_config_rejects_non_root_objectives(
    tmp_path: Path,
    learner: EzV2LearnerConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_lc0_pretraining_config(replace(_config(tmp_path), learner=learner))


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_pretraining_requires_nonempty_train_and_validation_splits(tmp_path: Path, fraction: float) -> None:
    config = _config(tmp_path)
    dataset = replace(config.dataset, validation_fraction=fraction)
    with pytest.raises(ValueError, match="validation_fraction strictly between"):
        validate_lc0_pretraining_config(replace(config, dataset=dataset))


def test_wandb_requires_explicit_identity_and_resume_semantics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="explicit wandb_run_id and wandb_run_name"):
        validate_lc0_pretraining_config(
            replace(config, wandb_project="ChessRL", wandb_run_id=None, wandb_run_name=None)
        )
    with pytest.raises(ValueError, match=r"fresh.*existing W&B"):
        validate_lc0_pretraining_config(replace(config, wandb_resume="must"))

    config.output_dir.mkdir()
    resume = config.output_dir / "latest.pth.tar"
    resume.write_bytes(b"checkpoint")
    resumed = replace(config, source_checkpoint=None, resume_checkpoint=resume)
    with pytest.raises(ValueError, match=r"resumed.*allow.*must"):
        validate_lc0_pretraining_config(resumed)


def test_pretraining_accepts_a_multi_archive_dataset(tmp_path: Path) -> None:
    config = _config(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.tar").write_bytes(b"archive")

    validate_lc0_pretraining_config(replace(config, dataset_path=corpus))


def test_online_source_requires_matching_joint_pretraining_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pth.tar"
    fingerprint = "a" * 64
    torch.save(
        {
            LC0_CHECKPOINT_METADATA_KEY: {
                "dataset_fingerprint": fingerprint,
                "train_scope": "representation_and_heads",
            }
        },
        checkpoint,
    )

    validate_lc0_online_source(checkpoint, fingerprint)

    with pytest.raises(ValueError, match="corpus fingerprint"):
        validate_lc0_online_source(checkpoint, "b" * 64)


def test_online_source_rejects_heads_only_pretraining(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pth.tar"
    torch.save(
        {
            LC0_CHECKPOINT_METADATA_KEY: {
                "dataset_fingerprint": "a" * 64,
                "train_scope": "prediction_heads",
            }
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="not jointly trained"):
        validate_lc0_online_source(checkpoint, "a" * 64)


def test_online_source_rejects_missing_pretraining_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pth.tar"
    torch.save({}, checkpoint)

    with pytest.raises(ValueError, match="no pretraining metadata"):
        validate_lc0_online_source(checkpoint, "a" * 64)


def test_optimizer_step_changes_only_policy_and_value_heads() -> None:
    game = ChessGame()
    network = LunaNetwork(game, _learner())
    before = {name: parameter.detach().clone() for name, parameter in network.nnet.named_parameters()}
    frozen_digest = _freeze_for_root_supervision(network)

    metrics = _train_batch(network, _batch(game), total_steps=1)

    changed = {
        name for name, parameter in network.nnet.named_parameters() if not torch.equal(parameter.detach(), before[name])
    }
    assert changed
    assert any(name.startswith("prediction.policy_head.") for name in changed)
    assert any(name.startswith("prediction.value_head.") for name in changed)
    assert all(name.startswith(("prediction.policy_head.", "prediction.value_head.")) for name in changed)
    assert metrics.positions == 2
    assert _frozen_parameter_digest(network) == frozen_digest


def test_joint_scope_updates_representation_and_heads_only() -> None:
    game = ChessGame()
    network = LunaNetwork(game, _learner())
    before = {name: parameter.detach().clone() for name, parameter in network.nnet.named_parameters()}
    _freeze_for_root_supervision(network, "representation_and_heads")

    _train_batch(network, _batch(game), total_steps=1)

    changed = {
        name for name, parameter in network.nnet.named_parameters() if not torch.equal(parameter.detach(), before[name])
    }
    assert any(name.startswith("representation.") for name in changed)
    assert any(name.startswith("prediction.policy_head.") for name in changed)
    assert any(name.startswith("prediction.value_head.") for name in changed)
    assert all(
        name.startswith(("representation.", "prediction.policy_head.", "prediction.value_head.")) for name in changed
    )


def test_validation_restores_joint_training_modes() -> None:
    network = LunaNetwork(ChessGame(), _learner())
    network.nnet.eval()
    network.nnet.representation.train()
    network.nnet.prediction.policy_head.train()
    network.nnet.prediction.value_head.train()
    modes = _capture_modes(network)

    network.nnet.eval()
    _restore_modes(network, modes)

    assert network.nnet.training is False
    assert network.nnet.representation.training is True
    assert network.nnet.prediction.policy_head.training is True
    assert network.nnet.prediction.value_head.training is True


def test_value_loss_consumes_exact_wdl_distribution() -> None:
    game = ChessGame()
    network = LunaNetwork(game, _learner())
    _freeze_for_root_supervision(network)
    batch = _batch(game)

    with patch("luna.lc0_pretraining.soft_ce_with_support", wraps=soft_ce_with_support) as loss:
        _microbatch_losses(network, batch, np.arange(2))

    target = loss.call_args.args[1].detach().cpu().numpy()
    np.testing.assert_array_equal(target, batch.value_targets)


class _FakeNetwork:
    def __init__(self) -> None:
        self.global_step = 0
        self.training_phase_provenance = None
        self.initialized_from: tuple[str, str] | None = None
        self.saved: list[tuple[str, dict[str, object] | None]] = []

    def initialize_training_phase(self, folder: str, filename: str) -> None:
        self.initialized_from = (folder, filename)

    def _resolve_lr_schedule_total(self, requested_total: int, current_steps: int) -> int:
        del current_steps
        return requested_total

    def save_checkpoint(
        self,
        folder: str,
        filename: str,
        *,
        extra_state: dict[str, object] | None = None,
    ) -> None:
        del folder
        self.saved.append((filename, extra_state))


def test_run_initializes_new_phase_and_publishes_numbered_checkpoints(tmp_path: Path) -> None:
    config = _config(tmp_path)
    network = _FakeNetwork()
    validation = Lc0ValidationMetrics(1.0, 0.2, 0.5, 0.8, 0.4, 2)

    def train_steps(fake: _FakeNetwork, batches: object, steps: int, total: int) -> Lc0TrainingMetrics:
        del batches, total
        fake.global_step += steps
        return Lc0TrainingMetrics(1.0, 0.8, 1.8, steps * 2)

    with (
        patch("luna.lc0_pretraining.dataset_fingerprint", return_value="a" * 64),
        patch("luna.lc0_pretraining.LunaNetwork", return_value=network),
        patch("luna.lc0_pretraining._freeze_for_root_supervision", return_value="b" * 64),
        patch("luna.lc0_pretraining._assert_frozen_parameters"),
        patch("luna.lc0_pretraining._evaluate_validation", return_value=validation),
        patch("luna.lc0_pretraining._train_steps", side_effect=train_steps),
        patch("luna.lc0_pretraining.wandb.run", None),
    ):
        result = run_lc0_pretraining(config)

    assert result.global_step == 5
    source_checkpoint = config.source_checkpoint
    assert source_checkpoint is not None
    assert network.initialized_from == (str(source_checkpoint.parent.resolve()), source_checkpoint.name)
    assert [name for name, _metadata in network.saved] == [
        "lc0_step_00000000.pth.tar",
        "latest.pth.tar",
        "lc0_step_00000000.pth.tar",
        "latest.pth.tar",
        "best.pth.tar",
        "lc0_step_00000002.pth.tar",
        "latest.pth.tar",
        "lc0_step_00000004.pth.tar",
        "latest.pth.tar",
        "lc0_step_00000005.pth.tar",
        "latest.pth.tar",
    ]
    metadata = network.saved[-1][1]
    assert metadata is not None
    lc0_metadata = metadata["lc0_pretraining"]
    assert isinstance(lc0_metadata, dict)
    assert lc0_metadata["pretraining_kind"] == "lc0_policy_value_heads"
    assert lc0_metadata["validation_objective"] == pytest.approx(1.8)


def test_keyboard_interrupt_publishes_completed_optimizer_state(tmp_path: Path) -> None:
    config = _config(tmp_path)
    network = _FakeNetwork()
    validation = Lc0ValidationMetrics(1.0, 0.2, 0.5, 0.8, 0.4, 2)

    def interrupt(fake: _FakeNetwork, batches: object, steps: int, total: int) -> Lc0TrainingMetrics:
        del batches, steps, total
        fake.global_step = 1
        raise KeyboardInterrupt

    with (
        patch("luna.lc0_pretraining.dataset_fingerprint", return_value="a" * 64),
        patch("luna.lc0_pretraining.LunaNetwork", return_value=network),
        patch("luna.lc0_pretraining._freeze_for_root_supervision", return_value="b" * 64),
        patch("luna.lc0_pretraining._assert_frozen_parameters"),
        patch("luna.lc0_pretraining._evaluate_validation", return_value=validation),
        patch("luna.lc0_pretraining._train_steps", side_effect=interrupt),
        patch("luna.lc0_pretraining.wandb.run", None),
        pytest.raises(KeyboardInterrupt),
    ):
        run_lc0_pretraining(config)

    assert [name for name, _metadata in network.saved] == [
        "lc0_step_00000000.pth.tar",
        "latest.pth.tar",
        "lc0_step_00000000.pth.tar",
        "latest.pth.tar",
        "best.pth.tar",
        "lc0_step_00000001.pth.tar",
        "latest.pth.tar",
    ]


def test_cli_reports_keyboard_interrupt_as_safe_stop() -> None:
    with (
        patch("pretrain_lc0.tyro.cli", return_value=Lc0PretrainingConfig()),
        patch.object(pretrain_lc0, "run_lc0_pretraining", side_effect=KeyboardInterrupt),
    ):
        assert pretrain_lc0.main() == 130
