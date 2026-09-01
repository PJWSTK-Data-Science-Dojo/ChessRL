"""Regression tests for EfficientZeroV2 training loop."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from luna.config import EzV2LearnerConfig
from luna.game.chess_game import ACTION_SIZE, OBS_PLANES, ChessGame
from luna.network import (
    LunaNetwork,
)
from luna.replay_buffer import Trajectory


def _make_trajectory(length: int = 4) -> Trajectory:
    return Trajectory(
        observations=[np.random.randn(8, 8, OBS_PLANES).astype(np.float32) for _ in range(length)],
        actions=[np.random.randint(0, min(256, ACTION_SIZE)) for _ in range(length)],
        rewards=np.zeros(length, dtype=np.float32),
        root_policies=[np.full(ACTION_SIZE, 1.0 / ACTION_SIZE, dtype=np.float32) for _ in range(length)],
        root_values=np.zeros(length, dtype=np.float32),
        valids=np.ones((length, ACTION_SIZE), dtype=np.float32),
    )


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
