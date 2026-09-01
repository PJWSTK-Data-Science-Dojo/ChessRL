"""Regression tests for EfficientZeroV2 training loop."""

from dataclasses import replace
from hashlib import file_digest
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.amp import GradScaler

from luna.balanced_networks import BalancedNetworks
from luna.config import EzV2LearnerConfig
from luna.game.chess_game import ACTION_SIZE, OBS_PLANES, ChessGame
from luna.network import (
    LunaNetwork,
    TrainingPhaseProvenance,
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


def test_checkpoint_reconstructs_state_anchored_model_from_factory_metadata(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    config = replace(
        small_learner_config,
        model_name="balanced_reconstruction",
        reconstruction_loss_weight=0.5,
    )
    network = LunaNetwork(chess_game, config)
    network.save_checkpoint(str(tmp_path), "state-anchored.pth.tar")

    restored = LunaNetwork.from_checkpoint(
        chess_game,
        tmp_path / "state-anchored.pth.tar",
        device="cpu",
    )

    assert restored._learner.model_name == "balanced_reconstruction"
    assert restored._learner.reconstruction_loss_weight == pytest.approx(0.5)
    assert isinstance(restored.nnet, BalancedNetworks)
    assert restored.nnet.piece_reconstruction is not None
    for name, tensor in network.nnet.state_dict().items():
        torch.testing.assert_close(tensor, restored.nnet.state_dict()[name])


def test_checkpoint_loader_defaults_missing_reconstruction_weight_to_zero(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    network.save_checkpoint(str(tmp_path), "current.pth.tar")
    checkpoint = torch.load(tmp_path / "current.pth.tar", map_location="cpu", weights_only=True)
    del checkpoint["learner_config"]["reconstruction_loss_weight"]
    torch.save(checkpoint, tmp_path / "pre-reconstruction-objective.pth.tar")

    restored = LunaNetwork.from_checkpoint(
        chess_game,
        tmp_path / "pre-reconstruction-objective.pth.tar",
        device="cpu",
    )

    assert restored._learner.reconstruction_loss_weight == pytest.approx(0.0)


def test_checkpoint_loader_rejects_missing_reconstruction_weight_for_state_anchored_model(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    config = replace(
        small_learner_config,
        model_name="balanced_reconstruction",
        reconstruction_loss_weight=0.5,
    )
    network = LunaNetwork(chess_game, config)
    network.save_checkpoint(str(tmp_path), "state-anchored.pth.tar")
    checkpoint = torch.load(tmp_path / "state-anchored.pth.tar", map_location="cpu", weights_only=True)
    del checkpoint["learner_config"]["reconstruction_loss_weight"]
    torch.save(checkpoint, tmp_path / "missing-state-anchor-weight.pth.tar")

    with pytest.raises(ValueError, match=r"missing=.*reconstruction_loss_weight"):
        LunaNetwork.from_checkpoint(
            chess_game,
            tmp_path / "missing-state-anchor-weight.pth.tar",
            device="cpu",
        )


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
