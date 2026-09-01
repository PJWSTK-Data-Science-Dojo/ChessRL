"""Regression tests for EfficientZeroV2 training loop."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.amp import GradScaler

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
