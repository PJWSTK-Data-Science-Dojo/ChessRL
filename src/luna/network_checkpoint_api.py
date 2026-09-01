"""Checkpoint-facing methods composed into :class:`luna.network.LunaNetwork`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Self, cast

import torch

from luna.config import EzV2LearnerConfig, ModelName
from luna.game.chess_game import ChessGame
from luna.network_checkpoint_io import (
    checkpoint_path as resolve_checkpoint_path,
)
from luna.network_checkpoint_io import (
    read_checkpoint,
    read_checkpoint_with_sha256,
    save_checkpoint,
    validate_checkpoint_payload,
)
from luna.network_checkpoint_phase import initialize_training_phase
from luna.network_checkpoint_validation import (
    checkpoint_counter,
    checkpoint_learner_config,
    checkpoint_model_name,
    checkpoint_training_phase_provenance,
    normalize_compiled_state_dict,
    restore_checkpoint,
    restore_training_state,
    validate_checkpoint_state,
    validate_learner_config,
    validate_phase_model_config,
    validate_phase_state_dict,
)
from luna.network_types import NetworkRuntime, TrainingPhaseProvenance, ValidatedCheckpoint


class NetworkCheckpointMixin:
    def __init__(self, game: ChessGame, learner: EzV2LearnerConfig | None = None) -> None:
        raise NotImplementedError

    def save_checkpoint(
        self,
        folder: str = "checkpoint",
        filename: str = "checkpoint.pth.tar",
        *,
        extra_state: dict[str, object] | None = None,
    ) -> None:
        save_checkpoint(self._runtime(), folder, filename, extra_state)

    def load_checkpoint(
        self,
        folder: str = "checkpoint",
        filename: str = "checkpoint.pth.tar",
        *,
        load_optimizer: bool = True,
    ) -> None:
        path = resolve_checkpoint_path(folder, filename)
        checkpoint = self._read_checkpoint(path)
        self._restore_checkpoint(checkpoint, path, load_optimizer=load_optimizer)

    def initialize_training_phase(
        self,
        folder: str = "checkpoint",
        filename: str = "checkpoint.pth.tar",
    ) -> None:
        initialize_training_phase(self._runtime(), folder, filename)

    @classmethod
    def _read_checkpoint(cls, filepath: str | os.PathLike[str]) -> dict[str, Any]:
        return read_checkpoint(filepath)

    @classmethod
    def _read_checkpoint_with_sha256(
        cls,
        filepath: str | os.PathLike[str],
    ) -> tuple[dict[str, Any], str]:
        return read_checkpoint_with_sha256(filepath)

    @staticmethod
    def _validate_checkpoint_payload(
        checkpoint: object,
        filepath: str | os.PathLike[str],
    ) -> dict[str, Any]:
        return validate_checkpoint_payload(checkpoint, filepath)

    @classmethod
    def checkpoint_trainer_iteration(cls, filepath: str | os.PathLike[str]) -> int:
        checkpoint = cls._read_checkpoint(filepath)
        return cls._checkpoint_counter(checkpoint, "trainer_iteration", filepath)

    @staticmethod
    def _checkpoint_counter(
        checkpoint: dict[str, Any],
        name: str,
        filepath: str | os.PathLike[str],
    ) -> int:
        return checkpoint_counter(checkpoint, name, filepath)

    @classmethod
    def _checkpoint_training_phase_provenance(
        cls,
        checkpoint: dict[str, Any],
        filepath: str | os.PathLike[str],
    ) -> TrainingPhaseProvenance | None:
        return checkpoint_training_phase_provenance(checkpoint, filepath)

    @staticmethod
    def _checkpoint_learner_config(
        checkpoint: dict[str, Any],
        filepath: str | os.PathLike[str],
    ) -> dict[str, Any]:
        return checkpoint_learner_config(checkpoint, filepath)

    @staticmethod
    def _checkpoint_model_name(
        checkpoint: dict[str, Any],
        filepath: str | os.PathLike[str],
    ) -> ModelName:
        return checkpoint_model_name(checkpoint, filepath)

    def _validate_learner_config(
        self,
        checkpoint: dict[str, Any],
        filepath: str | os.PathLike[str],
    ) -> None:
        validate_learner_config(self._runtime(), checkpoint, filepath)

    def _validate_phase_model_config(
        self,
        checkpoint: dict[str, Any],
        filepath: str | os.PathLike[str],
    ) -> None:
        validate_phase_model_config(self._runtime(), checkpoint, filepath)

    def _validate_phase_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        filepath: str | os.PathLike[str],
    ) -> None:
        validate_phase_state_dict(self._runtime(), state_dict, filepath)

    def _restore_training_state(
        self,
        checkpoint: dict[str, Any],
        filepath: str | os.PathLike[str],
    ) -> None:
        restore_training_state(self._runtime(), checkpoint, filepath)

    def _validate_checkpoint_state(
        self,
        checkpoint: dict[str, Any],
        filepath: str | os.PathLike[str],
    ) -> ValidatedCheckpoint:
        return validate_checkpoint_state(self._runtime(), checkpoint, filepath)

    def _restore_checkpoint(
        self,
        checkpoint: dict[str, Any],
        filepath: str | os.PathLike[str],
        *,
        load_optimizer: bool,
    ) -> None:
        restore_checkpoint(self._runtime(), checkpoint, filepath, load_optimizer=load_optimizer)

    @staticmethod
    def _normalize_compiled_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return normalize_compiled_state_dict(state_dict)

    @classmethod
    def from_checkpoint(
        cls,
        game: ChessGame,
        checkpoint_path: str | os.PathLike[str],
        *,
        device: str = "cuda",
        cuda_device: int | None = None,
        compile_inference: bool = False,
        load_optimizer: bool = False,
    ) -> Self:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"No model in path {path}")
        checkpoint = cls._read_checkpoint(path)
        config_values = cls._checkpoint_learner_config(checkpoint, path)
        config_values["model_name"] = cls._checkpoint_model_name(checkpoint, path)
        config_values.update(
            device=device,
            cuda_device=cuda_device,
            compile_inference=compile_inference,
            compile_training=False,
        )
        network = cls(game, EzV2LearnerConfig(**config_values))
        network._restore_checkpoint(checkpoint, path, load_optimizer=load_optimizer)
        return network

    def _runtime(self) -> NetworkRuntime:
        return cast(NetworkRuntime, self)
