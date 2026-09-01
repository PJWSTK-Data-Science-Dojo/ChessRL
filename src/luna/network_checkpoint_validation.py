"""Checkpoint metadata validation and transactional state restoration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, cast

import torch

from luna.config import MODEL_NAMES, EzV2LearnerConfig, ModelName
from luna.network_checkpoint_io import TRAINING_PHASE_PROVENANCE_FIELD
from luna.network_checkpoint_state import clone_state_to_cpu, validate_finite_state, validate_grad_scaler_state
from luna.network_types import NetworkRuntime, TrainingPhaseProvenance, ValidatedCheckpoint

RUNTIME_LEARNER_FIELDS = frozenset({"device", "cuda_device", "compile_inference", "compile_training"})
MODEL_LEARNER_FIELDS = frozenset(
    {"model_name", "num_channels", "support_size", "repr_blocks", "dyn_blocks", "proj_dim"}
)


def checkpoint_counter(
    checkpoint: Mapping[str, Any],
    name: str,
    filepath: str | os.PathLike[str],
) -> int:
    value: object = checkpoint[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Checkpoint field '{name}' must be a non-negative integer: {filepath}")
    return value


def checkpoint_training_phase_provenance(
    checkpoint: Mapping[str, Any],
    filepath: str | os.PathLike[str],
) -> TrainingPhaseProvenance | None:
    if TRAINING_PHASE_PROVENANCE_FIELD not in checkpoint:
        return None
    raw = checkpoint[TRAINING_PHASE_PROVENANCE_FIELD]
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ValueError(f"Checkpoint training_phase_provenance must be a string-keyed mapping: {filepath}")
    _validate_provenance_fields(raw, filepath)
    digest = raw["source_checkpoint_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            "Checkpoint training_phase_provenance source_checkpoint_sha256 "
            f"must be 64 lowercase hexadecimal characters: {filepath}"
        )
    return TrainingPhaseProvenance(
        source_checkpoint_sha256=digest,
        source_trainer_iteration=checkpoint_counter(raw, "source_trainer_iteration", filepath),
        source_global_step=checkpoint_counter(raw, "source_global_step", filepath),
    )


def _validate_provenance_fields(raw: Mapping[str, Any], filepath: str | os.PathLike[str]) -> None:
    expected = {"source_checkpoint_sha256", "source_trainer_iteration", "source_global_step"}
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"Checkpoint training_phase_provenance fields are invalid: {filepath} "
            f"(missing={missing}, unexpected={unexpected})."
        )


def checkpoint_learner_config(
    checkpoint: Mapping[str, Any],
    filepath: str | os.PathLike[str],
) -> dict[str, Any]:
    raw = checkpoint["learner_config"]
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ValueError(f"Checkpoint learner_config must be a string-keyed mapping: {filepath}")
    stored = dict(raw)
    model_spec = checkpoint.get("model_spec")
    stored_model_name = model_spec.get("model_name", "baseline") if isinstance(model_spec, dict) else None
    if "reconstruction_loss_weight" not in stored and stored_model_name != "balanced_reconstruction":
        stored["reconstruction_loss_weight"] = 0.0
    expected = {field.name for field in fields(EzV2LearnerConfig)} - {"model_name"}
    actual = set(stored)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"Checkpoint learner_config does not match format version 2: {filepath} "
            f"(missing={missing}, unexpected={unexpected})."
        )
    return cast(dict[str, Any], stored)


def checkpoint_model_name(checkpoint: Mapping[str, Any], filepath: str | os.PathLike[str]) -> ModelName:
    model_spec = checkpoint.get("model_spec")
    if not isinstance(model_spec, dict):
        raise ValueError(f"Checkpoint is missing model_spec metadata: {filepath}")
    model_name = model_spec.get("model_name", "baseline")
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Checkpoint has unknown model_name {model_name!r}: {filepath}")
    return cast(ModelName, model_name)


def validate_learner_config(
    network: NetworkRuntime,
    checkpoint: Mapping[str, Any],
    filepath: str | os.PathLike[str],
) -> None:
    stored = checkpoint_learner_config(checkpoint, filepath)
    current = asdict(network._learner)
    mismatched = sorted(name for name in stored if name not in RUNTIME_LEARNER_FIELDS and stored[name] != current[name])
    if checkpoint_model_name(checkpoint, filepath) != network._learner.model_name:
        mismatched.append("model_name")
    if mismatched:
        raise ValueError(f"Checkpoint learner configuration differs in fields {mismatched}: {filepath}")


def validate_phase_model_config(
    network: NetworkRuntime,
    checkpoint: Mapping[str, Any],
    filepath: str | os.PathLike[str],
) -> None:
    stored = checkpoint_learner_config(checkpoint, filepath)
    current = asdict(network._learner)
    stored_model = {**stored, "model_name": checkpoint_model_name(checkpoint, filepath)}
    mismatched = sorted(name for name in MODEL_LEARNER_FIELDS if stored_model[name] != current[name])
    if mismatched:
        raise ValueError(f"Checkpoint model configuration differs in fields {mismatched}: {filepath}")


def validate_phase_state_dict(
    network: NetworkRuntime,
    state_dict: Mapping[str, torch.Tensor],
    filepath: str | os.PathLike[str],
) -> None:
    expected = network.nnet.state_dict()
    missing = sorted(expected.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - expected.keys())
    incompatible = sorted(
        name
        for name in expected.keys() & state_dict.keys()
        if expected[name].shape != state_dict[name].shape or expected[name].dtype != state_dict[name].dtype
    )
    if missing or unexpected or incompatible:
        raise ValueError(
            f"Checkpoint model state does not strictly match the configured network: {filepath} "
            f"(missing={missing}, unexpected={unexpected}, incompatible={incompatible})."
        )


def validate_checkpoint_state(
    network: NetworkRuntime,
    checkpoint: Mapping[str, Any],
    filepath: str | os.PathLike[str],
) -> ValidatedCheckpoint:
    counters = _checkpoint_counters(checkpoint, filepath)
    if not isinstance(checkpoint["optimizer"], dict) or not isinstance(checkpoint["scaler"], dict):
        raise ValueError(f"Checkpoint optimizer and scaler states must be mappings: {filepath}")
    _validate_game_spec(network, checkpoint, filepath)
    state_dict = _validated_state_dict(checkpoint, filepath)
    validate_finite_state(state_dict, "checkpoint.state_dict")
    validate_finite_state(checkpoint["optimizer"], "checkpoint.optimizer")
    validate_finite_state(checkpoint["scaler"], "checkpoint.scaler")
    validate_grad_scaler_state(checkpoint["scaler"])
    return ValidatedCheckpoint(
        state_dict,
        counters[0],
        counters[1],
        counters[2],
        checkpoint_training_phase_provenance(checkpoint, filepath),
    )


def _checkpoint_counters(
    checkpoint: Mapping[str, Any],
    filepath: str | os.PathLike[str],
) -> tuple[int, int, int]:
    return (
        checkpoint_counter(checkpoint, "global_step", filepath),
        checkpoint_counter(checkpoint, "trainer_iteration", filepath),
        checkpoint_counter(checkpoint, "lr_schedule_total_steps", filepath),
    )


def _validate_game_spec(
    network: NetworkRuntime,
    checkpoint: Mapping[str, Any],
    filepath: str | os.PathLike[str],
) -> None:
    model_spec = checkpoint.get("model_spec")
    expected_shape = [network.board_x, network.board_y, network.board_z]
    if not isinstance(model_spec, dict):
        raise ValueError(f"Checkpoint is missing model_spec metadata: {filepath}")
    if model_spec.get("action_size") != network.action_size or model_spec.get("observation_shape") != expected_shape:
        raise ValueError(
            f"Checkpoint model specification does not match this game: {filepath} "
            f"(expected action_size={network.action_size}, observation_shape={expected_shape})."
        )


def _validated_state_dict(
    checkpoint: Mapping[str, Any],
    filepath: str | os.PathLike[str],
) -> dict[str, torch.Tensor]:
    raw = checkpoint.get("state_dict")
    if not isinstance(raw, dict) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor) for name, tensor in raw.items()
    ):
        raise ValueError(f"Checkpoint state_dict must map string names to tensors: {filepath}")
    return normalize_compiled_state_dict(cast(dict[str, torch.Tensor], raw))


def restore_training_state(
    network: NetworkRuntime,
    checkpoint: Mapping[str, Any],
    filepath: str | os.PathLike[str],
) -> None:
    try:
        network.optimizer.load_state_dict(checkpoint["optimizer"])
        network.scaler.load_state_dict(checkpoint["scaler"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Checkpoint training state is incompatible: {filepath}") from exc


def restore_checkpoint(
    network: NetworkRuntime,
    checkpoint: dict[str, Any],
    filepath: str | os.PathLike[str],
    *,
    load_optimizer: bool,
) -> None:
    validate_learner_config(network, checkpoint, filepath)
    validated = validate_checkpoint_state(network, checkpoint, filepath)
    previous = _capture_restore_state(network, load_optimizer)
    try:
        _apply_checkpoint(network, checkpoint, validated, filepath, load_optimizer)
    except (KeyError, RuntimeError, TypeError, ValueError) as restore_error:
        _rollback_checkpoint(network, previous, load_optimizer, restore_error)
        raise


def _capture_restore_state(network: NetworkRuntime, load_optimizer: bool) -> dict[str, object]:
    return {
        "model": {name: tensor.detach().cpu().clone() for name, tensor in network.nnet.state_dict().items()},
        "optimizer": clone_state_to_cpu(network.optimizer.state_dict()) if load_optimizer else None,
        "scaler": deepcopy(network.scaler.state_dict()) if load_optimizer else None,
        "global_step": network._global_step,
        "trainer_iteration": network._trainer_iteration,
        "lr_schedule_total_steps": network._lr_schedule_total_steps,
        "loaded_checkpoint_path": network._loaded_checkpoint_path,
        "training_phase_provenance": network._training_phase_provenance,
    }


def _apply_checkpoint(
    network: NetworkRuntime,
    checkpoint: Mapping[str, Any],
    validated: ValidatedCheckpoint,
    filepath: str | os.PathLike[str],
    load_optimizer: bool,
) -> None:
    try:
        network.nnet.load_state_dict(validated.state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint architecture does not match the configured network: {filepath}. "
            "Construct it with LunaNetwork.from_checkpoint() or use matching learner settings."
        ) from exc
    if load_optimizer:
        restore_training_state(network, checkpoint, filepath)
    network._global_step = validated.global_step
    network._trainer_iteration = validated.trainer_iteration
    network._lr_schedule_total_steps = validated.lr_schedule_total_steps
    network._loaded_checkpoint_path = Path(filepath).expanduser().resolve()
    network._training_phase_provenance = validated.training_phase_provenance


def _rollback_checkpoint(
    network: NetworkRuntime,
    previous: Mapping[str, object],
    load_optimizer: bool,
    restore_error: BaseException,
) -> None:
    model = previous["model"]
    if not isinstance(model, dict):
        raise RuntimeError("Checkpoint rollback model state is invalid") from restore_error
    network.nnet.load_state_dict(model, strict=True)
    if load_optimizer:
        optimizer = previous["optimizer"]
        scaler = previous["scaler"]
        if not isinstance(optimizer, dict) or not isinstance(scaler, dict):
            raise RuntimeError("Checkpoint rollback state is invalid") from restore_error
        network.optimizer.load_state_dict(optimizer)
        network.scaler.load_state_dict(scaler)
    network._global_step = cast(int, previous["global_step"])
    network._trainer_iteration = cast(int, previous["trainer_iteration"])
    network._lr_schedule_total_steps = cast(int, previous["lr_schedule_total_steps"])
    network._loaded_checkpoint_path = cast(Path | None, previous["loaded_checkpoint_path"])
    network._training_phase_provenance = cast(
        TrainingPhaseProvenance | None,
        previous["training_phase_provenance"],
    )


def normalize_compiled_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key.replace("._orig_mod.", ".").removeprefix("_orig_mod.")
        if normalized_key in normalized:
            raise ValueError(f"Checkpoint state_dict contains duplicate normalized key {normalized_key!r}")
        normalized[normalized_key] = value
    return normalized
