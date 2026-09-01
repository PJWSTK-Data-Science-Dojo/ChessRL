"""Atomic checkpoint persistence and safe format-v2 loading."""

from __future__ import annotations

import os
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import torch

from luna.network_checkpoint_state import validate_finite_state, validate_grad_scaler_state
from luna.network_types import NetworkRuntime

FORMAT_VERSION = 2
TRAINING_PHASE_PROVENANCE_FIELD = "training_phase_provenance"
REQUIRED_FIELDS = frozenset(
    {
        "state_dict",
        "optimizer",
        "scaler",
        "global_step",
        "trainer_iteration",
        "lr_schedule_total_steps",
        "learner_config",
        "model_spec",
    }
)


def save_checkpoint(
    network: NetworkRuntime,
    folder: str,
    filename: str,
    extra_state: dict[str, object] | None,
) -> None:
    filepath = os.path.join(folder, filename)
    output_dir = os.path.dirname(filepath) or "."
    os.makedirs(output_dir, exist_ok=True)
    payload = _checkpoint_payload(network, extra_state)
    validate_finite_state(payload, "checkpoint")
    temporary_path = f"{filepath}.tmp-{os.getpid()}"
    try:
        _write_checkpoint_file(temporary_path, payload)
        os.replace(temporary_path, filepath)
        _fsync_directory(output_dir)
    finally:
        _remove_temporary_file(temporary_path)


def _remove_temporary_file(filepath: str) -> None:
    try:
        os.unlink(filepath)
    except FileNotFoundError:
        return


def _checkpoint_payload(
    network: NetworkRuntime,
    extra_state: dict[str, object] | None,
) -> dict[str, object]:
    scaler_state = network.scaler.state_dict()
    validate_grad_scaler_state(scaler_state)
    learner_config = asdict(network._learner)
    learner_config.pop("model_name")
    payload: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "state_dict": network.nnet.state_dict(),
        "optimizer": network.optimizer.state_dict(),
        "scaler": scaler_state,
        "global_step": network._global_step,
        "trainer_iteration": network._trainer_iteration,
        "lr_schedule_total_steps": network._lr_schedule_total_steps,
        "learner_config": learner_config,
        "model_spec": {
            "model_name": network._learner.model_name,
            "action_size": network.action_size,
            "observation_shape": [network.board_x, network.board_y, network.board_z],
        },
    }
    if network._training_phase_provenance is not None:
        payload[TRAINING_PHASE_PROVENANCE_FIELD] = network._training_phase_provenance.as_config()
    if extra_state:
        reserved = (set(payload) | {TRAINING_PHASE_PROVENANCE_FIELD}) & extra_state.keys()
        if reserved:
            raise ValueError(f"extra_state cannot replace reserved checkpoint fields: {sorted(reserved)}")
        payload.update(extra_state)
    return payload


def _write_checkpoint_file(filepath: str, payload: dict[str, object]) -> None:
    with open(filepath, "wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(directory: str) -> None:
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_checkpoint(filepath: str | os.PathLike[str]) -> dict[str, Any]:
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)
    return validate_checkpoint_payload(checkpoint, filepath)


def read_checkpoint_with_sha256(
    filepath: str | os.PathLike[str],
) -> tuple[dict[str, Any], str]:
    digest = sha256()
    with open(filepath, "rb") as checkpoint_stream:
        while chunk := checkpoint_stream.read(1024 * 1024):
            digest.update(chunk)
        checkpoint_stream.seek(0)
        checkpoint = torch.load(checkpoint_stream, map_location="cpu", weights_only=True)
    return validate_checkpoint_payload(checkpoint, filepath), digest.hexdigest()


def validate_checkpoint_payload(
    checkpoint: object,
    filepath: str | os.PathLike[str],
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint payload is not a mapping: {filepath}")
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format in {filepath}; only format version 2 is accepted.")
    missing = sorted(REQUIRED_FIELDS - checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint is missing required fields {missing}: {filepath}")
    return cast(dict[str, Any], checkpoint)


def checkpoint_path(folder: str, filename: str) -> Path:
    path = Path(folder) / filename
    if not path.exists():
        raise FileNotFoundError(f"No model in path {path}")
    return path
