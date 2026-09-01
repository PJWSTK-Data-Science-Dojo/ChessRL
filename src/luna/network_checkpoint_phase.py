"""Transactional initialization of a fresh training phase from model weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.optim as optim

from luna.network_checkpoint_io import checkpoint_path, read_checkpoint_with_sha256
from luna.network_checkpoint_validation import (
    validate_checkpoint_state,
    validate_phase_model_config,
    validate_phase_state_dict,
)
from luna.network_types import NetworkRuntime, TrainingPhaseProvenance


@dataclass(frozen=True, slots=True)
class PhaseRollbackState:
    model: dict[str, torch.Tensor]
    optimizer: optim.AdamW
    scaler: torch.GradScaler
    global_step: int
    trainer_iteration: int
    lr_schedule_total_steps: int
    lr_schedule_mismatch_warned: bool
    loaded_checkpoint_path: Path | None
    provenance: TrainingPhaseProvenance | None


def initialize_training_phase(
    network: NetworkRuntime,
    folder: str,
    filename: str,
) -> None:
    path = checkpoint_path(folder, filename)
    checkpoint, digest = read_checkpoint_with_sha256(path)
    validate_phase_model_config(network, checkpoint, path)
    validated = validate_checkpoint_state(network, checkpoint, path)
    validate_phase_state_dict(network, validated.state_dict, path)
    provenance = TrainingPhaseProvenance(
        source_checkpoint_sha256=digest,
        source_trainer_iteration=validated.trainer_iteration,
        source_global_step=validated.global_step,
    )
    previous = _capture_state(network)
    try:
        _start_phase(network, validated.state_dict, provenance)
    except (KeyError, RuntimeError, TypeError, ValueError):
        _restore_state(network, previous)
        raise


def _capture_state(network: NetworkRuntime) -> PhaseRollbackState:
    return PhaseRollbackState(
        model={name: tensor.detach().cpu().clone() for name, tensor in network.nnet.state_dict().items()},
        optimizer=network.optimizer,
        scaler=network.scaler,
        global_step=network._global_step,
        trainer_iteration=network._trainer_iteration,
        lr_schedule_total_steps=network._lr_schedule_total_steps,
        lr_schedule_mismatch_warned=network._lr_schedule_mismatch_warned,
        loaded_checkpoint_path=network._loaded_checkpoint_path,
        provenance=network._training_phase_provenance,
    )


def _start_phase(
    network: NetworkRuntime,
    state_dict: dict[str, torch.Tensor],
    provenance: TrainingPhaseProvenance,
) -> None:
    network.nnet.load_state_dict(state_dict, strict=True)
    network.optimizer = network._new_optimizer()
    network.scaler = network._new_grad_scaler()
    network._global_step = 0
    network._trainer_iteration = 0
    network._lr_schedule_total_steps = 0
    network._lr_schedule_mismatch_warned = False
    network._loaded_checkpoint_path = None
    network._training_phase_provenance = provenance


def _restore_state(network: NetworkRuntime, previous: PhaseRollbackState) -> None:
    network.nnet.load_state_dict(previous.model, strict=True)
    network.optimizer = previous.optimizer
    network.scaler = previous.scaler
    network._global_step = previous.global_step
    network._trainer_iteration = previous.trainer_iteration
    network._lr_schedule_total_steps = previous.lr_schedule_total_steps
    network._lr_schedule_mismatch_warned = previous.lr_schedule_mismatch_warned
    network._loaded_checkpoint_path = previous.loaded_checkpoint_path
    network._training_phase_provenance = previous.provenance
