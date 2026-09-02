from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import wandb
from loguru import logger

from luna.game.chess_game import ChessGame
from luna.lc0_dataset import Lc0Batch, Lc0DatasetConfig, iter_lc0_batches
from luna.lc0_pretraining_config import Lc0PretrainingConfig
from luna.network import LunaNetwork


@dataclass(frozen=True, slots=True)
class Lc0ValidationMetrics:
    policy_cross_entropy: float
    policy_top1: float
    policy_top5_mass: float
    value_cross_entropy: float
    value_mae: float
    positions: int

    def as_wandb(self, global_step: int) -> dict[str, float | int]:
        return {
            "global_step": global_step,
            "validation/policy_cross_entropy": self.policy_cross_entropy,
            "validation/policy_top1": self.policy_top1,
            "validation/policy_top5_mass": self.policy_top5_mass,
            "validation/value_cross_entropy": self.value_cross_entropy,
            "validation/value_mae": self.value_mae,
            "validation/positions": self.positions,
        }


@dataclass(frozen=True, slots=True)
class Lc0TrainingMetrics:
    policy_loss: float
    value_loss: float
    total_loss: float
    positions: int


@dataclass(frozen=True, slots=True)
class Lc0ValidationPlan:
    dataset_path: Path
    dataset: Lc0DatasetConfig
    game: ChessGame
    batch_size: int
    positions: int


def evaluate_lc0_validation(network: LunaNetwork, plan: Lc0ValidationPlan) -> Lc0ValidationMetrics:
    config = replace(
        plan.dataset,
        split="validation",
        epoch=0,
        batch_size=plan.batch_size,
        max_samples=plan.positions,
    )
    totals = np.zeros(5, dtype=np.float64)
    positions = 0
    modes = _capture_modes(network)
    network.nnet.eval()
    try:
        with torch.no_grad():
            for batch in iter_lc0_batches(plan.dataset_path, config, plan.game):
                totals += _validation_batch(network, batch)
                positions += len(batch.observations)
    finally:
        _restore_modes(network, modes)
    if positions == 0:
        raise ValueError("LC0 validation split contains no accepted samples")
    averages = totals / positions
    return Lc0ValidationMetrics(
        policy_cross_entropy=float(averages[0]),
        policy_top1=float(averages[1]),
        policy_top5_mass=float(averages[2]),
        value_cross_entropy=float(averages[3]),
        value_mae=float(averages[4]),
        positions=positions,
    )


def initialize_lc0_wandb(
    network: LunaNetwork,
    config: Lc0PretrainingConfig,
    dataset_metadata: dict[str, object],
) -> bool:
    if config.wandb_project is None:
        return False
    if config.wandb_run_id is None or config.wandb_run_name is None:
        raise RuntimeError("Validated W&B identity is unexpectedly missing")
    provenance = network.training_phase_provenance
    wandb.init(
        project=config.wandb_project,
        id=config.wandb_run_id,
        name=config.wandb_run_name,
        resume=config.wandb_resume,
        config={
            "pretraining_kind": dataset_metadata["pretraining_kind"],
            "train_scope": config.train_scope,
            "learner": asdict(config.learner),
            "dataset": dataset_metadata,
            "total_steps": config.total_steps,
            "chunk_steps": config.chunk_steps,
            "training_phase_provenance": provenance.as_config() if provenance is not None else None,
        },
        tags=["chess", "ezv2", "lc0-pretraining"],
    )
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("validation/*", step_metric="global_step")
    return True


def log_lc0_training(step: int, metrics: Lc0TrainingMetrics) -> None:
    logger.info("LC0 train step {}: {}", step, metrics)
    if wandb.run is not None:
        wandb.log(
            {
                "global_step": step,
                "train/policy_cross_entropy": metrics.policy_loss,
                "train/value_cross_entropy": metrics.value_loss,
                "train/loss_total": metrics.total_loss,
                "train/positions": metrics.positions,
            }
        )


def log_lc0_validation(step: int, metrics: Lc0ValidationMetrics) -> None:
    logger.info("LC0 validation step {}: {}", step, metrics)
    if wandb.run is not None:
        wandb.log(metrics.as_wandb(step))


def _validation_batch(network: LunaNetwork, batch: Lc0Batch) -> np.ndarray:
    validate_lc0_batch(batch, network)
    observations = torch.as_tensor(batch.observations, device=network.device, dtype=torch.float32)
    valid_moves = torch.as_tensor(batch.valid_moves, device=network.device, dtype=torch.float32)
    policies = torch.as_tensor(batch.policies, device=network.device, dtype=torch.float32)
    values = torch.as_tensor(batch.value_targets, device=network.device, dtype=torch.float32)
    with torch.autocast(
        "cuda",
        enabled=network._learner.mixed_precision and network.device.type == "cuda",
        dtype=network._amp_dtype,
    ):
        _latent, log_policy, value_logits = network._training_initial_inference(observations, valid_moves)
    value_log_probs = torch.log_softmax(value_logits.float(), dim=1)
    predicted_top = log_policy.float().topk(min(5, network.action_size), dim=1).indices
    target_top = policies.argmax(dim=1)
    support = torch.tensor([-1.0, 0.0, 1.0], device=network.device)
    predicted_value = torch.softmax(value_logits.float(), dim=1) @ support
    target_value = values @ support
    return np.asarray(
        [
            float((-(policies * log_policy.float()).sum(dim=1)).sum().item()),
            float((predicted_top[:, 0] == target_top).sum().item()),
            float(policies.gather(1, predicted_top).sum().item()),
            float((-(values * value_log_probs).sum(dim=1)).sum().item()),
            float((predicted_value - target_value).abs().sum().item()),
        ],
        dtype=np.float64,
    )


def validate_lc0_batch(batch: Lc0Batch, network: LunaNetwork) -> int:
    size = len(batch.observations)
    expected_observations = (size, network.board_x, network.board_y, network.board_z)
    if size <= 0 or batch.observations.shape != expected_observations:
        raise ValueError(f"LC0 observations must have shape {expected_observations}, got {batch.observations.shape}")
    expected_policy = (size, network.action_size)
    if batch.policies.shape != expected_policy or batch.valid_moves.shape != expected_policy:
        raise ValueError("LC0 policy targets and legal masks do not match the network action space")
    if batch.value_targets.shape != (size, 3):
        raise ValueError(f"LC0 WDL targets must have shape {(size, 3)}, got {batch.value_targets.shape}")
    return size


def _capture_modes(network: LunaNetwork) -> tuple[bool, bool, bool]:
    return (
        network.nnet.training,
        network.nnet.prediction.policy_head.training,
        network.nnet.prediction.value_head.training,
    )


def _restore_modes(network: LunaNetwork, modes: tuple[bool, bool, bool]) -> None:
    network.nnet.train(modes[0])
    network.nnet.prediction.policy_head.train(modes[1])
    network.nnet.prediction.value_head.train(modes[2])
