import random
from dataclasses import asdict, dataclass
from hashlib import file_digest
from pathlib import Path

import numpy as np
import torch
import wandb
from loguru import logger

from luna.game.chess_game import ChessGame
from luna.network import LunaNetwork
from luna.pgn_dataset import PgnDataset, load_pgn_dataset
from luna.pgn_pretraining_checkpoints import (
    CheckpointPublication,
    publish_pretraining_checkpoints,
    resolve_pretraining_resume,
    validate_resume_contract,
)
from luna.pgn_pretraining_config import PgnPretrainingConfig, validate_pretraining_config
from luna.pgn_pretraining_validation import ValidationMetrics, ValidationPlan, evaluate_validation
from luna.replay_buffer import PrioritizedReplayBuffer


@dataclass(frozen=True, slots=True)
class PgnPretrainingResult:
    global_step: int
    validation: ValidationMetrics
    latest_checkpoint: Path


@dataclass(frozen=True, slots=True)
class _PretrainingContext:
    config: PgnPretrainingConfig
    dataset: PgnDataset
    dataset_sha256: str


def run_pgn_pretraining(config: PgnPretrainingConfig) -> PgnPretrainingResult:
    validate_pretraining_config(config)
    _seed_everything(config.seed)
    context, game = _load_context(config)
    replay = _build_replay(context.dataset)
    network = LunaNetwork(game, config.learner)
    _restore_network(network, context)
    config.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    wandb_started = _initialize_wandb(context, network)
    try:
        return _train_chunks(network, replay, context)
    finally:
        if wandb_started:
            wandb.finish()


def _load_context(config: PgnPretrainingConfig) -> tuple[_PretrainingContext, ChessGame]:
    dataset_sha256 = _sha256(config.dataset_path)
    game = ChessGame()
    dataset = load_pgn_dataset(config.dataset_path.expanduser().resolve(), config.dataset, game)
    _validate_dataset(dataset)
    return _PretrainingContext(config, dataset, dataset_sha256), game


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    with path.expanduser().open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _validate_dataset(dataset: PgnDataset) -> None:
    if not dataset.train_trajectories:
        raise ValueError("PGN dataset contains no training trajectories")
    if not dataset.validation_trajectories:
        raise ValueError("PGN dataset contains no validation trajectories")


def _build_replay(dataset: PgnDataset) -> PrioritizedReplayBuffer:
    capacity = sum(trajectory.game_length for trajectory in dataset.train_trajectories)
    replay = PrioritizedReplayBuffer(capacity=capacity, alpha=0.0, beta=0.0, beta_increment=0.0)
    for trajectory in dataset.train_trajectories:
        replay.save_trajectory(trajectory)
    return replay


def _restore_network(network: LunaNetwork, context: _PretrainingContext) -> None:
    config = context.config
    if config.resume_checkpoint is not None:
        checkpoint = resolve_pretraining_resume(config.resume_checkpoint, config.output_dir)
        _validate_resume_dataset(checkpoint, context)
        network.load_checkpoint(str(checkpoint.parent), checkpoint.name, load_optimizer=True)
        if network.global_step > config.total_steps:
            raise ValueError("resume checkpoint is beyond total_steps")
        _seed_everything(_resume_seed(config.seed, network.global_step))
        return
    source = config.source_checkpoint
    if source is None:
        raise RuntimeError("Validated source checkpoint is unexpectedly missing")
    source = source.expanduser().resolve()
    network.initialize_training_phase(str(source.parent), source.name)


def _resume_seed(seed: int, global_step: int) -> int:
    state = np.random.SeedSequence([seed, global_step]).generate_state(1)
    return int(state[0])


def _validate_resume_dataset(
    checkpoint: Path,
    context: _PretrainingContext,
) -> None:
    config = context.config
    expected: dict[str, object] = {
        "dataset_sha256": context.dataset_sha256,
        "dataset_config": asdict(config.dataset),
        "planned_steps": config.total_steps,
        "dataset_source": config.dataset_source,
        "dataset_license": config.dataset_license,
        "seed": config.seed,
        "wandb_run_id": config.wandb_run_id,
    }
    validate_resume_contract(checkpoint, expected)


def _initialize_wandb(context: _PretrainingContext, network: LunaNetwork) -> bool:
    config = context.config
    if config.wandb_project is None:
        return False
    if config.wandb_run_id is None:
        raise RuntimeError("Validated W&B run ID is unexpectedly missing")
    wandb.init(
        project=config.wandb_project,
        id=config.wandb_run_id,
        name=config.wandb_run_name,
        resume=config.wandb_resume,
        config=_wandb_config(context, network),
        tags=["chess", "ezv2", "pgn-pretraining"],
    )
    wandb.define_metric("global_step")
    _define_wandb_metrics()
    logger.info("W&B initialized for PGN pretraining in project {}", config.wandb_project)
    return True


def _define_wandb_metrics() -> None:
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("validation/*", step_metric="global_step")


def _wandb_config(
    context: _PretrainingContext,
    network: LunaNetwork,
) -> dict[str, object]:
    config = context.config
    provenance = network.training_phase_provenance
    return {
        "seed": config.seed,
        "total_steps": config.total_steps,
        "chunk_steps": config.chunk_steps,
        "checkpoint_top_k": config.checkpoint_top_k,
        "validation_batch_size": config.validation_batch_size,
        "validation_positions": config.validation_positions,
        "learner": asdict(config.learner),
        "dataset": _dataset_metadata(context),
        "training_phase_provenance": provenance.as_config() if provenance is not None else None,
    }


def _train_chunks(
    network: LunaNetwork,
    replay: PrioritizedReplayBuffer,
    context: _PretrainingContext,
) -> PgnPretrainingResult:
    validation = _evaluate_and_log(network, context)
    if network.global_step >= context.config.total_steps:
        _publish_checkpoints(network, context)
    try:
        while network.global_step < context.config.total_steps:
            validation = _train_chunk(network, replay, context)
    except KeyboardInterrupt:
        _publish_checkpoints(network, context)
        raise
    return PgnPretrainingResult(network.global_step, validation, context.config.output_dir / "latest.pth.tar")


def _train_chunk(
    network: LunaNetwork,
    replay: PrioritizedReplayBuffer,
    context: _PretrainingContext,
) -> ValidationMetrics:
    config = context.config
    next_boundary = (network.global_step // config.chunk_steps + 1) * config.chunk_steps
    steps = min(next_boundary, config.total_steps) - network.global_step
    network.train_ezv2(replay, steps, total_train_steps=config.total_steps)
    validation = _evaluate_and_log(network, context)
    _publish_checkpoints(network, context)
    return validation


def _evaluate_and_log(network: LunaNetwork, context: _PretrainingContext) -> ValidationMetrics:
    config = context.config
    plan = ValidationPlan(config.validation_batch_size, config.validation_positions, config.seed)
    validation = evaluate_validation(network, context.dataset.validation_trajectories, plan)
    _log_validation(network.global_step, validation)
    return validation


def _log_validation(global_step: int, metrics: ValidationMetrics) -> None:
    logger.info(
        "PGN validation step {} | top1={:.3f} top5={:.3f} nll={:.4f} value_mae={:.4f}",
        global_step,
        metrics.policy_top1,
        metrics.policy_top5,
        metrics.policy_nll,
        metrics.value_mae,
    )
    if wandb.run is not None:
        wandb.log(metrics.as_wandb(global_step))


def _publish_checkpoints(
    network: LunaNetwork,
    context: _PretrainingContext,
) -> None:
    config = context.config
    metadata = _dataset_metadata(context)
    publication = CheckpointPublication(
        config.output_dir,
        config.checkpoint_top_k,
        metadata,
        protected_step_interval=config.chunk_steps,
    )
    publish_pretraining_checkpoints(network, publication)


def _dataset_metadata(context: _PretrainingContext) -> dict[str, object]:
    config = context.config
    return {
        "dataset_filename": config.dataset_path.name,
        "dataset_sha256": context.dataset_sha256,
        "dataset_source": config.dataset_source,
        "dataset_license": config.dataset_license,
        "dataset_config": asdict(config.dataset),
        "dataset_stats": asdict(context.dataset.stats),
        "planned_steps": config.total_steps,
        "seed": config.seed,
        "wandb_run_id": config.wandb_run_id,
    }
