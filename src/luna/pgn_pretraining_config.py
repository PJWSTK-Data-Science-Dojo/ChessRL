from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from luna.config import (
    EzV2LearnerConfig,
    WandbResumeMode,
    validate_learner_config,
    validate_wandb_resume,
    validate_wandb_run_id,
    validate_wandb_run_name,
)
from luna.pgn_dataset import PgnDatasetConfig
from luna.pgn_pretraining_checkpoints import pretraining_resume_exists


def _default_learner() -> EzV2LearnerConfig:
    return _offline_objectives(_offline_optimization(_offline_architecture()))


def _offline_architecture() -> EzV2LearnerConfig:
    return EzV2LearnerConfig(
        model_name="balanced_reconstruction",
        num_channels=128,
        repr_blocks=10,
        dyn_blocks=1,
        proj_dim=256,
    )


def _offline_optimization(learner: EzV2LearnerConfig) -> EzV2LearnerConfig:
    return replace(
        learner,
        lr=1e-4,
        lr_min=1e-5,
        lr_warmup_steps=500,
        batch_size=256,
        grad_accum_steps=2,
        amp_dtype="bfloat16",
        compile_training=True,
    )


def _offline_objectives(learner: EzV2LearnerConfig) -> EzV2LearnerConfig:
    return replace(
        learner,
        unroll_steps=5,
        td_steps=0,
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        reward_loss_weight=0.0,
        consistency_loss_weight=0.0,
        reconstruction_loss_weight=0.5,
        reanalyze_mcts_sims=0,
        reanalyze_prob=0.0,
        reanalyze_policy=False,
        reanalyze_start_step=0,
    )


@dataclass(frozen=True, slots=True)
class PgnPretrainingConfig:
    dataset_path: Path = Path("data/lichess_db_broadcast_2026-07.pgn.zst")
    output_dir: Path = Path("runs/luna-balanced-pgn-pretrain-v1")
    source_checkpoint: Path | None = None
    resume_checkpoint: Path | None = None
    total_steps: int = 10_000
    chunk_steps: int = 1_000
    checkpoint_top_k: int = 10
    validation_batch_size: int = 512
    validation_positions: int = 20_000
    dataset_source: str = "Lichess Broadcast database"
    dataset_license: str = "CC BY-SA 4.0"
    seed: int = 0
    wandb_project: str | None = None
    wandb_run_id: str | None = None
    wandb_run_name: str | None = None
    wandb_resume: WandbResumeMode = "never"
    dataset: PgnDatasetConfig = field(default_factory=PgnDatasetConfig)
    learner: EzV2LearnerConfig = field(default_factory=_default_learner)


def validate_pretraining_config(config: PgnPretrainingConfig) -> None:
    validate_learner_config(config.learner)
    validate_wandb_run_id(config.wandb_run_id)
    validate_wandb_run_name(config.wandb_run_name)
    validate_wandb_resume(config.wandb_resume)
    _validate_positive_fields(config)
    _validate_offline_learner(config.learner)
    if config.wandb_project is not None and config.wandb_run_id is None:
        raise ValueError("PGN pretraining with W&B requires an explicit wandb_run_id")
    _validate_paths(config)


def _validate_positive_fields(config: PgnPretrainingConfig) -> None:
    names = ("total_steps", "chunk_steps", "checkpoint_top_k", "validation_batch_size", "validation_positions")
    for name in names:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not config.dataset_source.strip() or not config.dataset_license.strip():
        raise ValueError("dataset_source and dataset_license cannot be blank")


def _validate_offline_learner(learner: EzV2LearnerConfig) -> None:
    if learner.td_steps != 0:
        raise ValueError("PGN pretraining requires learner.td_steps=0")
    if learner.reward_loss_weight != 0.0 or learner.consistency_loss_weight != 0.0:
        raise ValueError("PGN pretraining requires reward and consistency loss weights of zero")
    if learner.reanalyze_mcts_sims != 0 or learner.reanalyze_prob != 0.0 or learner.reanalyze_policy:
        raise ValueError("PGN pretraining must disable replay reanalysis")


def _validate_paths(config: PgnPretrainingConfig) -> None:
    if not config.dataset_path.expanduser().is_file():
        raise FileNotFoundError(f"PGN dataset does not exist: {config.dataset_path.expanduser().resolve()}")
    if (config.source_checkpoint is None) == (config.resume_checkpoint is None):
        raise ValueError("Provide exactly one of source_checkpoint or resume_checkpoint")
    if config.resume_checkpoint is None:
        _validate_fresh_paths(config)
        return
    _validate_resume_paths(config)


def _validate_fresh_paths(config: PgnPretrainingConfig) -> None:
    source = config.source_checkpoint
    if source is None or not source.expanduser().is_file():
        raise FileNotFoundError(f"Training checkpoint does not exist: {source}")
    if config.wandb_resume == "must":
        raise ValueError("A fresh PGN phase cannot require an existing W&B run")
    _validate_fresh_output(config.output_dir)


def _validate_resume_paths(config: PgnPretrainingConfig) -> None:
    resume = config.resume_checkpoint
    if resume is None or not pretraining_resume_exists(resume, config.output_dir):
        raise FileNotFoundError(f"Training checkpoint does not exist: {resume}")
    if config.wandb_resume == "never":
        raise ValueError("A resumed PGN phase requires wandb_resume='allow' or 'must'")
    if resume.expanduser().resolve().parent != config.output_dir.expanduser().resolve():
        raise ValueError("resume_checkpoint must belong to output_dir")


def _validate_fresh_output(output_dir: Path) -> None:
    target = output_dir.expanduser().resolve()
    if not target.exists():
        return
    if not target.is_dir():
        raise FileExistsError(f"Pretraining output is not a directory: {target}")
    contents = sorted(path.name for path in target.iterdir())
    if contents:
        raise FileExistsError(f"Fresh PGN pretraining requires an empty output directory: {contents}")
