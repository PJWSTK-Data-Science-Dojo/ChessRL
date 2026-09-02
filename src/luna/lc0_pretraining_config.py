from __future__ import annotations

import pickle
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from luna.config import (
    EzV2LearnerConfig,
    WandbResumeMode,
    validate_learner_config,
    validate_wandb_resume,
    validate_wandb_run_id,
    validate_wandb_run_name,
)
from luna.lc0_corpus import LC0_ADAPTER_VERSION, lc0_archive_paths
from luna.lc0_dataset import Lc0DatasetConfig
from luna.pgn_pretraining_checkpoints import pretraining_resume_exists

LC0_CHECKPOINT_METADATA_KEY = "lc0_pretraining"
LC0_CHECKPOINT_PREFIX = "lc0_step_"
Lc0TrainScope = Literal["prediction_heads", "representation_and_heads"]


def _default_learner() -> EzV2LearnerConfig:
    learner = EzV2LearnerConfig(
        model_name="balanced_reconstruction",
        num_channels=128,
        repr_blocks=10,
        dyn_blocks=1,
        proj_dim=256,
    )
    return replace(
        learner,
        lr=1e-4,
        lr_min=1e-5,
        lr_warmup_steps=50,
        batch_size=512,
        grad_accum_steps=2,
        amp_dtype="bfloat16",
        compile_training=False,
        unroll_steps=1,
        td_steps=0,
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        reward_loss_weight=0.0,
        consistency_loss_weight=0.0,
        reconstruction_loss_weight=0.0,
        reanalyze_mcts_sims=0,
        reanalyze_prob=0.0,
        reanalyze_policy=False,
        reanalyze_start_step=0,
    )


@dataclass(frozen=True, slots=True)
class Lc0PretrainingConfig:
    dataset_path: Path = Path("data/lc0/training-run2-test91-20260901-1317.tar")
    output_dir: Path = Path("runs/luna-balanced-lc0-heads-pretrain-v1")
    source_checkpoint: Path | None = None
    resume_checkpoint: Path | None = None
    total_steps: int = 1_000
    chunk_steps: int = 250
    checkpoint_top_k: int = 4
    validation_batch_size: int = 512
    validation_positions: int = 20_000
    dataset_source: str = "Official Leela Chess Zero training data"
    dataset_license: str = "ODbL 1.0 (collection); DBCL 1.0 (contents)"
    seed: int = 0
    train_scope: Lc0TrainScope = "prediction_heads"
    wandb_project: str | None = None
    wandb_run_id: str | None = "luna-balanced-lc0-heads-pretrain-v1"
    wandb_run_name: str | None = "Luna Balanced · LC0 Policy+Value Heads v1"
    wandb_resume: WandbResumeMode = "never"
    dataset: Lc0DatasetConfig = field(default_factory=Lc0DatasetConfig)
    learner: EzV2LearnerConfig = field(default_factory=_default_learner)


def validate_lc0_pretraining_config(config: Lc0PretrainingConfig) -> None:
    validate_learner_config(config.learner)
    validate_wandb_run_id(config.wandb_run_id)
    validate_wandb_run_name(config.wandb_run_name)
    validate_wandb_resume(config.wandb_resume)
    if config.train_scope not in {"prediction_heads", "representation_and_heads"}:
        raise ValueError("train_scope must be 'prediction_heads' or 'representation_and_heads'")
    _validate_positive_fields(config)
    _validate_objectives(config.learner)
    _validate_wandb(config)
    _validate_paths(config)


def _validate_positive_fields(config: Lc0PretrainingConfig) -> None:
    names = ("total_steps", "chunk_steps", "checkpoint_top_k", "validation_batch_size", "validation_positions")
    for name in names:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise ValueError("seed must be an integer")
    if not config.dataset_source.strip() or not config.dataset_license.strip():
        raise ValueError("dataset_source and dataset_license cannot be blank")
    if not 0.0 < config.dataset.validation_fraction < 1.0:
        raise ValueError("LC0 pretraining requires validation_fraction strictly between zero and one")


def _validate_objectives(learner: EzV2LearnerConfig) -> None:
    if learner.support_size != 1:
        raise ValueError("LC0 WDL pretraining requires learner.support_size=1")
    if learner.policy_loss_weight <= 0.0 or learner.value_loss_weight <= 0.0:
        raise ValueError("LC0 pretraining requires positive policy and value loss weights")
    auxiliary = (
        learner.reward_loss_weight,
        learner.consistency_loss_weight,
        learner.reconstruction_loss_weight,
    )
    if any(weight != 0.0 for weight in auxiliary):
        raise ValueError("LC0 root-only pretraining requires reward, consistency, and reconstruction weights of zero")
    if learner.td_steps != 0:
        raise ValueError("LC0 root-only pretraining requires learner.td_steps=0")
    if learner.reanalyze_mcts_sims != 0 or learner.reanalyze_prob != 0.0 or learner.reanalyze_policy:
        raise ValueError("LC0 pretraining must disable replay reanalysis")


def _validate_wandb(config: Lc0PretrainingConfig) -> None:
    if config.wandb_project is None:
        return
    if config.wandb_run_id is None or config.wandb_run_name is None:
        raise ValueError("LC0 pretraining with W&B requires explicit wandb_run_id and wandb_run_name")


def _validate_paths(config: Lc0PretrainingConfig) -> None:
    dataset = config.dataset_path.expanduser()
    lc0_archive_paths(dataset)
    if (config.source_checkpoint is None) == (config.resume_checkpoint is None):
        raise ValueError("Provide exactly one of source_checkpoint or resume_checkpoint")
    if config.resume_checkpoint is None:
        _validate_fresh_paths(config)
        return
    _validate_resume_paths(config)


def _validate_fresh_paths(config: Lc0PretrainingConfig) -> None:
    source = config.source_checkpoint
    if source is None or not source.expanduser().is_file():
        raise FileNotFoundError(f"Training checkpoint does not exist: {source}")
    if config.wandb_resume == "must":
        raise ValueError("A fresh LC0 phase cannot require an existing W&B run")
    output = config.output_dir.expanduser().resolve()
    if not output.exists():
        return
    if not output.is_dir():
        raise FileExistsError(f"Pretraining output is not a directory: {output}")
    contents = sorted(path.name for path in output.iterdir())
    if contents:
        raise FileExistsError(f"Fresh LC0 pretraining requires an empty output directory: {contents}")


def _validate_resume_paths(config: Lc0PretrainingConfig) -> None:
    resume = config.resume_checkpoint
    if resume is None or not pretraining_resume_exists(
        resume,
        config.output_dir,
        checkpoint_prefix=LC0_CHECKPOINT_PREFIX,
    ):
        raise FileNotFoundError(f"Training checkpoint does not exist: {resume}")
    if config.wandb_resume == "never":
        raise ValueError("A resumed LC0 phase requires wandb_resume='allow' or 'must'")
    if resume.expanduser().resolve().parent != config.output_dir.expanduser().resolve():
        raise ValueError("resume_checkpoint must belong to output_dir")


def lc0_resume_contract(
    config: Lc0PretrainingConfig,
    fingerprint: str,
    frozen_digest: str | None = None,
) -> dict[str, object]:
    contract: dict[str, object] = {
        "pretraining_kind": _pretraining_kind(config.train_scope),
        "train_scope": config.train_scope,
        "lc0_adapter_version": LC0_ADAPTER_VERSION,
        "dataset_fingerprint": fingerprint,
        "dataset_config": asdict(config.dataset),
        "dataset_source": config.dataset_source,
        "dataset_license": config.dataset_license,
        "planned_steps": config.total_steps,
        "chunk_steps": config.chunk_steps,
        "validation_positions": config.validation_positions,
        "seed": config.seed,
        "wandb_run_id": config.wandb_run_id,
        "wandb_run_name": config.wandb_run_name,
    }
    if frozen_digest is not None:
        contract["frozen_parameters_sha256"] = frozen_digest
    return contract


def _pretraining_kind(scope: Lc0TrainScope) -> str:
    if scope == "representation_and_heads":
        return "lc0_representation_policy_value"
    return "lc0_policy_value_heads"


def lc0_dataset_metadata(
    config: Lc0PretrainingConfig,
    fingerprint: str,
    frozen_digest: str,
) -> dict[str, object]:
    return {
        **lc0_resume_contract(config, fingerprint, frozen_digest),
        "dataset_filename": config.dataset_path.name,
        "dataset_source": config.dataset_source,
        "dataset_license": config.dataset_license,
        "effective_train_batch_size": config.learner.batch_size,
        "validation_positions": config.validation_positions,
        "checkpoint_metadata_namespace": LC0_CHECKPOINT_METADATA_KEY,
    }


def seed_lc0_pretraining(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lc0_resume_seed(seed: int, global_step: int) -> int:
    state = np.random.SeedSequence([seed, global_step]).generate_state(1)
    return int(state[0])


def validate_lc0_online_source(checkpoint: Path, expected_fingerprint: str) -> None:
    resolved = checkpoint.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"LC0 online source checkpoint does not exist: {resolved}")
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except (EOFError, IndexError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Cannot read LC0 online source checkpoint: {resolved}") from exc
    metadata = payload.get(LC0_CHECKPOINT_METADATA_KEY) if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError(f"LC0 online source checkpoint has no pretraining metadata: {resolved}")
    if metadata.get("dataset_fingerprint") != expected_fingerprint:
        raise ValueError(f"LC0 online source checkpoint corpus fingerprint does not match: {resolved}")
    if metadata.get("train_scope") != "representation_and_heads":
        raise ValueError(f"LC0 online source checkpoint was not jointly trained: {resolved}")
