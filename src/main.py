"""Luna-Chess EfficientZeroV2 training entry point."""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import tyro
from loguru import logger

from luna.coach import Coach, validate_fresh_checkpoint_target, validate_resume_checkpoint_target
from luna.config import (
    TrainCliConfig,
    validate_training_configuration,
    validate_wandb_resume,
    validate_wandb_run_id,
    validate_wandb_run_name,
)
from luna.game.chess_game import ChessGame as Game
from luna.network import LunaNetwork


def validate_new_training_phase_target(checkpoint_dir: str) -> None:
    """Require a dedicated empty directory for a weights-only training phase."""
    if not checkpoint_dir.strip():
        raise ValueError("new_training_phase requires a non-empty --run.checkpoint directory")
    target = Path(checkpoint_dir).expanduser().resolve()
    if not target.exists():
        return
    if not target.is_dir():
        raise FileExistsError(f"New training phase target is not a directory: {target}")
    contents = sorted(path.name for path in target.iterdir())
    if contents:
        raise FileExistsError(
            f"New training phase requires an empty checkpoint directory, but {target} contains {contents}. "
            "Choose a new --run.checkpoint directory."
        )


def main() -> int:
    torch.set_float32_matmul_precision("medium")

    cfg = tyro.cli(TrainCliConfig)

    logger.remove()
    try:
        logger.add(sys.stderr, level=cfg.log_level.upper())
    except ValueError:
        logger.add(sys.stderr, level="INFO")
        logger.error("Invalid log level: {!r}", cfg.log_level)
        return 2

    run_cfg = cfg.to_training_run()
    learner = cfg.to_learner_config()
    learner.discount = run_cfg.discount
    try:
        validate_training_configuration(run_cfg, learner)
        validate_wandb_run_id(cfg.wandb_run_id)
        validate_wandb_run_name(cfg.wandb_run_name)
        validate_wandb_resume(cfg.wandb_resume)
        if cfg.load_model and cfg.new_training_phase:
            raise ValueError("--load-model and --new-training-phase are mutually exclusive")
        source_checkpoint = Path(cfg.load_checkpoint_dir) / cfg.load_checkpoint_file
        if cfg.load_model:
            validate_resume_checkpoint_target(run_cfg, source_checkpoint)
        elif cfg.new_training_phase:
            if not source_checkpoint.expanduser().is_file():
                raise FileNotFoundError(f"No model in path {source_checkpoint.expanduser().resolve()}")
            validate_new_training_phase_target(run_cfg.checkpoint)
        else:
            validate_fresh_checkpoint_target(run_cfg)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        logger.error("Invalid training setup: {}", exc)
        return 2

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    logger.info("Loading {}...", Game.__name__)
    game = Game()

    logger.info("Loading {}...", LunaNetwork.__name__)
    nnet = LunaNetwork(game, learner)
    nnet.log_model_summary()

    if cfg.load_model:
        logger.info(
            'Loading checkpoint "{}" / "{}"...',
            cfg.load_checkpoint_dir,
            cfg.load_checkpoint_file,
        )
        nnet.load_checkpoint(cfg.load_checkpoint_dir, cfg.load_checkpoint_file)
    elif cfg.new_training_phase:
        logger.info(
            'Starting a new training phase from weights in "{}" / "{}"; '
            "optimizer, scaler, counters, and LR schedule will start fresh.",
            cfg.load_checkpoint_dir,
            cfg.load_checkpoint_file,
        )
        nnet.initialize_training_phase(cfg.load_checkpoint_dir, cfg.load_checkpoint_file)
    else:
        logger.info("Starting a new run from randomly initialized weights.")

    if run_cfg.profile:
        logger.info(
            "Profiling on: phase timings each iter; Kineto on iter {} ({} steps) -> chrome in {} | TensorBoard logdir={}",
            run_cfg.profile_torch_iter,
            run_cfg.profile_torch_steps,
            run_cfg.profile_dir,
            run_cfg.profile_tensorboard_logdir,
        )
        logger.info(
            "Kineto traces are *.pt.trace.json under your TensorBoard logdir (no scalars). "
            "Run: uv run tensorboard --logdir <that-dir>  then open the PYTORCH_PROFILER tab "
            "(needs torch-tb-profiler, listed in pyproject). Or load the same .json in chrome://tracing.",
        )

    logger.info("Loading the Coach...")
    c = Coach(
        game,
        nnet,
        run_cfg,
        wandb_project=cfg.wandb_project,
        wandb_run_id=cfg.wandb_run_id,
        wandb_run_name=cfg.wandb_run_name,
        wandb_resume=cfg.wandb_resume,
        seed=cfg.seed,
    )

    logger.info("Starting EfficientZeroV2 learning process")
    c.learn()

    return 0


if __name__ == "__main__":
    sys.exit(main())
