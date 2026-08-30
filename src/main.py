"""Luna-Chess EfficientZeroV2 training entry point."""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import tyro
from loguru import logger

from luna.coach import Coach, validate_fresh_checkpoint_target, validate_resume_checkpoint_target
from luna.config import TrainCliConfig, validate_training_configuration
from luna.game.chess_game import ChessGame as Game
from luna.network import LunaNetwork


def main() -> int:
    torch.set_float32_matmul_precision("medium")

    cfg = tyro.cli(TrainCliConfig)

    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level.upper())

    run_cfg = cfg.to_training_run()
    learner = cfg.to_learner_config()
    learner.discount = run_cfg.discount
    try:
        validate_training_configuration(run_cfg, learner)
        if cfg.load_model:
            source_checkpoint = Path(cfg.load_checkpoint_dir) / cfg.load_checkpoint_file
            validate_resume_checkpoint_target(run_cfg, source_checkpoint)
        else:
            validate_fresh_checkpoint_target(run_cfg)
    except (FileExistsError, ValueError) as exc:
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
    c = Coach(game, nnet, run_cfg, wandb_project=cfg.wandb_project, seed=cfg.seed)

    logger.info("Starting EfficientZeroV2 learning process")
    c.learn()

    return 0


if __name__ == "__main__":
    sys.exit(main())
