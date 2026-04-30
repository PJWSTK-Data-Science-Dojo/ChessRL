"""Luna-Chess EfficientZeroV2 training entry point."""

from __future__ import annotations

import sys

import tyro
from loguru import logger

from luna.coach import Coach
from luna.config import TrainCliConfig
from luna.game import ChessGame as Game
from luna.network import LunaNetwork


def main() -> int:
    cfg = tyro.cli(TrainCliConfig)

    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level.upper())

    logger.info("Loading {}...", Game.__name__)
    game = Game()

    learner = cfg.to_learner_config()
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
        logger.warning("Not loading a checkpoint!")

    run_cfg = cfg.to_training_run()
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
    c = Coach(game, nnet, run_cfg)

    logger.info("Starting EfficientZeroV2 learning process")
    c.learn()

    return 0


if __name__ == "__main__":
    sys.exit(main())
