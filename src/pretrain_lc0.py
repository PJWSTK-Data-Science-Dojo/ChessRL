"""Command-line entry point for supervised LC0 policy/value pretraining."""

from __future__ import annotations

import sys

import torch
import tyro
from loguru import logger

from luna.lc0_pretraining import run_lc0_pretraining
from luna.lc0_pretraining_config import Lc0PretrainingConfig


def _configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level)


def main() -> int:
    torch.set_float32_matmul_precision("medium")
    _configure_logging()
    config = tyro.cli(Lc0PretrainingConfig)
    try:
        result = run_lc0_pretraining(config)
    except KeyboardInterrupt:
        logger.info("LC0 pretraining interrupted after publishing the latest completed optimizer state.")
        return 130
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("LC0 pretraining failed: {}", exc)
        return 2
    logger.info("LC0 pretraining completed at step {}; checkpoint={}", result.global_step, result.latest_checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
