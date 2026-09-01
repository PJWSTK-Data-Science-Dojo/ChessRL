"""Command-line entry point for offline PGN warm-start training."""

from __future__ import annotations

import sys

import torch
import tyro
from loguru import logger

from luna.network import RepresentationCollapseError
from luna.pgn_pretraining import run_pgn_pretraining
from luna.pgn_pretraining_config import PgnPretrainingConfig


def _configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level)


def main() -> int:
    torch.set_float32_matmul_precision("medium")
    _configure_logging()
    config = tyro.cli(PgnPretrainingConfig)
    try:
        result = run_pgn_pretraining(config)
    except KeyboardInterrupt:
        logger.info("PGN pretraining interrupted after publishing the latest completed optimizer state.")
        return 130
    except RepresentationCollapseError as exc:
        logger.critical("PGN pretraining stopped by the representation-collapse guard: {}", exc)
        return 78
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("PGN pretraining failed: {}", exc)
        return 2
    logger.info("PGN pretraining completed at step {}; checkpoint={}", result.global_step, result.latest_checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
