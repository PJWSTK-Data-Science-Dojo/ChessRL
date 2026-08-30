"""Benchmark a Luna checkpoint vs Stockfish (no training loop).

Example:
    uv run python src/eval_vs_stockfish.py --checkpoint ./temp/latest.pth.tar \\
      --run.stockfish-eval-games 10 --run.stockfish-elo 1320
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from loguru import logger

from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.game.chess_game import ChessGame as Game
from luna.game.stockfish_eval import StockfishEvalScores, run_stockfish_eval
from luna.network import LunaNetwork


@dataclass
class EvalVsStockfishCli:
    """Load ``checkpoint`` and run :func:`~luna.game.stockfish_eval.run_stockfish_eval`."""

    checkpoint: str = "./temp/latest.pth.tar"
    """Path to a ``*.pth.tar`` file."""

    log_level: str = "INFO"
    learner: EzV2LearnerConfig = field(default_factory=EzV2LearnerConfig)
    run: TrainingRunConfig = field(default_factory=TrainingRunConfig)


def main() -> int:
    cfg = tyro.cli(EvalVsStockfishCli)
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level.upper())

    ck = Path(cfg.checkpoint).resolve()
    if not ck.is_file():
        logger.error("Checkpoint not found: {}", ck)
        return 1

    game = Game()
    nnet = LunaNetwork.from_checkpoint(
        game,
        ck,
        device=cfg.learner.device,
        cuda_device=cfg.learner.cuda_device,
        compile_inference=cfg.learner.compile_inference,
        load_optimizer=False,
    )
    logger.info("Loaded checkpoint {}", ck)

    out = run_stockfish_eval(game, nnet, cfg.run, iteration=None)
    if isinstance(out, StockfishEvalScores):
        return 0
    if out.reason == "too_few_games":
        logger.error("Stockfish eval skipped: {} — set --run.stockfish-eval-games to an even number ≥ 2.", out.message)
        return 3
    if out.reason == "no_engine":
        logger.error("Stockfish eval skipped (engine): {}", out.message)
        return 2
    logger.error("Stockfish eval failed: {}", out.message)
    return 4


if __name__ == "__main__":
    sys.exit(main())
