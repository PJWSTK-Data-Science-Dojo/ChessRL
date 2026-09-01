"""Benchmark a Luna checkpoint vs Stockfish (no training loop).

Example:
    uv run python src/eval_vs_stockfish.py --checkpoint ./runs/luna-main/latest.pth.tar \\
      --run.stockfish-eval-games 10 --run.stockfish-elo 1500
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from loguru import logger

from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.game.chess_game import ChessGame as Game
from luna.game.stockfish_eval import (
    EngineMatchSettings,
    StockfishEvalOutcome,
    StockfishEvalScores,
    ladder_match_settings,
    run_stockfish_eval,
    validate_ladder_configuration,
)
from luna.network import LunaNetwork


@dataclass
class EvalVsStockfishCli:
    checkpoint: str = "./runs/luna-main/latest.pth.tar"
    opponent: Literal["stockfish", "fairy"] = "stockfish"
    log_level: str = "INFO"
    learner: EzV2LearnerConfig = field(default_factory=EzV2LearnerConfig)
    run: TrainingRunConfig = field(default_factory=TrainingRunConfig)


def main() -> int:
    cfg = tyro.cli(EvalVsStockfishCli)
    _configure_logging(cfg.log_level)
    checkpoint = Path(cfg.checkpoint).resolve()
    if not checkpoint.is_file():
        logger.error("Checkpoint not found: {}", checkpoint)
        return 1
    game, network = _load_network(checkpoint, cfg.learner)
    settings = _evaluation_settings(cfg)
    outcome = run_stockfish_eval(game, network, cfg.run, iteration=None, settings=settings)
    return _outcome_exit_code(outcome)


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper())


def _load_network(checkpoint: Path, learner: EzV2LearnerConfig) -> tuple[Game, LunaNetwork]:
    game = Game()
    network = LunaNetwork.from_checkpoint(
        game,
        checkpoint,
        device=learner.device,
        cuda_device=learner.cuda_device,
        compile_inference=learner.compile_inference,
        load_optimizer=False,
    )
    logger.info("Loaded checkpoint {}", checkpoint)
    return game, network


def _evaluation_settings(config: EvalVsStockfishCli) -> EngineMatchSettings | None:
    settings = None
    if config.opponent == "fairy":
        validate_ladder_configuration(config.run)
        settings = ladder_match_settings(config.run, config.run.ladder_start_elo)
    return settings


def _outcome_exit_code(outcome: StockfishEvalOutcome) -> int:
    if isinstance(outcome, StockfishEvalScores):
        return 0
    if outcome.reason in {"too_few_games", "too_many_games"}:
        logger.error(
            "Stockfish eval skipped: {} — set --run.stockfish-eval-games to an even number from 2 through 20.",
            outcome.message,
        )
        return 3
    if outcome.reason == "no_engine":
        logger.error("Stockfish eval skipped (engine): {}", outcome.message)
        return 2
    logger.error("Stockfish eval failed: {}", outcome.message)
    return 4


if __name__ == "__main__":
    sys.exit(main())
