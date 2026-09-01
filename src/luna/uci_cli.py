"""Command-line startup for the Luna UCI engine."""

from __future__ import annotations

import argparse
import sys

import torch
from loguru import logger

from luna.game.chess_game import ChessGame
from luna.network import LunaNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Luna checkpoint as a UCI chess engine.")
    parser.add_argument("--checkpoint", default="./runs/luna-main/latest.pth.tar")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--mcts-sims", type=int, default=100)
    parser.add_argument("--minimum-sims", type=int, default=8)
    parser.add_argument("--estimated-sim-ms", type=float, default=4.0)
    parser.add_argument("--compile-inference", action="store_true")
    return parser.parse_args()


def run_cli(args: argparse.Namespace) -> int:
    from luna.uci import LunaUciEngine, UciOptions

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    torch.set_float32_matmul_precision("medium")
    game = ChessGame(claim_draw=False)
    try:
        options = UciOptions(
            mcts_simulations=args.mcts_sims,
            minimum_simulations=args.minimum_sims,
            estimated_simulation_ms=args.estimated_sim_ms,
        )
    except ValueError as exc:
        logger.error("Invalid UCI search options: {}", exc)
        return 2
    try:
        network = LunaNetwork.from_checkpoint(
            game,
            args.checkpoint,
            device=args.device,
            cuda_device=args.cuda_device,
            compile_inference=args.compile_inference,
            load_optimizer=False,
        )
    except (KeyError, OSError, RuntimeError, ValueError):
        logger.exception("Could not load Luna checkpoint")
        return 2
    engine = LunaUciEngine(
        network,
        options,
    )
    engine.run()
    return 0
