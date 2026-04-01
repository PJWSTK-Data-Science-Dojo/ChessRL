"""Luna-Chess EfficientZeroV2 training entry point."""

from __future__ import annotations

import gc
import logging
import sys

import coloredlogs
import torch

from luna.coach import Coach
from luna.game import ChessGame as Game
from luna.network import LunaNetwork
from luna.utils import dotdict

gc.enable()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

args = dotdict(
    {
        # ---- iteration control ----
        "numIters": 5,
        "numEps": 10,
        "tempThreshold": 10,
        "updateThreshold": 0.6,
        "arenaCompare": 10,
        # ---- MCTS ----
        "numMCTSSims": 50,
        "cpuct": 1.25,
        "dir_noise": True,
        "dir_alpha": 0.3,
        # ---- EZV2 training ----
        "unroll_steps": 5,
        "td_steps": 10,
        "discount": 0.997,
        "batch_size": 64,
        "train_steps_per_iter": 200,
        "support_size": 10,
        # ---- replay ----
        "replay_capacity": 100_000,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "reanalyze_ratio": 0.0,
        # ---- checkpoints ----
        "checkpoint": "./temp/",
        "load_model": False,
        "load_folder_file": ("./pretrained_models/", "best.pth.tar"),
        "save_anyway": True,
    }
)


def main() -> int:
    log.info("Loading %s...", Game.__name__)
    game = Game()

    log.info("Loading %s...", LunaNetwork.__name__)
    nnet = LunaNetwork(game)

    if args.load_model:
        log.info('Loading checkpoint "%s/"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning("Not loading a checkpoint!")

    log.info("Loading the Coach...")
    c = Coach(game, nnet, args)

    log.info("Starting EfficientZeroV2 learning process")
    c.learn()

    return 0


if __name__ == "__main__":
    sys.exit(main())
