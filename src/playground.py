"""Debug/playground script."""

from __future__ import annotations

import logging
import sys

import coloredlogs
import numpy as np

from luna.coach import Coach
from luna.game import ChessGame as Game
from luna.game.arena import Arena
from luna.mcts import MCTS
from luna.network import LunaNetwork
from luna.utils import dotdict

log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

args = dotdict(
    {
        "numIters": 1,
        "numEps": 2,
        "tempThreshold": 10,
        "updateThreshold": 0.6,
        "arenaCompare": 2,
        "numMCTSSims": 10,
        "cpuct": 1.25,
        "dir_noise": False,
        "dir_alpha": 0.3,
        "unroll_steps": 3,
        "td_steps": 5,
        "discount": 0.997,
        "batch_size": 8,
        "train_steps_per_iter": 10,
        "support_size": 10,
        "replay_capacity": 10000,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "checkpoint": "./temp/",
        "load_model": False,
        "load_folder_file": ("./pretrained_models/", "best.pth.tar"),
        "save_anyway": False,
    }
)


def main() -> int:
    log.info("Loading %s...", Game.__name__)
    g = Game()

    log.info("Loading %s...", LunaNetwork.__name__)
    nnet = LunaNetwork(g)

    try:
        nnet.load_checkpoint(args["checkpoint"], "best.pth.tar")
        log.info("Loaded checkpoint")
    except FileNotFoundError:
        log.warning("No checkpoint found, using untrained network")

    nmcts = MCTS(g, nnet, args)
    nmcts2 = MCTS(g, nnet, args)

    def _print(x: object) -> None:
        print(x)

    arena = Arena(
        lambda x: int(np.argmax(nmcts.get_action_prob(x, temp=0))),
        lambda x: int(np.argmax(nmcts2.get_action_prob(x, temp=0))),
        g,
        display=_print,
    )

    arena.play_game(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
