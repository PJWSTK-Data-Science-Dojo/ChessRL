"""Debug/playground script."""

from __future__ import annotations

import sys

import numpy as np
from loguru import logger

from luna.config import TrainingRunConfig
from luna.game import ChessGame as Game
from luna.game.arena import Arena
from luna.mcts import MCTS
from luna.network import LunaNetwork

PLAYGROUND_RUN = TrainingRunConfig(
    num_iters=1,
    num_episodes=2,
    temp_threshold=10,
    update_threshold=0.6,
    arena_compare=2,
    num_mcts_sims=10,
    cpuct=1.25,
    dir_noise=False,
    dir_alpha=0.3,
    discount=0.997,
    batch_size=8,
    train_steps_per_iter=10,
    replay_capacity=10_000,
    per_alpha=0.6,
    per_beta=0.4,
    checkpoint="./temp/",
    save_anyway=False,
)


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("Loading {}...", Game.__name__)
    g = Game()

    logger.info("Loading {}...", LunaNetwork.__name__)
    nnet = LunaNetwork(g)

    try:
        nnet.load_checkpoint(PLAYGROUND_RUN.checkpoint, "best.pth.tar")
        logger.info("Loaded checkpoint")
    except FileNotFoundError:
        logger.warning("No checkpoint found, using untrained network")

    nmcts = MCTS(g, nnet, PLAYGROUND_RUN)
    nmcts2 = MCTS(g, nnet, PLAYGROUND_RUN)

    def _display(x: object) -> None:
        logger.info("{}", x)

    arena = Arena(
        lambda x: int(np.argmax(nmcts.get_action_prob(x, temp=0))),
        lambda x: int(np.argmax(nmcts2.get_action_prob(x, temp=0))),
        g,
        display=_display,
    )

    arena.play_game(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
