"""Single-GPU throughput benchmarks for EfficientZeroV2 pipeline."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro
from loguru import logger

from luna.coach import Coach
from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.game.chess_game import OBS_PLANES, ChessGame
from luna.network import LunaNetwork
from luna.replay_buffer import PrioritizedReplayBuffer, Trajectory


def _make_trajectory(game: ChessGame, length: int = 30) -> Trajectory:
    action_size = game.get_action_size()
    return Trajectory(
        observations=[np.random.randn(8, 8, OBS_PLANES).astype(np.float32) for _ in range(length)],
        actions=[np.random.randint(0, action_size) for _ in range(length)],
        rewards=[0.0] * (length - 1) + [1.0],
        root_policies=[np.random.dirichlet(np.ones(action_size)).astype(np.float32) for _ in range(length)],
        root_values=[np.random.uniform(-1, 1) for _ in range(length)],
        valids=[np.random.randint(0, 2, size=action_size).astype(np.float32) for _ in range(length)],
    )


def _make_small_learner() -> EzV2LearnerConfig:
    return EzV2LearnerConfig(num_channels=32, repr_blocks=2, dyn_blocks=1, proj_dim=64)


def bench_self_play(num_games: int = 3, num_sims: int = 5, max_ply: int | None = 80) -> None:
    """Benchmarks batched self-play via Coach.execute_episodes_batched.

    Vary ``parallel_games`` in ``TrainingRunConfig`` to measure throughput vs batch size.
    """
    game = ChessGame()
    nnet = LunaNetwork(game, _make_small_learner())
    run = TrainingRunConfig(
        num_mcts_sims=num_sims,
        cpuct=1.25,
        dir_noise=False,
        dir_alpha=0.3,
        discount=0.997,
        max_ply=max_ply,
        temp_threshold=12,
        parallel_games=min(4, num_games),
    )
    coach = Coach(game, nnet, run)

    total_positions = 0
    t0 = time.time()
    trajs = coach.execute_episodes_batched(num_games)
    for traj in trajs:
        total_positions += traj.game_length

    elapsed = time.time() - t0
    logger.info(
        "Self-play (batched, max_ply={}): {} positions in {:.2f}s = {:.1f} pos/s",
        max_ply,
        total_positions,
        elapsed,
        total_positions / elapsed,
    )


def bench_self_play_sequential(num_games: int = 3, num_sims: int = 5, max_ply: int | None = 80) -> None:
    """Benchmarks sequential self-play for comparison."""
    game = ChessGame()
    nnet = LunaNetwork(game, _make_small_learner())
    run = TrainingRunConfig(
        num_mcts_sims=num_sims,
        cpuct=1.25,
        dir_noise=False,
        dir_alpha=0.3,
        discount=0.997,
        max_ply=max_ply,
        temp_threshold=12,
    )
    coach = Coach(game, nnet, run)

    total_positions = 0
    t0 = time.time()
    for _ in range(num_games):
        traj = coach.execute_episode()
        total_positions += traj.game_length

    elapsed = time.time() - t0
    logger.info(
        "Self-play (sequential, max_ply={}): {} positions in {:.2f}s = {:.1f} pos/s",
        max_ply,
        total_positions,
        elapsed,
        total_positions / elapsed,
    )


def bench_learner(steps: int = 50) -> None:
    game = ChessGame()
    nnet = LunaNetwork(game, _make_small_learner())
    replay = PrioritizedReplayBuffer(capacity=10_000)

    for _ in range(20):
        replay.save_trajectory(_make_trajectory(game, length=30))

    t0 = time.time()
    nnet.train_ezv2(replay, steps=steps)
    elapsed = time.time() - t0
    logger.info("Learner: {} steps in {:.2f}s = {:.1f} steps/s", steps, elapsed, steps / elapsed)


def bench_inference(batch_sizes: list[int] | None = None) -> None:
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64]
    game = ChessGame()
    nnet = LunaNetwork(game, _make_small_learner())
    device = nnet.device

    for bs in batch_sizes:
        obs = torch.randn(bs, 8, 8, OBS_PLANES, device=device)
        valid = torch.ones(bs, game.get_action_size(), device=device)
        nnet.nnet.eval()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(100):
                nnet.nnet.initial_inference(obs, valid)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        logger.info(
            "Inference bs={}: 100 fwd in {:.3f}s = {:.0f} samples/s",
            bs,
            elapsed,
            100 * bs / elapsed,
        )


@dataclass
class BenchCliConfig:
    log_level: str = "INFO"
    selfplay_games: int = 2
    mcts_sims: int = 3
    max_ply: int | None = 80
    learner_steps: int = 20


def main() -> None:
    cfg = tyro.cli(BenchCliConfig)
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level.upper())

    probe = LunaNetwork(ChessGame(), _make_small_learner())
    logger.info("Device: {}\n", probe.device)
    logger.info("=== Inference Benchmark ===")
    bench_inference()
    logger.info("\n=== Self-Play Benchmark (batched) ===")
    bench_self_play(num_games=cfg.selfplay_games, num_sims=cfg.mcts_sims, max_ply=cfg.max_ply)
    logger.info("\n=== Self-Play Benchmark (sequential) ===")
    bench_self_play_sequential(num_games=cfg.selfplay_games, num_sims=cfg.mcts_sims, max_ply=cfg.max_ply)
    logger.info("\n=== Learner Benchmark ===")
    bench_learner(steps=cfg.learner_steps)


if __name__ == "__main__":
    main()
