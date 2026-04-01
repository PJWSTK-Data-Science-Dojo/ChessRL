"""Single-GPU throughput benchmarks for EfficientZeroV2 pipeline."""

from __future__ import annotations

import sys
import os
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from luna.game.luna_game import ChessGame
from luna.NNet import Luna_Network
from luna.mcts import MCTS
from luna.replay_buffer import PrioritizedReplayBuffer, Trajectory
from luna.utils import dotdict


def _make_trajectory(game: ChessGame, length: int = 30) -> Trajectory:
    action_size = game.getActionSize()
    return Trajectory(
        observations=[np.random.randn(8 * 8 * 6).astype(np.float32) for _ in range(length)],
        actions=[np.random.randint(0, action_size) for _ in range(length)],
        rewards=[0.0] * (length - 1) + [1.0],
        root_policies=[np.random.dirichlet(np.ones(action_size)).astype(np.float32) for _ in range(length)],
        root_values=[np.random.uniform(-1, 1) for _ in range(length)],
        valids=[np.random.randint(0, 2, size=action_size).astype(np.float32) for _ in range(length)],
    )


def bench_self_play(num_games: int = 3, num_sims: int = 5) -> None:
    game = ChessGame()
    nnet = Luna_Network(game)
    args = dotdict({"numMCTSSims": num_sims, "cpuct": 1.25, "dir_noise": False, "dir_alpha": 0.3})

    total_positions = 0
    t0 = time.time()
    for _ in range(num_games):
        mcts = MCTS(game, nnet, args)
        board = game.getInitBoard()
        curPlayer = 1
        steps = 0
        while game.getGameEnded(board, curPlayer) == 0 and steps < 80:
            canonical = game.getCanonicalForm(board, curPlayer)
            pi = mcts.getActionProb(canonical, temp=0 if steps > 10 else 1)
            action = int(np.argmax(pi)) if steps > 10 else int(np.random.choice(len(pi), p=pi))
            board, curPlayer = game.getNextState(board, curPlayer, action)
            steps += 1
            total_positions += 1

    elapsed = time.time() - t0
    print(f"Self-play: {total_positions} positions in {elapsed:.2f}s = {total_positions / elapsed:.1f} pos/s")


def bench_learner(steps: int = 50) -> None:
    game = ChessGame()
    nnet = Luna_Network(game)
    replay = PrioritizedReplayBuffer(capacity=10_000)

    for _ in range(20):
        replay.save_trajectory(_make_trajectory(game, length=30))

    t0 = time.time()
    nnet.train_ezv2(replay, steps=steps)
    elapsed = time.time() - t0
    print(f"Learner: {steps} steps in {elapsed:.2f}s = {steps / elapsed:.1f} steps/s")


def bench_inference(batch_sizes: list[int] | None = None) -> None:
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64]
    game = ChessGame()
    nnet = Luna_Network(game)
    device = nnet.device

    for bs in batch_sizes:
        obs = torch.randn(bs, 8, 8, 6, device=device)
        valid = torch.ones(bs, game.getActionSize(), device=device)
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
        print(f"Inference bs={bs}: 100 fwd in {elapsed:.3f}s = {100 * bs / elapsed:.0f} samples/s")


if __name__ == "__main__":
    from luna.NNet import _get_device

    print(f"Device: {_get_device()}\n")
    print("=== Inference Benchmark ===")
    bench_inference()
    print("\n=== Self-Play Benchmark ===")
    bench_self_play(num_games=2, num_sims=3)
    print("\n=== Learner Benchmark ===")
    bench_learner(steps=20)
