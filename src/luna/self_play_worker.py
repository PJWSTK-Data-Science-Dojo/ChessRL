"""Self-play actor process protocol and worker entry point."""

from __future__ import annotations

import os
import random
import traceback
from contextlib import suppress
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import torch

from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.replay_buffer import Trajectory


@dataclass(frozen=True)
class _CollectRequest:
    generation: int
    seed: int
    episode_count: int
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class _StopRequest:
    pass


@dataclass(frozen=True)
class _ActorReady:
    actor_id: int


@dataclass(frozen=True)
class _ActorTrajectory:
    actor_id: int
    generation: int
    episode_index: int
    trajectory: Trajectory


@dataclass(frozen=True)
class _ActorCollectionDone:
    actor_id: int
    generation: int
    episode_count: int


@dataclass(frozen=True)
class _ActorFailure:
    actor_id: int
    generation: int | None
    exception_type: str
    message: str
    traceback_text: str


_ActorRequest = _CollectRequest | _StopRequest
_ActorResponse = _ActorReady | _ActorTrajectory | _ActorCollectionDone | _ActorFailure


def derive_actor_seed(base_seed: int, actor_id: int, generation: int) -> int:
    """Derive a repeatable, unique uint32 seed for one actor collection."""
    if not 0 <= actor_id < 2**32:
        raise ValueError("actor_id must fit an unsigned 32-bit integer")
    if generation < 0:
        raise ValueError("generation must be non-negative")
    normalized_base = base_seed % (2**32)
    sequence = np.random.SeedSequence([normalized_base, generation])
    generation_seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
    # The multiplier is odd, making actor IDs a bijection modulo 2**32 for
    # each generation rather than merely relying on a low collision chance.
    return (generation_seed + actor_id * 0x9E3779B1) % (2**32)


def partition_episode_counts(num_episodes: int, worker_count: int) -> list[int]:
    """Split episodes evenly while assigning every worker at least one game."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    active_workers = min(num_episodes, worker_count)
    quotient, remainder = divmod(num_episodes, active_workers)
    return [quotient + int(actor_id < remainder) for actor_id in range(active_workers)]


def seed_self_play_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _actor_learner_config(learner: EzV2LearnerConfig) -> EzV2LearnerConfig:
    """Return the eager-only runtime configuration for a self-play actor."""
    return replace(
        learner,
        compile_inference=False,
        compile_training=False,
        dataloader_workers=0,
        expert_anchor_path="",
        expert_anchor_fingerprint="",
        expert_anchor_fraction=0.0,
        expert_anchor_loss_weight=0.0,
    )


def _actor_entry(
    actor_id: int,
    connection: Connection,
    run: TrainingRunConfig,
    learner: EzV2LearnerConfig,
    base_seed: int,
    cache_dir: str,
) -> None:
    generation: int | None = None
    try:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(1)
        seed_self_play_rng(derive_actor_seed(base_seed, actor_id, 0))

        # Delayed imports avoid a module cycle when Coach creates this pool.
        from luna.coach import Coach
        from luna.game.chess_game import ChessGame
        from luna.network import LunaNetwork

        actor_run = replace(run, self_play_workers=1, profile=False)
        actor_learner = _actor_learner_config(learner)
        game = ChessGame()
        network = LunaNetwork(game, actor_learner)
        coach = Coach(game, network, actor_run, seed=derive_actor_seed(base_seed, actor_id, 0))
        warmed_up = False
        connection.send(_ActorReady(actor_id))

        while True:
            request = connection.recv()
            if isinstance(request, _StopRequest):
                return
            if not isinstance(request, _CollectRequest):
                raise TypeError(f"Actor {actor_id} received an unsupported request: {type(request).__name__}")

            generation = request.generation
            if request.episode_count <= 0:
                raise ValueError("Actor collection must request at least one episode")
            episode_count = request.episode_count
            seed_self_play_rng(request.seed)
            network.nnet.load_state_dict(request.state_dict, strict=True)
            del request
            if not warmed_up:
                network.warmup_mcts_inference(game)
                warmed_up = True
            trajectories = coach.execute_episodes_batched(episode_count, progress=False)
            while trajectories:
                episode_index = len(trajectories) - 1
                trajectory = trajectories.pop()
                connection.send(_ActorTrajectory(actor_id, generation, episode_index, trajectory))
                del trajectory
            connection.send(_ActorCollectionDone(actor_id, generation, episode_count))
    except (AssertionError, EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        failure = _ActorFailure(
            actor_id=actor_id,
            generation=generation,
            exception_type=type(exc).__name__,
            message=str(exc),
            traceback_text=traceback.format_exc(),
        )
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send(failure)
        raise
    finally:
        connection.close()
