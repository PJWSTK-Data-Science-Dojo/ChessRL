"""Run a paired-opening MCTS gate between two Luna checkpoints."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro
from loguru import logger

from luna.config import MCTSParams
from luna.game.checkpoint_arena import (
    CHECKPOINT_ARENA_SCHEMA_VERSION,
    ArenaMCTSPlayer,
    CheckpointArenaPlayers,
    CheckpointArenaProtocol,
    CheckpointArenaResult,
    CheckpointIdentity,
    checkpoint_arena_payload,
    checkpoint_identity,
    run_checkpoint_arena,
    validate_checkpoint_arena_protocol,
    write_checkpoint_arena_result,
)
from luna.game.chess_game import ChessGame
from luna.game.opening_suite import OPENING_SUITE_VERSION
from luna.network import LunaNetwork

_GATE_FAILED_EXIT_CODE = 3


@dataclass(frozen=True, slots=True)
class EvalVsCheckpointCli:
    """Direct arena configuration; checkpoint B is the candidate under test."""

    checkpoint_a: Path
    checkpoint_b: Path
    tree_state_mode: Literal["latent", "exact"]
    output: Path | None = None
    games: int = 20
    num_mcts_sims: int = 32
    gumbel_max_considered_actions: int = 8
    max_ply: int = 256
    device: str = "cuda"
    cuda_device: int | None = 0
    compile_inference: bool = False
    log_level: str = "INFO"


def main() -> int:
    return run(tyro.cli(EvalVsCheckpointCli))


def run(config: EvalVsCheckpointCli) -> int:
    _configure_logging(config.log_level)
    try:
        output = _result_path(config)
        result = evaluate_checkpoints(config)
        output = write_checkpoint_arena_result(output, result)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError):
        logger.exception("Checkpoint arena failed")
        return 2
    _report(result, output)
    return 0 if result.checkpoint_b_passed else _GATE_FAILED_EXIT_CODE


def evaluate_checkpoints(config: EvalVsCheckpointCli) -> CheckpointArenaResult:
    protocol = _protocol(config)
    validate_checkpoint_arena_protocol(protocol)
    checkpoint_a = checkpoint_identity(config.checkpoint_a)
    checkpoint_b = checkpoint_identity(config.checkpoint_b)
    game = ChessGame()
    players = _load_players(game, protocol, config)
    scores = run_checkpoint_arena(game, players, protocol)
    _require_unchanged(checkpoint_a)
    _require_unchanged(checkpoint_b)
    return CheckpointArenaResult(checkpoint_a, checkpoint_b, protocol, scores)


def _protocol(config: EvalVsCheckpointCli) -> CheckpointArenaProtocol:
    mcts = MCTSParams(
        num_mcts_sims=config.num_mcts_sims,
        search_mode="gumbel",
        tree_state_mode=config.tree_state_mode,
        gumbel_max_considered_actions=config.gumbel_max_considered_actions,
        dir_noise=False,
    )
    return CheckpointArenaProtocol(
        schema_version=CHECKPOINT_ARENA_SCHEMA_VERSION,
        opening_suite_version=OPENING_SUITE_VERSION,
        games=config.games,
        max_ply=config.max_ply,
        minimum_checkpoint_b_score=0.5,
        mcts=mcts,
    )


def _load_players(
    game: ChessGame,
    protocol: CheckpointArenaProtocol,
    config: EvalVsCheckpointCli,
) -> CheckpointArenaPlayers:
    network_a = _load_network(game, config.checkpoint_a, config)
    network_b = _load_network(game, config.checkpoint_b, config)
    return CheckpointArenaPlayers(
        ArenaMCTSPlayer(game, network_a, protocol.mcts),
        ArenaMCTSPlayer(game, network_b, protocol.mcts),
    )


def _load_network(game: ChessGame, path: Path, config: EvalVsCheckpointCli) -> LunaNetwork:
    network = LunaNetwork.from_checkpoint(
        game,
        path,
        device=config.device,
        cuda_device=config.cuda_device,
        compile_inference=config.compile_inference,
        load_optimizer=False,
    )
    logger.info("Loaded checkpoint {}", path.expanduser().resolve())
    return network


def _result_path(config: EvalVsCheckpointCli) -> Path:
    checkpoint_b = config.checkpoint_b.expanduser().resolve()
    checkpoint_a_name = config.checkpoint_a.expanduser().resolve().name
    default = checkpoint_b.with_name(f"{checkpoint_b.name}.vs-{checkpoint_a_name}.arena.json")
    output = default if config.output is None else config.output.expanduser().resolve()
    checkpoint_a = config.checkpoint_a.expanduser().resolve()
    if output in {checkpoint_a, checkpoint_b}:
        raise ValueError("Arena output must not overwrite an input checkpoint")
    return output


def _require_unchanged(identity: CheckpointIdentity) -> None:
    current = checkpoint_identity(Path(identity.path))
    if current.sha256 != identity.sha256:
        raise RuntimeError(f"Checkpoint changed during evaluation: {identity.path}")


def _report(result: CheckpointArenaResult, output: Path) -> None:
    scores = result.scores
    logger.info(
        "Checkpoint arena: A {} — {} — {} B | B score={:.3f} passed={} | record={}",
        scores.checkpoint_a_wins,
        scores.draws,
        scores.checkpoint_b_wins,
        scores.checkpoint_b_score,
        result.checkpoint_b_passed,
        output,
    )
    sys.stdout.write(json.dumps(checkpoint_arena_payload(result), indent=2, sort_keys=True) + "\n")


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper())


if __name__ == "__main__":
    sys.exit(main())
