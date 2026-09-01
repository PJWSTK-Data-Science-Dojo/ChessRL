"""Persistent adaptive Fairy-Stockfish evaluation ladder."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import wandb
from loguru import logger

from luna.config import TrainingRunConfig
from luna.game.chess_game import ChessGame
from luna.game.stockfish_eval import (
    StockfishEvalScores,
    StockfishEvalSkipped,
    _wandb_metrics,
    ladder_match_settings,
    retry_stockfish_eval,
    run_stockfish_eval,
)
from luna.game.stockfish_ladder_state import (
    FAIRY_STOCKFISH_RELEASE,
    FAIRY_STOCKFISH_SOURCE_COMMIT,
    LADDER_STATE_NAME,
    FairyLadderState,
    _validate_state_progress,
    fairy_ladder_protocol,
    load_fairy_ladder_state,
)
from luna.network import LunaNetwork

__all__ = [
    "FAIRY_STOCKFISH_RELEASE",
    "FAIRY_STOCKFISH_SOURCE_COMMIT",
    "LADDER_STATE_NAME",
    "FairyLadderState",
    "fairy_ladder_protocol",
    "load_fairy_ladder_state",
    "run_fairy_ladder_eval",
    "write_fairy_ladder_state",
]

_SHA256_HEX_LENGTH = 64


def _checkpoint_digest(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
    return value


def write_fairy_ladder_state(path: Path, state: FairyLadderState) -> None:
    """Publish ladder progress atomically."""
    payload = {
        "protocol": state.protocol,
        "current_elo": state.current_elo,
        "highest_passed_elo": state.highest_passed_elo,
        "consecutive_passes": state.consecutive_passes,
        "completed": state.completed,
        "last_iteration": state.last_iteration,
        "last_checkpoint_sha256": state.last_checkpoint_sha256,
        "last_tested_elo": state.last_tested_elo,
        "last_scores": asdict(state.last_scores) if state.last_scores is not None else None,
        "evaluation_step": state.evaluation_step,
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _advance_state(
    state: FairyLadderState,
    scores: StockfishEvalScores,
    run: TrainingRunConfig,
    iteration: int,
    checkpoint_sha256: str,
) -> tuple[FairyLadderState, bool, bool]:
    passed = scores.model_wins > scores.stockfish_wins
    consecutive_passes = state.consecutive_passes + 1 if passed else 0
    advanced = passed and consecutive_passes >= run.ladder_required_passes
    current_elo = state.current_elo
    highest_passed = state.highest_passed_elo
    completed = state.completed
    if advanced:
        highest_passed = current_elo
        completed = current_elo == run.ladder_max_elo
        current_elo = min(current_elo + run.ladder_step_elo, run.ladder_max_elo)
        consecutive_passes = 0
    return (
        FairyLadderState(
            protocol=state.protocol,
            current_elo=current_elo,
            highest_passed_elo=highest_passed,
            consecutive_passes=consecutive_passes,
            completed=completed,
            last_iteration=iteration,
            last_checkpoint_sha256=checkpoint_sha256,
            last_tested_elo=state.current_elo,
            last_scores=scores,
            evaluation_step=state.evaluation_step + 1,
        ),
        passed,
        advanced,
    )


def _log_ladder_state(
    state: FairyLadderState,
    *,
    iteration: int,
    duration_seconds: float | None = None,
) -> None:
    if wandb.run is None or state.last_scores is None or state.last_tested_elo is None:
        return
    passed = state.last_scores.model_wins > state.last_scores.stockfish_wins
    advanced = passed and state.consecutive_passes == 0 and state.highest_passed_elo == state.last_tested_elo
    metrics = _wandb_metrics(
        state.last_scores,
        iteration,
        prefix="ladder",
        opponent_elo=state.last_tested_elo,
        duration_seconds=duration_seconds,
    )
    metrics.update(
        {
            "ladder/evaluation_step": state.evaluation_step,
            "ladder/tested_elo": state.last_tested_elo,
            "ladder/current_elo": state.current_elo,
            "ladder/has_passed_rung": int(state.highest_passed_elo is not None),
            "ladder/passed": int(passed),
            "ladder/advanced": int(advanced),
            "ladder/consecutive_passes": state.consecutive_passes,
            "ladder/completed": int(state.completed),
        }
    )
    if state.highest_passed_elo is not None:
        metrics["ladder/highest_passed_elo"] = state.highest_passed_elo
    wandb.log(metrics)


def run_fairy_ladder_eval(
    game: ChessGame,
    nnet: LunaNetwork,
    run: TrainingRunConfig,
    *,
    iteration: int,
    checkpoint_sha256: str,
    state_required: bool = False,
) -> FairyLadderState:
    """Evaluate one rung, persist its decision, and log independently of best promotion."""
    if isinstance(iteration, bool) or iteration < 1:
        raise ValueError("iteration must be a positive integer")
    digest = _checkpoint_digest(checkpoint_sha256)
    folder = Path(run.checkpoint).expanduser().resolve()
    state_path = folder / LADDER_STATE_NAME
    state = load_fairy_ladder_state(state_path, run, required=state_required)
    if state.last_iteration is not None and iteration < state.last_iteration:
        raise RuntimeError("Fairy ladder cannot evaluate a checkpoint older than its last recorded iteration")
    if state.last_iteration == iteration and state.last_checkpoint_sha256 != digest:
        raise RuntimeError("Fairy ladder checkpoint changed for an already evaluated iteration")

    if state.last_iteration == iteration and state.last_checkpoint_sha256 == digest:
        if state.last_tested_elo is None:
            raise RuntimeError("Fairy ladder duplicate key is missing last_tested_elo")
        previous_key = (state.last_iteration, state.last_checkpoint_sha256, state.last_tested_elo)
        evaluation_key = (iteration, digest, state.last_tested_elo)
        if evaluation_key != previous_key:
            raise RuntimeError("Fairy ladder duplicate evaluation key is inconsistent")
        logger.info(
            "Fairy ladder: checkpoint already evaluated at iteration {} and Elo {}; skipping duplicate",
            iteration,
            state.last_tested_elo,
        )
        _log_ladder_state(state, iteration=iteration)
        return state
    if state.completed:
        return state
    tested_elo = state.current_elo
    started_at = time.perf_counter()
    outcome = retry_stockfish_eval(
        lambda: run_stockfish_eval(
            game,
            nnet,
            run,
            iteration=iteration,
            settings=ladder_match_settings(run, tested_elo),
            metric_prefix=None,
        ),
        attempts=run.external_eval_attempts,
        retry_seconds=run.external_eval_retry_seconds,
    )
    if isinstance(outcome, StockfishEvalSkipped):
        raise RuntimeError(f"Fairy ladder evaluation did not complete ({outcome.reason}): {outcome.message}")
    next_state, passed, advanced = _advance_state(state, outcome, run, iteration, digest)
    _validate_state_progress(next_state, run)
    write_fairy_ladder_state(state_path, next_state)
    logger.info(
        "Fairy ladder: tested={} passed={} confirmation={}/{} advanced={} next={} highest={}",
        tested_elo,
        passed,
        next_state.consecutive_passes,
        run.ladder_required_passes,
        advanced,
        next_state.current_elo,
        next_state.highest_passed_elo,
    )
    _log_ladder_state(
        next_state,
        iteration=iteration,
        duration_seconds=time.perf_counter() - started_at,
    )
    return next_state
