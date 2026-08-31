"""Persistent adaptive Fairy-Stockfish evaluation ladder."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import wandb
from loguru import logger

from luna.config import TrainingRunConfig, evaluation_mcts_params
from luna.game.chess_game import ChessGame
from luna.game.stockfish_eval import (
    OPENING_SUITE_VERSION,
    StockfishEvalScores,
    StockfishEvalSkipped,
    _wandb_metrics,
    engine_binary_sha256,
    ladder_match_settings,
    retry_stockfish_eval,
    run_stockfish_eval,
)
from luna.network import LunaNetwork

FAIRY_STOCKFISH_RELEASE = "fairy_sf_14"
FAIRY_STOCKFISH_SOURCE_COMMIT = "f3e6969d11d1bec17eba26e7ae0e629ad4af71dd"
LADDER_STATE_NAME = "fairy_ladder.json"
_LADDER_SCHEMA_VERSION = 2
_SHA256_HEX_LENGTH = 64
_STATE_FIELDS = frozenset(
    {
        "protocol",
        "current_elo",
        "highest_passed_elo",
        "consecutive_passes",
        "completed",
        "last_iteration",
        "last_checkpoint_sha256",
        "last_tested_elo",
        "last_scores",
        "evaluation_step",
    }
)


@dataclass(frozen=True)
class FairyLadderState:
    """Durable progress through one immutable Fairy-Stockfish protocol."""

    protocol: dict[str, object]
    current_elo: int
    highest_passed_elo: int | None
    consecutive_passes: int
    completed: bool
    last_iteration: int | None
    last_checkpoint_sha256: str | None
    last_tested_elo: int | None
    last_scores: StockfishEvalScores | None
    evaluation_step: int


def fairy_ladder_protocol(run: TrainingRunConfig) -> dict[str, object]:
    """Capture every setting that affects ladder comparability and progression."""
    return {
        "schema_version": _LADDER_SCHEMA_VERSION,
        "engine_release": FAIRY_STOCKFISH_RELEASE,
        "engine_source_commit": FAIRY_STOCKFISH_SOURCE_COMMIT,
        "engine_binary_sha256": engine_binary_sha256(run.ladder_path),
        "engine_path": run.ladder_path,
        "opening_suite_version": OPENING_SUITE_VERSION,
        "games": run.ladder_eval_games,
        "depth": run.ladder_depth,
        "max_ply": run.ladder_eval_max_ply,
        "start_elo": run.ladder_start_elo,
        "step_elo": run.ladder_step_elo,
        "max_elo": run.ladder_max_elo,
        "required_passes": run.ladder_required_passes,
        "mcts": asdict(evaluation_mcts_params(run)),
    }


def _required_int(payload: dict[str, object], name: str, minimum: int) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"Fairy ladder state field {name!r} must be an integer of at least {minimum}")
    return value


def _optional_int(payload: dict[str, object], name: str, minimum: int) -> int | None:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"Fairy ladder state field {name!r} must be null or an integer of at least {minimum}")
    return value


def _optional_sha256(payload: dict[str, object], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"Fairy ladder state field {name!r} must be a lowercase SHA-256 digest or null")
    return value


def _checkpoint_digest(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
    return value


def _parse_scores(value: object) -> StockfishEvalScores | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError("Fairy ladder last_scores must be an object or null")
    payload = cast(dict[str, object], value)
    if set(payload) != {"model_wins", "draws", "stockfish_wins"}:
        raise RuntimeError("Fairy ladder last_scores fields do not match its schema")
    return StockfishEvalScores(
        model_wins=_required_int(payload, "model_wins", 0),
        draws=_required_int(payload, "draws", 0),
        stockfish_wins=_required_int(payload, "stockfish_wins", 0),
    )


def _new_state(run: TrainingRunConfig, protocol: dict[str, object]) -> FairyLadderState:
    return FairyLadderState(
        protocol=protocol,
        current_elo=run.ladder_start_elo,
        highest_passed_elo=None,
        consecutive_passes=0,
        completed=False,
        last_iteration=None,
        last_checkpoint_sha256=None,
        last_tested_elo=None,
        last_scores=None,
        evaluation_step=0,
    )


def load_fairy_ladder_state(
    path: Path,
    run: TrainingRunConfig,
    *,
    required: bool = False,
) -> FairyLadderState:
    """Load ladder state, optionally requiring evidence of prior progress."""
    protocol = fairy_ladder_protocol(run)
    if not path.exists():
        if required:
            raise RuntimeError(f"Required Fairy ladder state is missing: {path}")
        return _new_state(run, protocol)
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read Fairy ladder state: {path}") from exc
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise RuntimeError(f"Fairy ladder state must be a JSON object: {path}")
    payload = cast(dict[str, object], decoded)
    if set(payload) != _STATE_FIELDS:
        raise RuntimeError(f"Fairy ladder state fields do not match schema version {_LADDER_SCHEMA_VERSION}: {path}")
    if payload.get("protocol") != protocol:
        raise RuntimeError(
            f"Fairy ladder protocol differs from {path}; use a new checkpoint directory for this ladder contract"
        )
    completed = payload.get("completed")
    if not isinstance(completed, bool):
        raise RuntimeError("Fairy ladder state field 'completed' must be boolean")
    state = FairyLadderState(
        protocol=protocol,
        current_elo=_required_int(payload, "current_elo", run.ladder_start_elo),
        highest_passed_elo=_optional_int(payload, "highest_passed_elo", run.ladder_start_elo),
        consecutive_passes=_required_int(payload, "consecutive_passes", 0),
        completed=completed,
        last_iteration=_optional_int(payload, "last_iteration", 1),
        last_checkpoint_sha256=_optional_sha256(payload, "last_checkpoint_sha256"),
        last_tested_elo=_optional_int(payload, "last_tested_elo", run.ladder_start_elo),
        last_scores=_parse_scores(payload.get("last_scores")),
        evaluation_step=_required_int(payload, "evaluation_step", 0),
    )
    _validate_state_progress(state, run)
    return state


def _validate_state_progress(state: FairyLadderState, run: TrainingRunConfig) -> None:
    if not run.ladder_start_elo <= state.current_elo <= run.ladder_max_elo:
        raise RuntimeError("Fairy ladder current_elo is outside the configured ladder")
    if (state.current_elo - run.ladder_start_elo) % run.ladder_step_elo:
        raise RuntimeError("Fairy ladder current_elo is not aligned to ladder_step_elo")
    if state.highest_passed_elo is not None:
        if not run.ladder_start_elo <= state.highest_passed_elo <= run.ladder_max_elo:
            raise RuntimeError("Fairy ladder highest_passed_elo is outside the configured ladder")
        if (state.highest_passed_elo - run.ladder_start_elo) % run.ladder_step_elo:
            raise RuntimeError("Fairy ladder highest_passed_elo is not aligned to ladder_step_elo")
    if state.consecutive_passes >= run.ladder_required_passes:
        raise RuntimeError("Fairy ladder consecutive_passes must reset after an advance")

    last_fields = (
        state.last_iteration,
        state.last_checkpoint_sha256,
        state.last_tested_elo,
        state.last_scores,
    )
    if state.evaluation_step == 0:
        if any(value is not None for value in last_fields):
            raise RuntimeError("Unevaluated Fairy ladder state cannot contain a last evaluation")
        if (
            state.current_elo != run.ladder_start_elo
            or state.highest_passed_elo is not None
            or state.consecutive_passes != 0
            or state.completed
        ):
            raise RuntimeError("Unevaluated Fairy ladder state must be at the configured first rung")
        return
    if any(value is None for value in last_fields):
        raise RuntimeError("Evaluated Fairy ladder state must contain the complete last-evaluation key and scores")

    if state.last_iteration is None or state.last_tested_elo is None or state.last_scores is None:
        raise RuntimeError("Evaluated Fairy ladder state is missing required progress fields")
    if state.evaluation_step > state.last_iteration:
        raise RuntimeError("Fairy ladder evaluation_step cannot exceed last_iteration")
    if not run.ladder_start_elo <= state.last_tested_elo <= run.ladder_max_elo:
        raise RuntimeError("Fairy ladder last_tested_elo is outside the configured ladder")
    if (state.last_tested_elo - run.ladder_start_elo) % run.ladder_step_elo:
        raise RuntimeError("Fairy ladder last_tested_elo is not aligned to ladder_step_elo")

    total = state.last_scores.model_wins + state.last_scores.draws + state.last_scores.stockfish_wins
    if total != run.ladder_eval_games:
        raise RuntimeError("Fairy ladder last_scores game count differs from its protocol")

    if state.highest_passed_elo is None:
        if state.current_elo != run.ladder_start_elo:
            raise RuntimeError("Fairy ladder cannot leave its first rung before passing it")
        completed_rungs = 0
    else:
        completed_rungs = (state.highest_passed_elo - run.ladder_start_elo) // run.ladder_step_elo + 1
        if state.completed:
            if state.highest_passed_elo != run.ladder_max_elo or state.current_elo != run.ladder_max_elo:
                raise RuntimeError("Completed Fairy ladder state must have passed ladder_max_elo")
        elif (
            state.highest_passed_elo == run.ladder_max_elo
            or state.current_elo != state.highest_passed_elo + run.ladder_step_elo
        ):
            raise RuntimeError("Fairy ladder current_elo must immediately follow highest_passed_elo")

    passed = state.last_scores.model_wins > state.last_scores.stockfish_wins
    if state.completed and state.consecutive_passes != 0:
        raise RuntimeError("Completed Fairy ladder state cannot retain pass confirmations")

    if state.consecutive_passes > 0:
        if not passed or state.last_tested_elo != state.current_elo or state.completed:
            raise RuntimeError("Fairy ladder pass confirmations are inconsistent with the last result")
    elif passed:
        if state.highest_passed_elo != state.last_tested_elo:
            raise RuntimeError("A confirmed Fairy ladder pass must record the tested rung as passed")
    elif state.last_tested_elo != state.current_elo or state.completed:
        raise RuntimeError("A non-passing Fairy ladder result cannot advance or complete the ladder")

    minimum_evaluations = completed_rungs * run.ladder_required_passes + state.consecutive_passes + int(not passed)
    if state.evaluation_step < minimum_evaluations:
        raise RuntimeError("Fairy ladder evaluation_step is too small for its recorded progress")


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
