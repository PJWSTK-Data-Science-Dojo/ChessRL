"""Durable completion state for the fixed external benchmark."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from luna.game.stockfish_eval import StockfishEvalScores

BENCHMARK_STATE_NAME = "benchmark_state.json"
_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_STATE_FIELDS = {
    "schema_version",
    "protocol",
    "last_iteration",
    "last_checkpoint_sha256",
    "last_scores",
    "evaluation_step",
}
_SCORE_FIELDS = {"model_wins", "draws", "stockfish_wins"}


@dataclass(frozen=True)
class BenchmarkState:
    """Latest durable fixed-benchmark result under one immutable protocol."""

    protocol: dict[str, object]
    last_iteration: int | None
    last_checkpoint_sha256: str | None
    last_scores: StockfishEvalScores | None
    evaluation_step: int


def _validate_json_value(value: object, field: str) -> None:
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must not contain non-finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{field} object keys must be strings")
        for key, item in value.items():
            _validate_json_value(item, f"{field}.{key}")
        return
    raise ValueError(f"{field} contains a non-JSON value of type {type(value).__name__}")


def _normalized_protocol(protocol: dict[str, object]) -> dict[str, object]:
    _validate_json_value(protocol, "protocol")
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
    decoded: object = json.loads(encoded)
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise ValueError("protocol must be a JSON object with string keys")
    return cast(dict[str, object], decoded)


def _required_int(payload: dict[str, object], name: str, minimum: int) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"Benchmark state field {name!r} must be an integer of at least {minimum}")
    return value


def _optional_int(payload: dict[str, object], name: str, minimum: int) -> int | None:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"Benchmark state field {name!r} must be null or an integer of at least {minimum}")
    return value


def _validate_sha256(value: object, *, optional: bool) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        qualifier = "null or " if optional else ""
        raise RuntimeError(f"Benchmark checkpoint SHA256 must be {qualifier}64 lowercase hexadecimal characters")
    return value


def _parse_scores(value: object) -> StockfishEvalScores | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError("Benchmark state last_scores must be an object or null")
    payload = cast(dict[str, object], value)
    if set(payload) != _SCORE_FIELDS:
        raise RuntimeError(f"Benchmark state last_scores fields must be exactly {sorted(_SCORE_FIELDS)}")
    scores = StockfishEvalScores(
        model_wins=_required_int(payload, "model_wins", 0),
        draws=_required_int(payload, "draws", 0),
        stockfish_wins=_required_int(payload, "stockfish_wins", 0),
    )
    if scores.model_wins + scores.draws + scores.stockfish_wins == 0:
        raise RuntimeError("Benchmark state last_scores must contain at least one game")
    return scores


def _initial_state(protocol: dict[str, object]) -> BenchmarkState:
    return BenchmarkState(
        protocol=protocol,
        last_iteration=None,
        last_checkpoint_sha256=None,
        last_scores=None,
        evaluation_step=0,
    )


def _parse_state(decoded: object, expected_protocol: dict[str, object], path: Path) -> BenchmarkState:
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise RuntimeError(f"Benchmark state must be a JSON object: {path}")
    payload = cast(dict[str, object], decoded)
    if set(payload) != _STATE_FIELDS:
        raise RuntimeError(f"Benchmark state fields must be exactly {sorted(_STATE_FIELDS)}: {path}")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != _SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported benchmark state schema version in {path}")
    stored_protocol = payload.get("protocol")
    if not isinstance(stored_protocol, dict) or not all(isinstance(key, str) for key in stored_protocol):
        raise RuntimeError(f"Benchmark state protocol must be a JSON object: {path}")
    try:
        normalized_stored_protocol = _normalized_protocol(cast(dict[str, object], stored_protocol))
    except ValueError as exc:
        raise RuntimeError(f"Benchmark state protocol is invalid: {path}") from exc
    if normalized_stored_protocol != expected_protocol:
        raise RuntimeError(
            f"Benchmark protocol differs from {path}; use a new checkpoint directory for this benchmark contract"
        )

    state = BenchmarkState(
        protocol=expected_protocol,
        last_iteration=_optional_int(payload, "last_iteration", 1),
        last_checkpoint_sha256=_validate_sha256(payload.get("last_checkpoint_sha256"), optional=True),
        last_scores=_parse_scores(payload.get("last_scores")),
        evaluation_step=_required_int(payload, "evaluation_step", 0),
    )
    _validate_state_consistency(state)
    return state


def _validate_state_consistency(state: BenchmarkState) -> None:
    record_fields = (state.last_iteration, state.last_checkpoint_sha256, state.last_scores)
    present_count = sum(value is not None for value in record_fields)
    if present_count not in (0, len(record_fields)):
        raise RuntimeError("Benchmark state result fields must either all be null or all be present")
    if present_count == 0 and state.evaluation_step != 0:
        raise RuntimeError("Empty benchmark state must have evaluation_step 0")
    if present_count and state.evaluation_step < 1:
        raise RuntimeError("Recorded benchmark state must have a positive evaluation_step")
    if state.last_iteration is not None and state.evaluation_step > state.last_iteration:
        raise RuntimeError("Benchmark evaluation_step cannot exceed last_iteration")


def load_benchmark_state(
    path: Path,
    protocol: dict[str, object],
    *,
    required: bool = False,
) -> BenchmarkState:
    """Load a compatible sidecar or return a new in-memory state.

    When ``required`` is true, a missing sidecar is treated as lost managed state
    rather than as permission to reset benchmark progress.
    """
    normalized_protocol = _normalized_protocol(protocol)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required benchmark state is missing: {path}")
        return _initial_state(normalized_protocol)
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read benchmark state: {path}") from exc
    return _parse_state(decoded, normalized_protocol, path)


def _state_payload(state: BenchmarkState) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "protocol": state.protocol,
        "last_iteration": state.last_iteration,
        "last_checkpoint_sha256": state.last_checkpoint_sha256,
        "last_scores": asdict(state.last_scores) if state.last_scores is not None else None,
        "evaluation_step": state.evaluation_step,
    }


def write_benchmark_state(path: Path, state: BenchmarkState) -> None:
    """Atomically and durably publish validated benchmark state."""
    normalized_protocol = _normalized_protocol(state.protocol)
    validated = _parse_state(_state_payload(state), normalized_protocol, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_state_payload(validated), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def record_benchmark_result(
    path: Path,
    protocol: dict[str, object],
    *,
    iteration: int,
    checkpoint_sha256: str,
    scores: StockfishEvalScores,
) -> BenchmarkState:
    """Record one result, making exact retries no-ops and rejecting conflicts."""
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise ValueError("Benchmark iteration must be a positive integer")
    try:
        normalized_sha256 = _validate_sha256(checkpoint_sha256, optional=False)
        parsed_scores = _parse_scores(asdict(scores))
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    if normalized_sha256 is None or parsed_scores is None:
        raise AssertionError("Validated benchmark result fields cannot be null")

    state = load_benchmark_state(path, protocol, required=False)
    if state.last_iteration is not None:
        if iteration < state.last_iteration:
            raise RuntimeError(
                f"Benchmark result iteration {iteration} is older than recorded iteration {state.last_iteration}"
            )
        if iteration == state.last_iteration:
            if state.last_checkpoint_sha256 == normalized_sha256 and state.last_scores == parsed_scores:
                return state
            raise RuntimeError(f"Conflicting benchmark result already exists for iteration {iteration}")

    next_state = BenchmarkState(
        protocol=state.protocol,
        last_iteration=iteration,
        last_checkpoint_sha256=normalized_sha256,
        last_scores=parsed_scores,
        evaluation_step=state.evaluation_step + 1,
    )
    write_benchmark_state(path, next_state)
    return next_state
