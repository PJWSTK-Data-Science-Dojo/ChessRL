"""Tests for durable fixed-benchmark completion state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from luna.game.benchmark_state import (
    BENCHMARK_STATE_NAME,
    BenchmarkState,
    load_benchmark_state,
    record_benchmark_result,
    write_benchmark_state,
)
from luna.game.stockfish_eval import StockfishEvalScores

_CHECKPOINT_SHA = "a" * 64
_OTHER_CHECKPOINT_SHA = "b" * 64


@pytest.fixture
def protocol() -> dict[str, object]:
    return {
        "schema_version": 1,
        "engine": {"name": "Stockfish", "sha256": "c" * 64},
        "games": 20,
        "opening_ids": [1, 2, 3],
    }


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "run" / BENCHMARK_STATE_NAME


def test_result_round_trips_and_advances_monotonically(
    state_path: Path,
    protocol: dict[str, object],
) -> None:
    first_scores = StockfishEvalScores(model_wins=4, draws=6, stockfish_wins=10)
    second_scores = StockfishEvalScores(model_wins=7, draws=7, stockfish_wins=6)

    initial = load_benchmark_state(state_path, protocol)
    first = record_benchmark_result(
        state_path,
        protocol,
        iteration=25,
        checkpoint_sha256=_CHECKPOINT_SHA,
        scores=first_scores,
    )
    second = record_benchmark_result(
        state_path,
        protocol,
        iteration=50,
        checkpoint_sha256=_OTHER_CHECKPOINT_SHA,
        scores=second_scores,
    )
    restored = load_benchmark_state(state_path, protocol, required=True)

    assert initial == BenchmarkState(
        protocol=protocol, last_iteration=None, last_checkpoint_sha256=None, last_scores=None, evaluation_step=0
    )
    assert not state_path.with_name(f".{state_path.name}.tmp").exists()
    assert first.evaluation_step == 1
    assert second == restored
    assert restored.last_iteration == 50
    assert restored.last_checkpoint_sha256 == _OTHER_CHECKPOINT_SHA
    assert restored.last_scores == second_scores
    assert restored.evaluation_step == 2


def test_load_returns_protocol_copy(state_path: Path, protocol: dict[str, object]) -> None:
    state = load_benchmark_state(state_path, protocol)

    protocol["games"] = 2

    assert state.protocol["games"] == 20


def test_missing_required_state_fails_closed(state_path: Path, protocol: dict[str, object]) -> None:
    with pytest.raises(FileNotFoundError, match="Required benchmark state is missing"):
        load_benchmark_state(state_path, protocol, required=True)


@pytest.mark.parametrize(
    "contents",
    [
        "not json",
        json.dumps(
            {
                "schema_version": 1,
                "protocol": {},
                "last_iteration": None,
                "last_checkpoint_sha256": None,
                "last_scores": None,
            }
        ),
        json.dumps(
            {
                "schema_version": 1,
                "protocol": {},
                "last_iteration": 25,
                "last_checkpoint_sha256": _CHECKPOINT_SHA,
                "last_scores": None,
                "evaluation_step": 1,
            }
        ),
    ],
)
def test_corrupt_state_is_rejected(
    state_path: Path,
    protocol: dict[str, object],
    contents: str,
) -> None:
    state_path.parent.mkdir(parents=True)
    state_path.write_text(contents, encoding="utf-8")

    with pytest.raises(RuntimeError):
        load_benchmark_state(state_path, protocol, required=True)


def test_changed_protocol_is_rejected(state_path: Path, protocol: dict[str, object]) -> None:
    scores = StockfishEvalScores(model_wins=4, draws=6, stockfish_wins=10)
    record_benchmark_result(
        state_path,
        protocol,
        iteration=25,
        checkpoint_sha256=_CHECKPOINT_SHA,
        scores=scores,
    )
    changed_protocol = {**protocol, "games": 10}

    with pytest.raises(RuntimeError, match="Benchmark protocol differs"):
        load_benchmark_state(state_path, changed_protocol, required=True)


def test_exact_duplicate_is_a_no_op(state_path: Path, protocol: dict[str, object]) -> None:
    scores = StockfishEvalScores(model_wins=4, draws=6, stockfish_wins=10)
    recorded = record_benchmark_result(
        state_path,
        protocol,
        iteration=25,
        checkpoint_sha256=_CHECKPOINT_SHA,
        scores=scores,
    )

    with patch("luna.game.benchmark_state.write_benchmark_state") as write_state:
        duplicate = record_benchmark_result(
            state_path,
            protocol,
            iteration=25,
            checkpoint_sha256=_CHECKPOINT_SHA,
            scores=scores,
        )

    assert duplicate == recorded
    write_state.assert_not_called()


@pytest.mark.parametrize(
    ("checkpoint_sha256", "scores"),
    [
        (_OTHER_CHECKPOINT_SHA, StockfishEvalScores(model_wins=4, draws=6, stockfish_wins=10)),
        (_CHECKPOINT_SHA, StockfishEvalScores(model_wins=5, draws=5, stockfish_wins=10)),
    ],
)
def test_conflicting_duplicate_is_rejected(
    state_path: Path,
    protocol: dict[str, object],
    checkpoint_sha256: str,
    scores: StockfishEvalScores,
) -> None:
    original_scores = StockfishEvalScores(model_wins=4, draws=6, stockfish_wins=10)
    original = record_benchmark_result(
        state_path,
        protocol,
        iteration=25,
        checkpoint_sha256=_CHECKPOINT_SHA,
        scores=original_scores,
    )

    with pytest.raises(RuntimeError, match="Conflicting benchmark result"):
        record_benchmark_result(
            state_path,
            protocol,
            iteration=25,
            checkpoint_sha256=checkpoint_sha256,
            scores=scores,
        )

    assert load_benchmark_state(state_path, protocol, required=True) == original


def test_atomic_write_fsyncs_file_and_directory(
    state_path: Path,
    protocol: dict[str, object],
) -> None:
    state = BenchmarkState(
        protocol=protocol,
        last_iteration=25,
        last_checkpoint_sha256=_CHECKPOINT_SHA,
        last_scores=StockfishEvalScores(model_wins=4, draws=6, stockfish_wins=10),
        evaluation_step=1,
    )

    with patch("luna.game.benchmark_state.os.fsync", wraps=os.fsync) as fsync:
        write_benchmark_state(state_path, state)

    assert fsync.call_count == 2
    assert load_benchmark_state(state_path, protocol, required=True) == state


def test_failed_replace_cleans_temporary_file(
    state_path: Path,
    protocol: dict[str, object],
) -> None:
    state = BenchmarkState(
        protocol=protocol,
        last_iteration=25,
        last_checkpoint_sha256=_CHECKPOINT_SHA,
        last_scores=StockfishEvalScores(model_wins=4, draws=6, stockfish_wins=10),
        evaluation_step=1,
    )

    with (
        patch("luna.game.benchmark_state.os.replace", side_effect=OSError("replace failed")),
        pytest.raises(OSError, match="replace failed"),
    ):
        write_benchmark_state(state_path, state)

    assert not state_path.exists()
    assert list(state_path.parent.glob(f".{state_path.name}.tmp-*")) == []
