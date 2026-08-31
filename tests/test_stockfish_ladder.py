"""Tests for persistent adaptive Fairy-Stockfish ladder state."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from luna.config import TrainingRunConfig
from luna.game.chess_game import ChessGame
from luna.game.stockfish_eval import StockfishEvalScores, StockfishEvalSkipped
from luna.game.stockfish_ladder import (
    LADDER_STATE_NAME,
    FairyLadderState,
    load_fairy_ladder_state,
    run_fairy_ladder_eval,
    write_fairy_ladder_state,
)
from luna.network import LunaNetwork


@pytest.fixture
def ladder_run(tmp_path: Path) -> TrainingRunConfig:
    engine_path = tmp_path / "fairy-stockfish"
    engine_path.write_bytes(b"pinned-fairy-stockfish-binary")
    checkpoint_path = tmp_path / "run"
    checkpoint_path.mkdir()
    return TrainingRunConfig(
        checkpoint=str(checkpoint_path),
        ladder_eval_every=5,
        ladder_eval_games=20,
        ladder_start_elo=500,
        ladder_step_elo=100,
        ladder_max_elo=700,
        ladder_required_passes=2,
        ladder_path=str(engine_path),
        evaluation_num_mcts_sims=1,
    )


def _state_path(run: TrainingRunConfig) -> Path:
    return Path(run.checkpoint) / LADDER_STATE_NAME


def _checkpoint_digest(iteration: int) -> str:
    return hashlib.sha256(f"checkpoint-{iteration}".encode()).hexdigest()


def _run_evaluation(run: TrainingRunConfig, scores: StockfishEvalScores, iteration: int) -> FairyLadderState:
    game = cast(ChessGame, object())
    network = cast(LunaNetwork, object())
    with (
        patch("luna.game.stockfish_ladder.run_stockfish_eval", return_value=scores),
        patch("luna.game.stockfish_ladder.wandb.run", None),
    ):
        return run_fairy_ladder_eval(
            game,
            network,
            run,
            iteration=iteration,
            checkpoint_sha256=_checkpoint_digest(iteration),
        )


def test_initial_state_round_trips_atomically(ladder_run: TrainingRunConfig) -> None:
    path = _state_path(ladder_run)

    initial = load_fairy_ladder_state(path, ladder_run)
    write_fairy_ladder_state(path, initial)
    restored = load_fairy_ladder_state(path, ladder_run)

    expected_hash = hashlib.sha256(b"pinned-fairy-stockfish-binary").hexdigest()
    assert restored == initial
    assert restored.current_elo == 500
    assert restored.highest_passed_elo is None
    assert restored.consecutive_passes == 0
    assert restored.last_checkpoint_sha256 is None
    assert restored.last_tested_elo is None
    assert restored.last_scores is None
    assert restored.evaluation_step == 0
    assert restored.protocol["engine_binary_sha256"] == expected_hash
    assert json.loads(path.read_text(encoding="utf-8"))["current_elo"] == 500
    assert list(path.parent.glob(f".{path.name}.tmp-*")) == []


def test_two_consecutive_majority_results_advance_exactly_one_rung(
    ladder_run: TrainingRunConfig,
) -> None:
    majority = StockfishEvalScores(model_wins=11, draws=2, stockfish_wins=7)

    first = _run_evaluation(ladder_run, majority, iteration=5)
    second = _run_evaluation(ladder_run, majority, iteration=10)
    restored = load_fairy_ladder_state(_state_path(ladder_run), ladder_run)

    assert first.current_elo == 500
    assert first.highest_passed_elo is None
    assert first.consecutive_passes == 1
    assert second == restored
    assert second.current_elo == 600
    assert second.highest_passed_elo == 500
    assert second.consecutive_passes == 0
    assert second.last_iteration == 10
    assert second.last_checkpoint_sha256 == _checkpoint_digest(10)
    assert second.last_tested_elo == 500
    assert second.last_scores == majority
    assert second.evaluation_step == 2
    assert not (_state_path(ladder_run).parent / "best.pth.tar").exists()
    assert not (_state_path(ladder_run).parent / "best_eval.json").exists()


def test_non_majority_result_resets_confirmation_count(ladder_run: TrainingRunConfig) -> None:
    majority = StockfishEvalScores(model_wins=11, draws=2, stockfish_wins=7)
    failure = StockfishEvalScores(model_wins=7, draws=6, stockfish_wins=7)

    first = _run_evaluation(ladder_run, majority, iteration=5)
    second = _run_evaluation(ladder_run, failure, iteration=10)

    assert first.consecutive_passes == 1
    assert second.current_elo == 500
    assert second.highest_passed_elo is None
    assert second.consecutive_passes == 0
    assert second.last_scores == failure


def test_changed_binary_hash_invalidates_persisted_protocol(ladder_run: TrainingRunConfig) -> None:
    path = _state_path(ladder_run)
    state = load_fairy_ladder_state(path, ladder_run)
    write_fairy_ladder_state(path, state)
    Path(ladder_run.ladder_path).write_bytes(b"different-fairy-stockfish-binary")

    with pytest.raises(RuntimeError, match="protocol differs"):
        load_fairy_ladder_state(path, ladder_run)


def test_changed_match_protocol_invalidates_persisted_state(ladder_run: TrainingRunConfig) -> None:
    path = _state_path(ladder_run)
    state = load_fairy_ladder_state(path, ladder_run)
    write_fairy_ladder_state(path, state)
    changed_run = replace(ladder_run, ladder_depth=11)

    with pytest.raises(RuntimeError, match="protocol differs"):
        load_fairy_ladder_state(path, changed_run)


def test_ladder_logs_named_wandb_metrics_without_benchmark_promotion(
    ladder_run: TrainingRunConfig,
) -> None:
    scores = StockfishEvalScores(model_wins=12, draws=4, stockfish_wins=4)
    game = cast(ChessGame, object())
    network = cast(LunaNetwork, object())

    with (
        patch("luna.game.stockfish_ladder.run_stockfish_eval", return_value=scores) as evaluate,
        patch("luna.game.stockfish_ladder.time.perf_counter", side_effect=(100.0, 102.5)),
        patch("luna.game.stockfish_ladder.wandb.run", object()),
        patch("luna.game.stockfish_ladder.wandb.log") as wandb_log,
    ):
        state = run_fairy_ladder_eval(
            game,
            network,
            ladder_run,
            iteration=5,
            checkpoint_sha256=_checkpoint_digest(5),
        )

    settings = evaluate.call_args.kwargs["settings"]
    assert settings.opponent_name == "Fairy-Stockfish"
    assert settings.elo == 500
    assert evaluate.call_args.kwargs["metric_prefix"] is None
    metrics = wandb_log.call_args.args[0]
    assert metrics["iteration"] == 5
    assert metrics["ladder/evaluation_step"] == 1
    assert metrics["ladder/tested_elo"] == 500
    assert metrics["ladder/current_elo"] == 500
    assert metrics["ladder/opponent_elo"] == 500
    assert metrics["ladder/luna_wins"] == 12
    assert metrics["ladder/draws"] == 4
    assert metrics["ladder/stockfish_wins"] == 4
    assert metrics["ladder/games"] == 20
    assert metrics["ladder/win_rate"] == 0.6
    assert metrics["ladder/decisive_win_rate"] == 0.75
    assert metrics["ladder/score"] == 0.7
    assert metrics["ladder/score_approx_ci95_low"] < 0.7
    assert metrics["ladder/score_approx_ci95_high"] > 0.7
    assert metrics["ladder/duration_seconds"] == 2.5
    assert metrics["ladder/passed"] == 1
    assert metrics["ladder/advanced"] == 0
    assert metrics["ladder/consecutive_passes"] == 1
    assert metrics["ladder/has_passed_rung"] == 0
    assert "ladder/highest_passed_elo" not in metrics
    assert metrics["ladder/completed"] == 0
    assert all(not key.startswith("benchmark/") for key in metrics)
    assert state.consecutive_passes == 1
    assert not (_state_path(ladder_run).parent / "best.pth.tar").exists()
    assert not (_state_path(ladder_run).parent / "best_eval.json").exists()


def test_duplicate_checkpoint_evaluation_is_idempotent(ladder_run: TrainingRunConfig) -> None:
    scores = StockfishEvalScores(model_wins=11, draws=2, stockfish_wins=7)
    game = cast(ChessGame, object())
    network = cast(LunaNetwork, object())
    digest = _checkpoint_digest(5)

    with (
        patch("luna.game.stockfish_ladder.run_stockfish_eval", return_value=scores) as evaluate,
        patch("luna.game.stockfish_ladder.wandb.run", None),
    ):
        first = run_fairy_ladder_eval(
            game,
            network,
            ladder_run,
            iteration=5,
            checkpoint_sha256=digest,
        )
        duplicate = run_fairy_ladder_eval(
            game,
            network,
            ladder_run,
            iteration=5,
            checkpoint_sha256=digest,
            state_required=True,
        )

    assert evaluate.call_count == 1
    assert duplicate == first
    assert duplicate.consecutive_passes == 1
    assert duplicate.evaluation_step == 1
    assert duplicate.last_checkpoint_sha256 == digest
    assert duplicate.last_tested_elo == 500


def test_same_iteration_with_different_checkpoint_fails_closed(ladder_run: TrainingRunConfig) -> None:
    scores = StockfishEvalScores(model_wins=7, draws=6, stockfish_wins=7)
    _run_evaluation(ladder_run, scores, iteration=5)
    game = cast(ChessGame, object())
    network = cast(LunaNetwork, object())

    with (
        patch("luna.game.stockfish_ladder.run_stockfish_eval") as evaluate,
        patch("luna.game.stockfish_ladder.wandb.run", None),
        pytest.raises(RuntimeError, match="checkpoint changed"),
    ):
        run_fairy_ladder_eval(
            game,
            network,
            ladder_run,
            iteration=5,
            checkpoint_sha256=hashlib.sha256(b"replacement").hexdigest(),
            state_required=True,
        )

    evaluate.assert_not_called()


def test_missing_required_state_fails_closed(ladder_run: TrainingRunConfig) -> None:
    path = _state_path(ladder_run)

    initial = load_fairy_ladder_state(path, ladder_run)

    assert initial.evaluation_step == 0
    with pytest.raises(RuntimeError, match="Required Fairy ladder state is missing"):
        load_fairy_ladder_state(path, ladder_run, required=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("current_elo", 600, "cannot leave its first rung"),
        ("last_checkpoint_sha256", None, "complete last-evaluation key"),
        ("consecutive_passes", 1, "pass confirmations are inconsistent"),
        ("completed", True, "cannot advance or complete"),
        ("evaluation_step", 0, "cannot contain a last evaluation"),
    ],
)
def test_impossible_state_combinations_fail_closed(
    ladder_run: TrainingRunConfig,
    field: str,
    value: object,
    message: str,
) -> None:
    scores = StockfishEvalScores(model_wins=7, draws=6, stockfish_wins=7)
    _run_evaluation(ladder_run, scores, iteration=5)
    path = _state_path(ladder_run)
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        load_fairy_ladder_state(path, ladder_run, required=True)


def test_completed_ladder_never_evaluates_another_checkpoint(ladder_run: TrainingRunConfig) -> None:
    completed_run = replace(ladder_run, ladder_max_elo=500)
    scores = StockfishEvalScores(model_wins=11, draws=2, stockfish_wins=7)
    _run_evaluation(completed_run, scores, iteration=5)
    completed = _run_evaluation(completed_run, scores, iteration=10)
    game = cast(ChessGame, object())
    network = cast(LunaNetwork, object())

    with (
        patch("luna.game.stockfish_ladder.run_stockfish_eval") as evaluate,
        patch("luna.game.stockfish_ladder.wandb.run", None),
    ):
        unchanged = run_fairy_ladder_eval(
            game,
            network,
            completed_run,
            iteration=15,
            checkpoint_sha256=_checkpoint_digest(15),
            state_required=True,
        )

    evaluate.assert_not_called()
    assert unchanged == completed
    assert completed.completed
    assert completed.current_elo == 500
    assert completed.highest_passed_elo == 500
    assert completed.evaluation_step == 2


def test_skipped_match_does_not_change_persisted_state(ladder_run: TrainingRunConfig) -> None:
    path = _state_path(ladder_run)
    initial = load_fairy_ladder_state(path, ladder_run)
    write_fairy_ladder_state(path, initial)
    before = path.read_bytes()
    game = cast(ChessGame, object())
    network = cast(LunaNetwork, object())
    skipped = StockfishEvalSkipped(reason="runtime_error", message="engine exited")

    with (
        patch("luna.game.stockfish_ladder.run_stockfish_eval", return_value=skipped),
        patch("luna.game.stockfish_ladder.wandb.run", None),
        pytest.raises(RuntimeError, match="did not complete"),
    ):
        run_fairy_ladder_eval(
            game,
            network,
            ladder_run,
            iteration=5,
            checkpoint_sha256=_checkpoint_digest(5),
            state_required=True,
        )

    assert path.read_bytes() == before
    assert load_fairy_ladder_state(path, ladder_run, required=True) == initial


def test_wandb_failure_cannot_roll_back_ladder_progress(ladder_run: TrainingRunConfig) -> None:
    scores = StockfishEvalScores(model_wins=11, draws=2, stockfish_wins=7)
    game = cast(ChessGame, object())
    network = cast(LunaNetwork, object())
    digest = _checkpoint_digest(5)

    with (
        patch("luna.game.stockfish_ladder.run_stockfish_eval", return_value=scores),
        patch("luna.game.stockfish_ladder.wandb.run", object()),
        patch("luna.game.stockfish_ladder.wandb.log", side_effect=RuntimeError("offline")),
        pytest.raises(RuntimeError, match="offline"),
    ):
        run_fairy_ladder_eval(
            game,
            network,
            ladder_run,
            iteration=5,
            checkpoint_sha256=digest,
        )

    persisted = load_fairy_ladder_state(_state_path(ladder_run), ladder_run, required=True)
    assert persisted.consecutive_passes == 1
    assert persisted.evaluation_step == 1

    with (
        patch("luna.game.stockfish_ladder.run_stockfish_eval") as evaluate,
        patch("luna.game.stockfish_ladder.wandb.run", None),
    ):
        retried = run_fairy_ladder_eval(
            game,
            network,
            ladder_run,
            iteration=5,
            checkpoint_sha256=digest,
            state_required=True,
        )

    evaluate.assert_not_called()
    assert retried == persisted


def test_atomic_state_write_fsyncs_file_and_directory(ladder_run: TrainingRunConfig) -> None:
    path = _state_path(ladder_run)
    state = load_fairy_ladder_state(path, ladder_run)

    with patch("luna.game.stockfish_ladder.os.fsync", wraps=os.fsync) as fsync:
        write_fairy_ladder_state(path, state)

    assert fsync.call_count == 2
