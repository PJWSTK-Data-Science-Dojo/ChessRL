"""Tests for persistent adaptive Fairy-Stockfish ladder state."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from luna.config import TrainingRunConfig
from luna.game.chess_game import ChessGame
from luna.game.stockfish_eval import StockfishEvalScores
from luna.game.stockfish_ladder import (
    LADDER_STATE_NAME,
    FairyLadderState,
    load_fairy_ladder_state,
    run_fairy_ladder_eval,
    write_fairy_ladder_state,
)
from luna.game.stockfish_ladder_state import fairy_ladder_protocol
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


def test_self_play_search_contempt_is_absent_from_ladder_protocol(ladder_run: TrainingRunConfig) -> None:
    run = replace(ladder_run, search_contempt_visit_limit=4)

    protocol = fairy_ladder_protocol(run)

    assert "search_contempt_visit_limit" not in cast(dict[str, object], protocol["mcts"])


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
