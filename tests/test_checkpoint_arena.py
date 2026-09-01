"""Tests for direct paired-opening checkpoint evaluation."""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import chess
import pytest

from luna.config import MCTSParams
from luna.game.arena import Arena
from luna.game.checkpoint_arena import (
    CHECKPOINT_ARENA_SCHEMA_VERSION,
    CheckpointArenaPlayers,
    CheckpointArenaProtocol,
    CheckpointArenaResult,
    CheckpointArenaScores,
    CheckpointIdentity,
    checkpoint_arena_payload,
    checkpoint_identity,
    run_checkpoint_arena,
    validate_checkpoint_arena_protocol,
    write_checkpoint_arena_result,
)
from luna.game.chess_game import ChessGame
from luna.game.opening_suite import OPENING_SUITE_VERSION, evaluation_openings


def _protocol(*, games: int = 4) -> CheckpointArenaProtocol:
    return CheckpointArenaProtocol(
        schema_version=CHECKPOINT_ARENA_SCHEMA_VERSION,
        opening_suite_version=OPENING_SUITE_VERSION,
        games=games,
        max_ply=256,
        minimum_checkpoint_b_score=0.5,
        mcts=MCTSParams(
            num_mcts_sims=32,
            search_mode="gumbel",
            gumbel_max_considered_actions=8,
            dir_noise=False,
        ),
    )


def _player(_board: chess.Board) -> int:
    raise AssertionError("Patched Arena must not request a move")


def _result(scores: CheckpointArenaScores) -> CheckpointArenaResult:
    return CheckpointArenaResult(
        checkpoint_a=CheckpointIdentity("/checkpoints/a", "a" * 64),
        checkpoint_b=CheckpointIdentity("/checkpoints/b", "b" * 64),
        protocol=_protocol(),
        scores=scores,
    )


def test_opening_suite_returns_distinct_six_ply_positions() -> None:
    openings = evaluation_openings(3)

    assert len(openings) == 3
    assert len({board.fen() for board in openings}) == 3
    assert all(len(board.move_stack) == 6 for board in openings)
    assert all(board.turn == chess.WHITE for board in openings)


def test_arena_plays_each_opening_with_both_color_assignments() -> None:
    calls: list[tuple[str, bool, bool, int | None]] = []
    results: Iterator[float] = iter((1.0, 1.0, 0.0, -1.0))
    players = CheckpointArenaPlayers(_player, lambda board: _player(board))

    def record_game(
        arena: Arena,
        verbose: bool = False,
        max_ply: int | None = None,
        initial_board: chess.Board | None = None,
    ) -> float:
        del verbose
        assert initial_board is not None
        calls.append(
            (initial_board.fen(), arena.player1 is players.checkpoint_a, arena.player2 is players.checkpoint_a, max_ply)
        )
        return next(results)

    with patch.object(Arena, "play_game", new=record_game):
        scores = run_checkpoint_arena(ChessGame(), players, _protocol())

    assert scores == CheckpointArenaScores(checkpoint_a_wins=2, draws=1, checkpoint_b_wins=1)
    assert calls[0][0] == calls[1][0]
    assert calls[2][0] == calls[3][0]
    assert calls[0][0] != calls[2][0]
    assert [call[1:3] for call in calls] == [(True, False), (False, True), (True, False), (False, True)]
    assert all(call[3] == 256 for call in calls)


@pytest.mark.parametrize("games", [0, 1, 3, 22, True])
def test_protocol_rejects_non_even_game_budget(games: int) -> None:
    with pytest.raises(ValueError, match="games must be an even integer"):
        validate_checkpoint_arena_protocol(_protocol(games=games))


def test_checkpoint_identity_hashes_exact_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth.tar"
    checkpoint.write_bytes(b"immutable checkpoint")

    identity = checkpoint_identity(checkpoint)

    assert identity.path == str(checkpoint.resolve())
    assert identity.sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_atomic_result_contains_protocol_hashes_and_gate_decision(tmp_path: Path) -> None:
    result = _result(CheckpointArenaScores(checkpoint_a_wins=1, draws=2, checkpoint_b_wins=1))
    destination = tmp_path / "arena.json"

    written = write_checkpoint_arena_result(destination, result)
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert payload == checkpoint_arena_payload(result)
    assert payload["checkpoint_b_score"] == 0.5
    assert payload["checkpoint_b_passed"] is True
    assert payload["checkpoint_a"]["sha256"] == "a" * 64
    assert payload["protocol"]["mcts"]["num_mcts_sims"] == 32
    assert list(tmp_path.glob(".*.tmp-*")) == []


def test_candidate_must_reach_half_score() -> None:
    failed = _result(CheckpointArenaScores(checkpoint_a_wins=2, draws=1, checkpoint_b_wins=1))
    passed = _result(CheckpointArenaScores(checkpoint_a_wins=1, draws=2, checkpoint_b_wins=1))

    assert failed.checkpoint_b_passed is False
    assert passed.checkpoint_b_passed is True
