"""Direct, reproducible checkpoint-versus-checkpoint arena evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import chess

from luna.config import MCTSParams, validate_mcts_params
from luna.game.arena import Arena
from luna.game.chess_game import ChessGame
from luna.game.opening_suite import MAX_OPENING_PAIRS, OPENING_SUITE_VERSION, evaluation_openings
from luna.mcts import MCTS
from luna.network import LunaNetwork

CHECKPOINT_ARENA_SCHEMA_VERSION = 1
_WIN_THRESHOLD = 0.5
_MIN_GAMES = 2
_MAX_GAMES = MAX_OPENING_PAIRS * 2

ArenaPlayer = Callable[[chess.Board], int]


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """Path and content identity of one evaluated checkpoint."""

    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointArenaProtocol:
    """Settings that must match for comparable head-to-head results."""

    schema_version: int
    opening_suite_version: int
    games: int
    max_ply: int
    minimum_checkpoint_b_score: float
    mcts: MCTSParams


@dataclass(frozen=True, slots=True)
class CheckpointArenaPlayers:
    checkpoint_a: ArenaPlayer
    checkpoint_b: ArenaPlayer


@dataclass(frozen=True, slots=True)
class _MatchContext:
    game: ChessGame
    players: CheckpointArenaPlayers
    protocol: CheckpointArenaProtocol


@dataclass(frozen=True, slots=True)
class CheckpointArenaScores:
    checkpoint_a_wins: int
    draws: int
    checkpoint_b_wins: int

    @property
    def games(self) -> int:
        return self.checkpoint_a_wins + self.draws + self.checkpoint_b_wins

    @property
    def checkpoint_b_score(self) -> float:
        if self.games == 0:
            raise ValueError("Cannot score an empty arena result")
        return (self.checkpoint_b_wins + 0.5 * self.draws) / self.games


@dataclass(frozen=True, slots=True)
class CheckpointArenaResult:
    checkpoint_a: CheckpointIdentity
    checkpoint_b: CheckpointIdentity
    protocol: CheckpointArenaProtocol
    scores: CheckpointArenaScores

    @property
    def checkpoint_b_passed(self) -> bool:
        return self.scores.checkpoint_b_score >= self.protocol.minimum_checkpoint_b_score


class ArenaMCTSPlayer:
    """Greedy latent-MCTS player used by external and checkpoint arenas."""

    def __init__(self, game: ChessGame, nnet: LunaNetwork, mcts_params: MCTSParams) -> None:
        self._mcts = MCTS(game, nnet, mcts_params)

    def __call__(self, canonical_board: chess.Board) -> int:
        self._mcts.search_latent(canonical_board, temp=0.0, add_exploration_noise=False)
        if self._mcts.last_action is None:
            raise RuntimeError("Search returned no legal continuation")
        return self._mcts.last_action


@dataclass(slots=True)
class _ScoreTally:
    checkpoint_a_wins: int = 0
    draws: int = 0
    checkpoint_b_wins: int = 0

    def record(self, checkpoint_a_result: float) -> None:
        winner = _winner(checkpoint_a_result)
        if winner == 0:
            self.draws += 1
        elif winner == 1:
            self.checkpoint_a_wins += 1
        else:
            self.checkpoint_b_wins += 1

    def scores(self) -> CheckpointArenaScores:
        return CheckpointArenaScores(self.checkpoint_a_wins, self.draws, self.checkpoint_b_wins)


def checkpoint_identity(path: Path) -> CheckpointIdentity:
    """Resolve and hash an immutable evaluation input."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return CheckpointIdentity(str(resolved), digest.hexdigest())


def validate_checkpoint_arena_protocol(protocol: CheckpointArenaProtocol) -> CheckpointArenaProtocol:
    """Reject incomplete or incomparable arena settings before loading models."""
    if protocol.schema_version != CHECKPOINT_ARENA_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CHECKPOINT_ARENA_SCHEMA_VERSION}")
    if protocol.opening_suite_version != OPENING_SUITE_VERSION:
        raise ValueError(f"opening_suite_version must be {OPENING_SUITE_VERSION}")
    if isinstance(protocol.games, bool) or not _MIN_GAMES <= protocol.games <= _MAX_GAMES or protocol.games % 2:
        raise ValueError(f"games must be an even integer from {_MIN_GAMES} through {_MAX_GAMES}")
    if isinstance(protocol.max_ply, bool) or not isinstance(protocol.max_ply, int) or protocol.max_ply < 1:
        raise ValueError("max_ply must be a positive integer")
    if not 0.0 <= protocol.minimum_checkpoint_b_score <= 1.0:
        raise ValueError("minimum_checkpoint_b_score must be between 0 and 1")
    validate_mcts_params(protocol.mcts)
    return protocol


def run_checkpoint_arena(
    game: ChessGame,
    players: CheckpointArenaPlayers,
    protocol: CheckpointArenaProtocol,
) -> CheckpointArenaScores:
    """Play every opening with both checkpoint/color assignments."""
    validate_checkpoint_arena_protocol(protocol)
    tally = _ScoreTally()
    context = _MatchContext(game, players, protocol)
    for opening in evaluation_openings(protocol.games // 2):
        _play_opening_pair(context, opening, tally)
    return tally.scores()


def checkpoint_arena_payload(result: CheckpointArenaResult) -> dict[str, object]:
    """Return stable JSON-compatible result data."""
    payload: dict[str, object] = asdict(result)
    payload["checkpoint_b_score"] = result.scores.checkpoint_b_score
    payload["checkpoint_b_passed"] = result.checkpoint_b_passed
    return payload


def write_checkpoint_arena_result(path: Path, result: CheckpointArenaResult) -> Path:
    """Atomically persist one complete arena record."""
    _validate_result(result)
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
    content = json.dumps(checkpoint_arena_payload(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, resolved)
        _sync_directory(resolved.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved


def _play_opening_pair(
    context: _MatchContext,
    opening: chess.Board,
    tally: _ScoreTally,
) -> None:
    result = Arena(
        context.players.checkpoint_a,
        context.players.checkpoint_b,
        context.game,
    ).play_game(
        max_ply=context.protocol.max_ply,
        initial_board=opening,
    )
    tally.record(result)
    result = Arena(
        context.players.checkpoint_b,
        context.players.checkpoint_a,
        context.game,
    ).play_game(
        max_ply=context.protocol.max_ply,
        initial_board=opening,
    )
    tally.record(-result)


def _winner(result: float) -> int:
    if result > _WIN_THRESHOLD:
        return 1
    if result < -_WIN_THRESHOLD:
        return -1
    return 0


def _validate_result(result: CheckpointArenaResult) -> None:
    validate_checkpoint_arena_protocol(result.protocol)
    scores = result.scores
    values = (scores.checkpoint_a_wins, scores.draws, scores.checkpoint_b_wins)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("arena scores must be non-negative integers")
    if scores.games != result.protocol.games:
        raise ValueError("arena score count does not match the protocol")
    _validate_identity(result.checkpoint_a)
    _validate_identity(result.checkpoint_b)


def _validate_identity(identity: CheckpointIdentity) -> None:
    if not identity.path.strip():
        raise ValueError("checkpoint path cannot be blank")
    if len(identity.sha256) != 64 or any(character not in "0123456789abcdef" for character in identity.sha256):
        raise ValueError("checkpoint SHA-256 must be lowercase hexadecimal")


def _sync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
