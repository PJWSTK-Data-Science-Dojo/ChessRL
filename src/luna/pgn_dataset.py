"""Streaming expert-PGN ingestion for supervised MuZero warm starts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TextIO, cast

import chess
import chess.pgn
import numpy as np
import zstandard

from luna.game.chess_game import ChessGame, mirror_move, move_to_action, player_from_turn
from luna.replay_buffer import Trajectory

_COMPLETED_RESULTS: dict[str, float] = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
_STANDARD_VARIANTS = frozenset({"standard", "chess"})


@dataclass(frozen=True, slots=True)
class PgnDatasetConfig:
    min_player_elo: int = 2000
    max_positions: int = 300_000
    validation_fraction: float = 0.05
    split_seed: int = 0
    min_game_plies: int = 8
    max_game_plies: int = 512

    def __post_init__(self) -> None:
        if isinstance(self.min_player_elo, bool) or not isinstance(self.min_player_elo, int) or self.min_player_elo < 0:
            raise ValueError("min_player_elo must be a non-negative integer")
        if isinstance(self.max_positions, bool) or not isinstance(self.max_positions, int) or self.max_positions <= 0:
            raise ValueError("max_positions must be a positive integer")
        if (
            isinstance(self.validation_fraction, bool)
            or not isinstance(self.validation_fraction, int | float)
            or not math.isfinite(self.validation_fraction)
            or not 0.0 <= self.validation_fraction <= 1.0
        ):
            raise ValueError("validation_fraction must be finite and between zero and one")
        if isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int):
            raise ValueError("split_seed must be an integer")
        if (
            isinstance(self.min_game_plies, bool)
            or not isinstance(self.min_game_plies, int)
            or self.min_game_plies <= 0
        ):
            raise ValueError("min_game_plies must be a positive integer")
        if (
            isinstance(self.max_game_plies, bool)
            or not isinstance(self.max_game_plies, int)
            or self.max_game_plies < self.min_game_plies
        ):
            raise ValueError("max_game_plies must be at least min_game_plies")


@dataclass(frozen=True, slots=True)
class PgnDatasetStats:
    games_scanned: int
    games_loaded: int
    games_filtered: int
    duplicate_games: int
    capacity_skipped_games: int
    train_games: int
    validation_games: int
    train_positions: int
    validation_positions: int
    engine_evaluated_positions: int
    result_fallback_positions: int
    limit_reached: bool


@dataclass(frozen=True, slots=True)
class PgnDataset:
    train_trajectories: tuple[Trajectory, ...]
    validation_trajectories: tuple[Trajectory, ...]
    stats: PgnDatasetStats


@dataclass(frozen=True, slots=True)
class _ConvertedGame:
    trajectory: Trajectory
    engine_evaluated_positions: int
    result_fallback_positions: int


@dataclass(frozen=True, slots=True)
class _PositionContext:
    node: chess.pgn.GameNode
    board: chess.Board
    move: chess.Move
    result: float


@dataclass(frozen=True, slots=True)
class _PositionTargets:
    observation: np.ndarray
    action: int
    policy: np.ndarray
    value: float
    valid_moves: np.ndarray
    from_engine: bool


@dataclass(slots=True)
class _StatsAccumulator:
    games_scanned: int = 0
    games_loaded: int = 0
    games_filtered: int = 0
    duplicate_games: int = 0
    capacity_skipped_games: int = 0
    train_games: int = 0
    validation_games: int = 0
    train_positions: int = 0
    validation_positions: int = 0
    engine_evaluated_positions: int = 0
    result_fallback_positions: int = 0
    limit_reached: bool = False

    def freeze(self) -> PgnDatasetStats:
        return PgnDatasetStats(
            self.games_scanned,
            self.games_loaded,
            self.games_filtered,
            self.duplicate_games,
            self.capacity_skipped_games,
            self.train_games,
            self.validation_games,
            self.train_positions,
            self.validation_positions,
            self.engine_evaluated_positions,
            self.result_fallback_positions,
            self.limit_reached,
        )


@dataclass(slots=True)
class _LoadState:
    config: PgnDatasetConfig
    train: list[Trajectory] = field(default_factory=list)
    validation: list[Trajectory] = field(default_factory=list)
    stats: _StatsAccumulator = field(default_factory=_StatsAccumulator)
    fingerprints: set[bytes] = field(default_factory=set)

    @property
    def position_count(self) -> int:
        return self.stats.train_positions + self.stats.validation_positions

    def to_dataset(self) -> PgnDataset:
        return PgnDataset(tuple(self.train), tuple(self.validation), self.stats.freeze())


class _Eligibility(StrEnum):
    ACCEPTED = "accepted"
    PARSE_ERROR = "parse_error"
    NON_STANDARD = "non_standard"
    CUSTOM_START = "custom_start"
    INCOMPLETE = "incomplete"
    NON_EXPERT = "non_expert"
    BOT = "bot"
    EMPTY = "empty"
    ABNORMAL_LENGTH = "abnormal_length"
    ABNORMAL_TERMINATION = "abnormal_termination"
    INCONSISTENT_RESULT = "inconsistent_result"


class _LoadDecision(StrEnum):
    CONTINUE = "continue"
    STOP = "stop"


def load_pgn_dataset(path: Path, config: PgnDatasetConfig, game: ChessGame) -> PgnDataset:
    """Load complete expert games without exceeding the configured position budget."""
    state = _LoadState(config)
    with _open_pgn(path) as stream:
        for record in _read_games(stream):
            if _load_record(record, game, state) is _LoadDecision.STOP:
                break
    return state.to_dataset()


def _load_record(record: chess.pgn.Game, game: ChessGame, state: _LoadState) -> _LoadDecision:
    state.stats.games_scanned += 1
    if _eligibility(record, state.config) is not _Eligibility.ACCEPTED:
        state.stats.games_filtered += 1
        return _LoadDecision.CONTINUE
    fingerprint = _game_fingerprint(record)
    if fingerprint in state.fingerprints:
        state.stats.duplicate_games += 1
        return _LoadDecision.CONTINUE
    state.fingerprints.add(fingerprint)
    converted = _convert_game(record, game)
    if state.position_count + converted.trajectory.game_length > state.config.max_positions:
        state.stats.capacity_skipped_games += 1
        state.stats.limit_reached = True
        return _LoadDecision.STOP
    _store_game(converted, fingerprint, state)
    return _LoadDecision.CONTINUE


@contextmanager
def _open_pgn(path: Path) -> Iterator[TextIO]:
    name = path.name.casefold()
    if name.endswith(".pgn.zst"):
        stream = cast(TextIO, zstandard.open(path, mode="rt", encoding="utf-8"))
    elif name.endswith(".pgn"):
        stream = path.open(mode="rt", encoding="utf-8")
    else:
        raise ValueError(f"PGN path must end in .pgn or .pgn.zst: {path}")
    try:
        yield stream
    finally:
        stream.close()


def _read_games(stream: TextIO) -> Iterator[chess.pgn.Game]:
    while record := chess.pgn.read_game(stream):
        yield record


def _eligibility(record: chess.pgn.Game, config: PgnDatasetConfig) -> _Eligibility:
    if record.errors:
        return _Eligibility.PARSE_ERROR
    header_result = _header_eligibility(record, config)
    if header_result is not _Eligibility.ACCEPTED:
        return header_result
    if _has_bot_player(record.headers):
        return _Eligibility.BOT
    if not record.variations:
        return _Eligibility.EMPTY
    if not config.min_game_plies <= record.end().ply() <= config.max_game_plies:
        return _Eligibility.ABNORMAL_LENGTH
    if record.headers.get("Termination", "Normal").strip().casefold() != "normal":
        return _Eligibility.ABNORMAL_TERMINATION
    if not _result_is_consistent(record, record.headers["Result"]):
        return _Eligibility.INCONSISTENT_RESULT
    return _Eligibility.ACCEPTED


def _header_eligibility(record: chess.pgn.Game, config: PgnDatasetConfig) -> _Eligibility:
    if record.headers.get("Variant", "Standard").strip().casefold() not in _STANDARD_VARIANTS:
        return _Eligibility.NON_STANDARD
    if record.headers.get("SetUp", "0") == "1" or "FEN" in record.headers:
        return _Eligibility.CUSTOM_START
    if record.headers.get("Result", "*") not in _COMPLETED_RESULTS:
        return _Eligibility.INCOMPLETE
    if not _has_expert_players(record.headers, config.min_player_elo):
        return _Eligibility.NON_EXPERT
    return _Eligibility.ACCEPTED


def _has_expert_players(headers: chess.pgn.Headers, minimum: int) -> bool:
    try:
        ratings = (int(headers["WhiteElo"]), int(headers["BlackElo"]))
    except (KeyError, ValueError):
        return False
    return all(rating >= minimum for rating in ratings)


def _has_bot_player(headers: chess.pgn.Headers) -> bool:
    titles = (headers.get("WhiteTitle", ""), headers.get("BlackTitle", ""))
    return any(title.strip().casefold() == "bot" for title in titles)


def _result_is_consistent(record: chess.pgn.Game, result: str) -> bool:
    outcome = record.end().board().outcome(claim_draw=False)
    return outcome is None or outcome.result() == result


def _game_fingerprint(record: chess.pgn.Game) -> bytes:
    digest = hashlib.sha256(record.headers["Result"].encode("ascii"))
    for move in record.mainline_moves():
        digest.update(move.uci().encode("ascii"))
        digest.update(b"\0")
    return digest.digest()


def _convert_game(record: chess.pgn.Game, game: ChessGame) -> _ConvertedGame:
    board = record.board()
    result = _COMPLETED_RESULTS[record.headers["Result"]]
    fields = _TrajectoryFields()
    current_node: chess.pgn.GameNode = record
    for child in record.mainline():
        move = child.move
        if move is None:
            raise RuntimeError("A PGN mainline child is missing its move")
        context = _PositionContext(current_node, board, move, result)
        fields.append(_position_targets(context, game))
        board.push(move)
        current_node = child
    fields.rewards[-1] = -result * player_from_turn(board.turn)
    outcome = board.outcome(claim_draw=False)
    trajectory = fields.to_trajectory(outcome.termination if outcome is not None else None)
    return _ConvertedGame(trajectory, fields.engine_evaluated_positions, fields.result_fallback_positions)


@dataclass(slots=True)
class _TrajectoryFields:
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    root_policies: list[np.ndarray] = field(default_factory=list)
    root_values: list[float] = field(default_factory=list)
    valids: list[np.ndarray] = field(default_factory=list)
    engine_evaluated_positions: int = 0
    result_fallback_positions: int = 0

    def append(self, targets: _PositionTargets) -> None:
        self.observations.append(targets.observation)
        self.actions.append(targets.action)
        self.rewards.append(0.0)
        self.root_policies.append(targets.policy)
        self.root_values.append(targets.value)
        self.valids.append(targets.valid_moves)
        self.engine_evaluated_positions += int(targets.from_engine)
        self.result_fallback_positions += int(not targets.from_engine)

    def to_trajectory(self, termination: chess.Termination | None) -> Trajectory:
        return Trajectory(
            self.observations,
            self.actions,
            self.rewards,
            self.root_policies,
            self.root_values,
            self.valids,
            termination=termination,
        )


def _position_targets(context: _PositionContext, game: ChessGame) -> _PositionTargets:
    player = player_from_turn(context.board.turn)
    canonical_board = game.get_canonical_form(context.board, player)
    action = move_to_action(context.move if player == 1 else mirror_move(context.move))
    valid_moves = game.get_valid_moves(canonical_board, 1).astype(np.bool_)
    policy = np.zeros(game.get_action_size(), dtype=np.float16)
    policy[action] = 1.0
    value, from_engine = _position_value(context.node, context.board, context.result)
    observation = game.to_array(canonical_board).astype(np.float16)
    return _PositionTargets(observation, action, policy, value, valid_moves, from_engine)


def _position_value(node: chess.pgn.GameNode, board: chess.Board, result: float) -> tuple[float, bool]:
    score = node.eval()
    if score is None:
        return result * player_from_turn(board.turn), False
    expectation = score.pov(board.turn).wdl(model="sf16", ply=board.ply()).expectation()
    return float(np.clip(2.0 * expectation - 1.0, -1.0, 1.0)), True


def _store_game(converted: _ConvertedGame, fingerprint: bytes, state: _LoadState) -> None:
    if _is_validation_game(fingerprint, state.config):
        state.validation.append(converted.trajectory)
        state.stats.validation_games += 1
        state.stats.validation_positions += converted.trajectory.game_length
    else:
        state.train.append(converted.trajectory)
        state.stats.train_games += 1
        state.stats.train_positions += converted.trajectory.game_length
    state.stats.games_loaded += 1
    state.stats.engine_evaluated_positions += converted.engine_evaluated_positions
    state.stats.result_fallback_positions += converted.result_fallback_positions


def _is_validation_game(fingerprint: bytes, config: PgnDatasetConfig) -> bool:
    digest = hashlib.sha256(str(config.split_seed).encode("ascii") + fingerprint).digest()
    unit_interval = int.from_bytes(digest[:8], byteorder="big") / 2**64
    return unit_interval < config.validation_fraction
