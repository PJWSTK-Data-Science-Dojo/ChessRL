from __future__ import annotations

import secrets
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import chess
from flask import current_app, request, session

from web_engine import ColorName, EngineBusyError, EngineDecision, GameMode, LunaEngineService

API_PREFIX = "/api/v1"
GAME_TTL_SECONDS = 12 * 60 * 60
MAX_GAMES = 512
MAX_GAMES_PER_SESSION = 8


@dataclass
class GameRecord:
    game_id: str
    owner_id: str
    mode: GameMode
    human_color: ColorName | None
    strength: str
    board: chess.Board = field(default_factory=chess.Board)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_engine_move: str | None = None
    last_think_time_ms: int | None = None
    last_simulations: int | None = None
    last_evaluation_white: float | None = None
    last_confidence: float | None = None
    revision: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details
        self.headers = dict(headers or {})


class GameRegistry:
    def __init__(
        self,
        *,
        max_games: int = MAX_GAMES,
        max_games_per_session: int = MAX_GAMES_PER_SESSION,
        ttl_seconds: int = GAME_TTL_SECONDS,
    ) -> None:
        if max_games < 1 or max_games_per_session < 1 or ttl_seconds < 1:
            raise ValueError("Game registry limits must be positive")
        self._games: dict[str, GameRecord] = {}
        self._lock = threading.RLock()
        self.max_games = max_games
        self.max_games_per_session = max_games_per_session
        self.ttl_seconds = ttl_seconds

    def _prune_expired(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        expired = [game_id for game_id, game in self._games.items() if game.updated_at < cutoff]
        for game_id in expired:
            self._games.pop(game_id, None)

    def create(
        self,
        *,
        owner_id: str,
        mode: GameMode,
        human_color: ColorName | None,
        strength: str,
    ) -> GameRecord:
        with self._lock:
            self._prune_expired()
            self._evict_owned_games(owner_id)
            self._evict_oldest_games()
            record = GameRecord(
                game_id=str(uuid.uuid4()),
                owner_id=owner_id,
                mode=mode,
                human_color=human_color,
                strength=strength,
            )
            self._games[record.game_id] = record
            return record

    def _evict_owned_games(self, owner_id: str) -> None:
        owned = sorted(
            (game for game in self._games.values() if secrets.compare_digest(game.owner_id, owner_id)),
            key=lambda item: item.updated_at,
        )
        while len(owned) >= self.max_games_per_session:
            self._games.pop(owned.pop(0).game_id, None)

    def _evict_oldest_games(self) -> None:
        while len(self._games) >= self.max_games:
            oldest = min(self._games.values(), key=lambda item: item.updated_at)
            self._games.pop(oldest.game_id, None)

    def get(self, game_id: str, owner_id: str) -> GameRecord:
        try:
            normalized = str(uuid.UUID(game_id))
        except ValueError as exc:
            raise ApiError(404, "game_not_found", "That game does not exist.") from exc
        with self._lock:
            record = self._games.get(normalized)
            if record is None or not secrets.compare_digest(record.owner_id, owner_id):
                raise ApiError(404, "game_not_found", "That game does not exist.")
            if record.updated_at < time.time() - self.ttl_seconds:
                self._games.pop(normalized, None)
                raise ApiError(410, "game_expired", "That game has expired. Start a new mission.")
            record.updated_at = time.time()
            return record

    def delete(self, game_id: str, owner_id: str) -> None:
        record = self.get(game_id, owner_id)
        with self._lock:
            self._games.pop(record.game_id, None)


@contextmanager
def locked_game(record: GameRecord) -> Iterator[None]:
    if not record.lock.acquire(blocking=False):
        raise ApiError(
            429,
            "game_busy",
            "Another request is already changing this game. Please retry shortly.",
            headers={"Retry-After": "2"},
        )
    try:
        yield
    finally:
        record.lock.release()


def color_name(color: chess.Color) -> ColorName:
    return "white" if color == chess.WHITE else "black"


def client_id() -> str:
    identifier = session.get("luna_client_id")
    if not isinstance(identifier, str) or len(identifier) != 32:
        identifier = uuid.uuid4().hex
        session["luna_client_id"] = identifier
    return identifier


def engine() -> LunaEngineService:
    active_engine = current_app.extensions.get("luna_engine")
    if not isinstance(active_engine, LunaEngineService):
        raise ApiError(503, "model_unavailable", "Luna is offline because no compatible model is loaded.")
    return active_engine


def registry() -> GameRegistry:
    active_registry = current_app.extensions.get("luna_games")
    if not isinstance(active_registry, GameRegistry):
        raise RuntimeError("Game registry is not configured")
    return active_registry


def json_body() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError(400, "invalid_json", "Send a JSON object as the request body.")
    return payload


def require_revision(payload: Mapping[str, Any], record: GameRecord) -> None:
    revision = payload.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise ApiError(400, "missing_revision", "Include the current integer game revision.")
    if revision != record.revision:
        raise ApiError(
            409,
            "stale_position",
            "The position changed before this request was processed. Refresh the game and try again.",
            {"current_revision": record.revision},
        )


def search_api_error(error: EngineBusyError | RuntimeError | ValueError) -> ApiError:
    if isinstance(error, EngineBusyError):
        return ApiError(
            429,
            "engine_busy",
            "Luna is calculating for another player. Please retry shortly.",
            headers={"Retry-After": "2"},
        )
    return ApiError(500, "search_failed", "Luna could not complete the search. Please try again.")


def history(board: chess.Board) -> list[dict[str, Any]]:
    replay = chess.Board()
    entries: list[dict[str, Any]] = []
    for ply, move in enumerate(board.move_stack, start=1):
        entries.append(
            {
                "ply": ply,
                "move_number": replay.fullmove_number,
                "color": color_name(replay.turn),
                "san": replay.san(move),
                "uci": move.uci(),
            }
        )
        replay.push(move)
    return entries


def captured_pieces(board: chess.Board) -> dict[str, list[str]]:
    replay = chess.Board()
    captured: dict[str, list[str]] = {"white": [], "black": []}
    for move in board.move_stack:
        if replay.is_en_passant(move):
            offset = -8 if replay.turn == chess.WHITE else 8
            captured_piece = replay.piece_at(move.to_square + offset)
        else:
            captured_piece = replay.piece_at(move.to_square)
        if captured_piece is not None:
            captured[color_name(replay.turn)].append(captured_piece.symbol())
        replay.push(move)
    return captured


def game_result(board: chess.Board) -> dict[str, str] | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    headline = "Draw" if outcome.winner is None else f"{color_name(outcome.winner).title()} wins"
    reason = outcome.termination.name.replace("_", " ").lower()
    return {"headline": headline, "reason": reason, "notation": board.result(claim_draw=True)}


def can_undo(record: GameRecord, entries: list[dict[str, Any]]) -> bool:
    return bool(
        record.mode == "human"
        and record.human_color is not None
        and any(entry["color"] == record.human_color for entry in entries)
    )


def game_payload(record: GameRecord, active_engine: LunaEngineService) -> dict[str, Any]:
    entries = history(record.board)
    result = game_result(record.board)
    profile = active_engine.strengths[record.strength]
    return {
        "id": record.game_id,
        "revision": record.revision,
        "mode": record.mode,
        "human_color": record.human_color,
        "strength": {"id": record.strength, "name": profile.name, "simulations": profile.simulations},
        "fen": record.board.fen(),
        "turn": color_name(record.board.turn),
        "legal_moves": [move.uci() for move in record.board.legal_moves],
        "last_move": entries[-1]["uci"] if entries else None,
        "history": entries,
        "captured": captured_pieces(record.board),
        "is_check": record.board.is_check(),
        "is_game_over": result is not None,
        "result": result,
        "status": _status(record, result),
        "can_undo": can_undo(record, entries),
        "can_hint": record.mode == "human" and result is None and color_name(record.board.turn) == record.human_color,
        "engine": _engine_metrics(record),
    }


def _status(record: GameRecord, result: dict[str, str] | None) -> str:
    if result is not None:
        return result["headline"]
    if record.mode == "selfplay":
        return f"{color_name(record.board.turn).title()} to calculate"
    if color_name(record.board.turn) == record.human_color:
        return "Your turn"
    return "Luna to move"


def _engine_metrics(record: GameRecord) -> dict[str, str | int | float | None]:
    return {
        "last_move": record.last_engine_move,
        "think_time_ms": record.last_think_time_ms,
        "simulations": record.last_simulations,
        "evaluation_white": record.last_evaluation_white,
        "confidence": record.last_confidence,
    }


def apply_engine_move(record: GameRecord, active_engine: LunaEngineService) -> EngineDecision:
    decision = active_engine.analyze(record.board, record.strength)
    if decision.move not in record.board.legal_moves:
        raise RuntimeError("Engine produced an illegal move")
    record.board.push(decision.move)
    record.revision += 1
    record.last_engine_move = decision.move.uci()
    record.last_think_time_ms = decision.think_time_ms
    record.last_simulations = decision.simulations
    record.last_evaluation_white = decision.evaluation_white
    record.last_confidence = decision.confidence
    record.updated_at = time.time()
    return decision
