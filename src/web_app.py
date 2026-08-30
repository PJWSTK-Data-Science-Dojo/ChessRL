"""Production-oriented web interface for playing and spectating Luna Chess.

The web process owns one immutable inference network and a registry of isolated games.
Every browser session receives a private UUID namespace, while a process-wide lock
serializes accelerator inference. The API refuses to start without a compatible
checkpoint; serving moves from randomly initialized weights is never useful.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import chess
import numpy as np
import tyro
from flask import Flask, Response, current_app, jsonify, make_response, request, send_from_directory, session
from flask.typing import ResponseReturnValue
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from loguru import logger
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

API_PREFIX = "/api/v1"
GAME_TTL_SECONDS = 12 * 60 * 60
MAX_GAMES = 512
MAX_GAMES_PER_SESSION = 8
MAX_PENDING_SEARCHES = 2
SEARCH_ADMISSION_TIMEOUT_SECONDS = 1.0
SEARCH_DEADLINE_SECONDS = 75.0

ColorName = Literal["white", "black"]
GameMode = Literal["human", "selfplay"]


@dataclass(frozen=True)
class StrengthProfile:
    """A named search budget exposed by the web client."""

    name: str
    simulations: int
    description: str


@dataclass(frozen=True)
class EngineDecision:
    """One completed search result."""

    move: chess.Move
    san: str
    confidence: float
    evaluation_white: float
    think_time_ms: int
    simulations: int


@dataclass(frozen=True)
class WebAppConfig:
    """Security and proxy policy for one web process."""

    trusted_hosts: tuple[str, ...] = ()
    proxy_hops: int = 0
    hsts_max_age_seconds: int = 0
    rate_limit_storage_uri: str = "memory://"


def parse_exact_trusted_hosts(value: str) -> tuple[str, ...]:
    """Parse a required comma-separated list without wildcard trust patterns."""
    hosts = tuple(host.strip() for host in value.split(",") if host.strip())
    if not hosts:
        raise RuntimeError("LUNA_TRUSTED_HOSTS must contain at least one exact host")
    if any(host.startswith(".") or "*" in host or "://" in host or "/" in host for host in hosts):
        raise RuntimeError("LUNA_TRUSTED_HOSTS accepts exact host names only")
    return hosts


class EngineBusyError(RuntimeError):
    """Raised when the bounded inference queue has no capacity."""


class LunaEngineService:
    """Thread-safe adapter around one shared Luna inference network."""

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        device: str,
        search_simulations: int,
        compile_inference: bool,
    ) -> None:
        if search_simulations < 8:
            raise ValueError("search_simulations must be at least 8")

        from luna.game.chess_game import ChessGame
        from luna.network import LunaNetwork

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        self.game = ChessGame()
        self.network = LunaNetwork.from_checkpoint(
            self.game,
            checkpoint,
            device=device,
            compile_inference=compile_inference,
            load_optimizer=False,
        )
        self.network.nnet.eval()
        self.checkpoint_name = checkpoint.name
        self.device = device
        self._inference_lock = threading.Lock()
        self._search_slots = threading.BoundedSemaphore(MAX_PENDING_SEARCHES)
        self.strengths: dict[str, StrengthProfile] = {
            "quick": StrengthProfile(
                name="Quick scan",
                simulations=max(8, search_simulations // 2),
                description="Responsive play with a compact search.",
            ),
            "strong": StrengthProfile(
                name="Deep orbit",
                simulations=search_simulations,
                description="The recommended balance of speed and calculation.",
            ),
            "maximum": StrengthProfile(
                name="Event horizon",
                simulations=search_simulations * 2,
                description="The deepest search and the longest think time.",
            ),
        }
        if compile_inference:
            self.network.warmup_mcts_inference(self.game)

    def analyze(self, board: chess.Board, strength: str) -> EngineDecision:
        """Search ``board`` without mutating it and return the best legal move."""
        from luna.config import MCTSParams
        from luna.mcts import MCTS

        profile = self.strengths.get(strength)
        if profile is None:
            raise ValueError(f"Unknown strength profile: {strength}")
        if board.is_game_over(claim_draw=True):
            raise ValueError("The game is already over")

        player = 1 if board.turn == chess.WHITE else -1
        canonical_board = self.game.get_canonical_form(board, player)
        params = MCTSParams(
            num_mcts_sims=profile.simulations,
            cpuct=1.25,
            dir_noise=False,
            discount=1.0,
            recurrent_policy_topk=256,
        )
        started = time.monotonic()
        deadline = started + SEARCH_DEADLINE_SECONDS
        search = MCTS(self.game, self.network, params)
        admitted = self._search_slots.acquire(timeout=SEARCH_ADMISSION_TIMEOUT_SECONDS)
        if not admitted:
            raise EngineBusyError("The inference queue is full")
        try:
            with self._inference_lock:
                probabilities, root_value = search.search_latent(
                    canonical_board,
                    num_sims=profile.simulations,
                    temp=1.0,
                    add_exploration_noise=False,
                    should_stop=lambda: time.monotonic() >= deadline,
                )
        finally:
            self._search_slots.release()
        elapsed_ms = round((time.monotonic() - started) * 1000)

        if search.last_action is None:
            raise RuntimeError("Search returned no legal continuation")
        action = search.last_action
        if probabilities[action] <= 0:
            raise RuntimeError("Search returned no legal continuation")
        next_board, _ = self.game.get_next_state(board, player, action)
        move = next_board.peek()
        evaluation_white = float(root_value if board.turn == chess.WHITE else -root_value)
        return EngineDecision(
            move=move,
            san=board.san(move),
            confidence=float(probabilities[action]),
            evaluation_white=float(np.clip(evaluation_white, -1.0, 1.0)),
            think_time_ms=elapsed_ms,
            simulations=search.last_simulations,
        )


@dataclass
class GameRecord:
    """Mutable state for one browser-owned game."""

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
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class GameRegistry:
    """Bounded, in-memory game storage with browser-session ownership checks."""

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
            owned = sorted(
                (game for game in self._games.values() if secrets.compare_digest(game.owner_id, owner_id)),
                key=lambda item: item.updated_at,
            )
            while len(owned) >= self.max_games_per_session:
                self._games.pop(owned.pop(0).game_id, None)
            while len(self._games) >= self.max_games:
                oldest = min(self._games.values(), key=lambda item: item.updated_at)
                self._games.pop(oldest.game_id, None)
            record = GameRecord(
                game_id=str(uuid.uuid4()),
                owner_id=owner_id,
                mode=mode,
                human_color=human_color,
                strength=strength,
            )
            self._games[record.game_id] = record
            return record

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


class ApiError(Exception):
    """An expected API failure with a stable machine-readable code."""

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


@contextmanager
def _locked_game(record: GameRecord) -> Iterator[None]:
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


def _color_name(color: chess.Color) -> ColorName:
    return "white" if color == chess.WHITE else "black"


def _client_id() -> str:
    client_id = session.get("luna_client_id")
    if not isinstance(client_id, str) or len(client_id) != 32:
        client_id = uuid.uuid4().hex
        session["luna_client_id"] = client_id
    return client_id


def _engine() -> LunaEngineService:
    engine = current_app.extensions.get("luna_engine")
    if not isinstance(engine, LunaEngineService):
        raise ApiError(503, "model_unavailable", "Luna is offline because no compatible model is loaded.")
    return engine


def _registry() -> GameRegistry:
    registry = current_app.extensions.get("luna_games")
    if not isinstance(registry, GameRegistry):
        raise RuntimeError("Game registry is not configured")
    return registry


def _json_body() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError(400, "invalid_json", "Send a JSON object as the request body.")
    return payload


def _require_revision(payload: Mapping[str, Any], record: GameRecord) -> None:
    revision = payload.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise ApiError(400, "missing_revision", "Include the current integer game revision.")
    current_revision = len(record.board.move_stack)
    if revision != current_revision:
        raise ApiError(
            409,
            "stale_position",
            "The position changed before this request was processed. Refresh the game and try again.",
            {"current_revision": current_revision},
        )


def _search_api_error(error: EngineBusyError | RuntimeError | ValueError) -> ApiError:
    if isinstance(error, EngineBusyError):
        return ApiError(
            429,
            "engine_busy",
            "Luna is calculating for another player. Please retry shortly.",
            headers={"Retry-After": "2"},
        )
    return ApiError(500, "search_failed", "Luna could not complete the search. Please try again.")


def _history(board: chess.Board) -> list[dict[str, Any]]:
    replay = chess.Board()
    entries: list[dict[str, Any]] = []
    for ply, move in enumerate(board.move_stack, start=1):
        entries.append(
            {
                "ply": ply,
                "move_number": replay.fullmove_number,
                "color": _color_name(replay.turn),
                "san": replay.san(move),
                "uci": move.uci(),
            }
        )
        replay.push(move)
    return entries


def _captured_pieces(board: chess.Board) -> dict[str, list[str]]:
    replay = chess.Board()
    captured: dict[str, list[str]] = {"white": [], "black": []}
    for move in board.move_stack:
        captured_piece: chess.Piece | None
        if replay.is_en_passant(move):
            offset = -8 if replay.turn == chess.WHITE else 8
            captured_piece = replay.piece_at(move.to_square + offset)
        else:
            captured_piece = replay.piece_at(move.to_square)
        if captured_piece is not None:
            captured[_color_name(replay.turn)].append(captured_piece.symbol())
        replay.push(move)
    return captured


def _result(board: chess.Board) -> dict[str, str] | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        headline = "Draw"
    else:
        headline = f"{_color_name(outcome.winner).title()} wins"
    reason = outcome.termination.name.replace("_", " ").lower()
    return {"headline": headline, "reason": reason, "notation": board.result(claim_draw=True)}


def _can_undo(record: GameRecord, history: list[dict[str, Any]]) -> bool:
    return bool(
        record.mode == "human"
        and record.human_color is not None
        and any(entry["color"] == record.human_color for entry in history)
    )


def _game_payload(record: GameRecord, engine: LunaEngineService) -> dict[str, Any]:
    history = _history(record.board)
    result = _result(record.board)
    if result is not None:
        status = result["headline"]
    elif record.mode == "selfplay":
        status = f"{_color_name(record.board.turn).title()} to calculate"
    elif _color_name(record.board.turn) == record.human_color:
        status = "Your turn"
    else:
        status = "Luna to move"

    profile = engine.strengths[record.strength]
    return {
        "id": record.game_id,
        "revision": len(record.board.move_stack),
        "mode": record.mode,
        "human_color": record.human_color,
        "strength": {
            "id": record.strength,
            "name": profile.name,
            "simulations": profile.simulations,
        },
        "fen": record.board.fen(),
        "turn": _color_name(record.board.turn),
        "legal_moves": [move.uci() for move in record.board.legal_moves],
        "last_move": history[-1]["uci"] if history else None,
        "history": history,
        "captured": _captured_pieces(record.board),
        "is_check": record.board.is_check(),
        "is_game_over": result is not None,
        "result": result,
        "status": status,
        "can_undo": _can_undo(record, history),
        "can_hint": record.mode == "human" and result is None and _color_name(record.board.turn) == record.human_color,
        "engine": {
            "last_move": record.last_engine_move,
            "think_time_ms": record.last_think_time_ms,
            "simulations": record.last_simulations,
            "evaluation_white": record.last_evaluation_white,
            "confidence": record.last_confidence,
        },
    }


def _apply_engine_move(record: GameRecord, engine: LunaEngineService) -> EngineDecision:
    decision = engine.analyze(record.board, record.strength)
    if decision.move not in record.board.legal_moves:
        raise RuntimeError("Engine produced an illegal move")
    record.board.push(decision.move)
    record.last_engine_move = decision.move.uci()
    record.last_think_time_ms = decision.think_time_ms
    record.last_simulations = decision.simulations
    record.last_evaluation_white = decision.evaluation_white
    record.last_confidence = decision.confidence
    record.updated_at = time.time()
    return decision


def _configure_security(application: Flask, config: WebAppConfig) -> None:
    if config.proxy_hops < 0:
        raise ValueError("proxy_hops cannot be negative")
    if config.hsts_max_age_seconds < 0:
        raise ValueError("hsts_max_age_seconds cannot be negative")
    if config.proxy_hops:
        application.__dict__["wsgi_app"] = ProxyFix(
            application.wsgi_app,
            x_for=config.proxy_hops,
            x_proto=config.proxy_hops,
            x_host=config.proxy_hops,
        )
    if config.trusted_hosts:
        application.config["TRUSTED_HOSTS"] = list(config.trusted_hosts)


def _secure_response(response: Response, config: WebAppConfig) -> Response:
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src-elem 'self'; "
        "style-src-attr 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
        "font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    )
    response.headers["Permissions-Policy"] = "camera=(), geolocation=(), microphone=(), payment=(), usb=()"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    if request.path.startswith(API_PREFIX):
        response.headers["Cache-Control"] = "no-store"
    if config.hsts_max_age_seconds:
        response.headers["Strict-Transport-Security"] = f"max-age={config.hsts_max_age_seconds}; includeSubDomains"
    return response


def create_app(engine: LunaEngineService | None = None, config: WebAppConfig | None = None) -> Flask:
    """Build an isolated web process around a verified inference service."""
    web_config = config or WebAppConfig()
    application = Flask(__name__, static_folder="static", static_url_path="/static")
    application.config.update(
        SECRET_KEY=os.environ.get("LUNA_WEB_SECRET", secrets.token_hex(32)),
        MAX_CONTENT_LENGTH=16 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    _configure_security(application, web_config)
    application.extensions["luna_engine"] = engine
    application.extensions["luna_games"] = GameRegistry()
    limiter = Limiter(
        key_func=get_remote_address,
        app=application,
        default_limits=["120 per minute"],
        headers_enabled=False,
        storage_uri=web_config.rate_limit_storage_uri,
        retry_after="delta-seconds",
    )

    @application.after_request
    def add_security_headers(response: Response) -> Response:
        return _secure_response(response, web_config)

    @application.errorhandler(ApiError)
    def handle_api_error(error: ApiError) -> Response:
        payload: dict[str, Any] = {"error": {"code": error.code, "message": error.message}}
        if error.details is not None:
            payload["error"]["details"] = error.details
        response = make_response(jsonify(payload), error.status)
        response.headers.update(error.headers)
        return response

    @application.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException) -> ResponseReturnValue:
        if not request.path.startswith(API_PREFIX):
            return error.get_response()
        code = error.name.lower().replace(" ", "_")
        message = "Request limit reached. Please wait before trying again." if error.code == 429 else error.description
        response = make_response(jsonify({"error": {"code": code, "message": message}}), error.code or 500)
        if error.code == 429:
            response.headers["Retry-After"] = "60"
        return response

    @application.errorhandler(Exception)
    def handle_unexpected_error(error: Exception) -> ResponseReturnValue:
        logger.opt(exception=error).error("Unhandled web request failure")
        return jsonify({"error": {"code": "internal_error", "message": "Luna encountered an unexpected error."}}), 500

    @application.route("/")
    def index() -> Response:
        return send_from_directory(Path(application.root_path), "index.html")

    @application.get(f"{API_PREFIX}/health")
    @limiter.exempt
    def health() -> Response:
        active_engine = _engine()
        return jsonify(
            {
                "data": {
                    "ready": True,
                    "engine": "Luna",
                    "checkpoint": active_engine.checkpoint_name,
                    "strengths": [
                        {
                            "id": key,
                            "name": profile.name,
                            "simulations": profile.simulations,
                            "description": profile.description,
                        }
                        for key, profile in active_engine.strengths.items()
                    ],
                }
            }
        )

    @application.post(f"{API_PREFIX}/games")
    @limiter.limit("20 per minute")
    @limiter.limit("8 per minute", key_func=_client_id)
    def create_game() -> ResponseReturnValue:
        active_engine = _engine()
        payload = _json_body()
        mode = payload.get("mode", "human")
        if mode not in {"human", "selfplay"}:
            raise ApiError(422, "invalid_mode", "Mode must be 'human' or 'selfplay'.")
        strength = payload.get("strength", "strong")
        if strength not in active_engine.strengths:
            raise ApiError(
                422,
                "invalid_strength",
                "Choose one of the available strength profiles.",
                {"available": list(active_engine.strengths)},
            )

        human_color: ColorName | None = None
        if mode == "human":
            requested_color = payload.get("color", "white")
            if requested_color == "random":
                requested_color = secrets.choice(("white", "black"))
            if requested_color not in {"white", "black"}:
                raise ApiError(422, "invalid_color", "Color must be 'white', 'black', or 'random'.")
            human_color = requested_color

        record = _registry().create(
            owner_id=_client_id(),
            mode=mode,
            human_color=human_color,
            strength=strength,
        )
        with record.lock:
            if record.mode == "human" and record.human_color == "black":
                try:
                    _apply_engine_move(record, active_engine)
                except (EngineBusyError, RuntimeError, ValueError) as exc:
                    _registry().delete(record.game_id, record.owner_id)
                    if not isinstance(exc, EngineBusyError):
                        logger.exception("Could not calculate the opening move")
                    raise _search_api_error(exc) from None
            response = _game_payload(record, active_engine)
        return jsonify({"data": response}), 201

    @application.get(f"{API_PREFIX}/games/<game_id>")
    def game_state(game_id: str) -> Response:
        active_engine = _engine()
        record = _registry().get(game_id, _client_id())
        with _locked_game(record):
            return jsonify({"data": _game_payload(record, active_engine)})

    @application.delete(f"{API_PREFIX}/games/<game_id>")
    def delete_game(game_id: str) -> ResponseReturnValue:
        registry = _registry()
        owner_id = _client_id()
        record = registry.get(game_id, owner_id)
        with _locked_game(record):
            registry.delete(game_id, owner_id)
        return "", 204

    @application.post(f"{API_PREFIX}/games/<game_id>/moves")
    @limiter.limit("30 per minute")
    @limiter.limit("20 per minute", key_func=_client_id)
    def human_move(game_id: str) -> Response:
        active_engine = _engine()
        record = _registry().get(game_id, _client_id())
        payload = _json_body()
        move_text = payload.get("move")
        if not isinstance(move_text, str):
            raise ApiError(400, "missing_move", "Provide a move in UCI notation.")

        with _locked_game(record):
            _require_revision(payload, record)
            if record.mode != "human" or record.human_color is None:
                raise ApiError(409, "wrong_mode", "Moves can only be submitted in a human game.")
            if record.board.is_game_over(claim_draw=True):
                raise ApiError(409, "game_over", "The game is already over.")
            if _color_name(record.board.turn) != record.human_color:
                raise ApiError(409, "not_your_turn", "Wait for Luna to finish its move.")

            try:
                move = chess.Move.from_uci(move_text.lower())
            except ValueError as exc:
                raise ApiError(422, "invalid_move", "That is not valid UCI move notation.") from exc

            promotion_options = sorted(
                candidate.uci()
                for candidate in record.board.legal_moves
                if candidate.uci().startswith(move_text.lower()[:4]) and candidate.promotion is not None
            )
            if move not in record.board.legal_moves:
                if len(move_text) == 4 and promotion_options:
                    raise ApiError(
                        422,
                        "promotion_required",
                        "Choose a promotion piece.",
                        {"moves": promotion_options},
                    )
                raise ApiError(422, "illegal_move", "That move is not legal in this position.")

            previous = record.board.copy(stack=True)
            analytics = (
                record.last_engine_move,
                record.last_think_time_ms,
                record.last_simulations,
                record.last_evaluation_white,
                record.last_confidence,
            )
            human_san = record.board.san(move)
            record.board.push(move)
            engine_decision: EngineDecision | None = None
            search_succeeded = False
            try:
                if not record.board.is_game_over(claim_draw=True):
                    engine_decision = _apply_engine_move(record, active_engine)
                search_succeeded = True
            except (EngineBusyError, RuntimeError, ValueError) as exc:
                if not isinstance(exc, EngineBusyError):
                    logger.exception("Engine search failed after a human move")
                error = _search_api_error(exc)
                error.message = (
                    "Luna is busy; your move was not applied. Please retry shortly."
                    if isinstance(exc, EngineBusyError)
                    else "Luna could not calculate a reply. Your move was not applied."
                )
                raise error from None
            finally:
                if not search_succeeded:
                    record.board = previous
                    (
                        record.last_engine_move,
                        record.last_think_time_ms,
                        record.last_simulations,
                        record.last_evaluation_white,
                        record.last_confidence,
                    ) = analytics

            events: list[dict[str, Any]] = [{"actor": "human", "move": move.uci(), "san": human_san}]
            if engine_decision is not None:
                events.append(
                    {
                        "actor": "luna",
                        "move": engine_decision.move.uci(),
                        "san": engine_decision.san,
                        "think_time_ms": engine_decision.think_time_ms,
                    }
                )
            return jsonify({"data": _game_payload(record, active_engine), "events": events})

    @application.post(f"{API_PREFIX}/games/<game_id>/engine-move")
    @limiter.limit("24 per minute")
    @limiter.limit("16 per minute", key_func=_client_id)
    def selfplay_move(game_id: str) -> Response:
        active_engine = _engine()
        record = _registry().get(game_id, _client_id())
        payload = _json_body()
        with _locked_game(record):
            _require_revision(payload, record)
            if record.mode != "selfplay":
                raise ApiError(409, "wrong_mode", "Engine stepping is only available in observatory mode.")
            if record.board.is_game_over(claim_draw=True):
                raise ApiError(409, "game_over", "The game is already over.")
            previous = record.board.copy(stack=True)
            try:
                decision = _apply_engine_move(record, active_engine)
            except (EngineBusyError, RuntimeError, ValueError) as exc:
                record.board = previous
                if not isinstance(exc, EngineBusyError):
                    logger.exception("Self-play search failed")
                raise _search_api_error(exc) from None
            return jsonify(
                {
                    "data": _game_payload(record, active_engine),
                    "events": [
                        {
                            "actor": "luna",
                            "move": decision.move.uci(),
                            "san": decision.san,
                            "think_time_ms": decision.think_time_ms,
                        }
                    ],
                }
            )

    @application.post(f"{API_PREFIX}/games/<game_id>/hint")
    @limiter.limit("12 per minute")
    @limiter.limit("6 per minute", key_func=_client_id)
    def hint(game_id: str) -> Response:
        active_engine = _engine()
        record = _registry().get(game_id, _client_id())
        payload = _json_body()
        with _locked_game(record):
            _require_revision(payload, record)
            if record.mode != "human" or record.human_color is None:
                raise ApiError(409, "wrong_mode", "Hints are only available while playing Luna.")
            if record.board.is_game_over(claim_draw=True):
                raise ApiError(409, "game_over", "The game is already over.")
            if _color_name(record.board.turn) != record.human_color:
                raise ApiError(409, "not_your_turn", "Hints are available on your turn.")
            try:
                decision = active_engine.analyze(record.board, record.strength)
            except (EngineBusyError, RuntimeError, ValueError) as exc:
                if not isinstance(exc, EngineBusyError):
                    logger.exception("Hint search failed")
                raise _search_api_error(exc) from None
            record.last_evaluation_white = decision.evaluation_white
            record.last_confidence = decision.confidence
            record.last_think_time_ms = decision.think_time_ms
            record.last_simulations = decision.simulations
            return jsonify(
                {
                    "data": {
                        "move": decision.move.uci(),
                        "san": decision.san,
                        "confidence": decision.confidence,
                        "evaluation_white": decision.evaluation_white,
                        "think_time_ms": decision.think_time_ms,
                        "simulations": decision.simulations,
                    }
                }
            )

    @application.post(f"{API_PREFIX}/games/<game_id>/undo")
    @limiter.limit("30 per minute")
    @limiter.limit("20 per minute", key_func=_client_id)
    def undo(game_id: str) -> Response:
        active_engine = _engine()
        record = _registry().get(game_id, _client_id())
        payload = _json_body()
        with _locked_game(record):
            _require_revision(payload, record)
            history = _history(record.board)
            if not _can_undo(record, history) or record.human_color is None:
                raise ApiError(409, "nothing_to_undo", "There is no completed player move to undo.")

            removed_human_move = False
            while record.board.move_stack:
                record.board.pop()
                if _color_name(record.board.turn) == record.human_color:
                    removed_human_move = True
                    break
            if not removed_human_move:
                raise RuntimeError("Undo invariant violated")
            record.last_engine_move = None
            record.last_think_time_ms = None
            record.last_simulations = None
            record.last_evaluation_white = None
            record.last_confidence = None
            record.updated_at = time.time()
            return jsonify({"data": _game_payload(record, active_engine)})

    return application


@dataclass
class WebServeConfig:
    """Command-line options for the local Luna web server."""

    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    device: str = "cuda"
    checkpoint: str = "./temp/latest.pth.tar"
    search_simulations: int = 96
    compile_inference: bool = True


def main() -> None:
    """Load a verified checkpoint, then start the HTTP server."""
    cfg = tyro.cli(WebServeConfig)
    try:
        engine = LunaEngineService(
            cfg.checkpoint,
            device=cfg.device,
            search_simulations=cfg.search_simulations,
            compile_inference=cfg.compile_inference,
        )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError):
        logger.exception("Luna web server refused to start: the checkpoint could not be loaded")
        raise SystemExit(2) from None

    application = create_app(engine)
    logger.info("Luna web interface ready at http://{}:{}", cfg.host, cfg.port)
    application.run(host=cfg.host, port=cfg.port, debug=cfg.debug)


if __name__ == "__main__":
    main()
