from __future__ import annotations

import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from flask import Flask, Response, jsonify, send_from_directory
from flask.typing import ResponseReturnValue
from flask_limiter import Limiter
from loguru import logger

from web_engine import ColorName, EngineBusyError, GameMode, StrengthProfile
from web_games import (
    API_PREFIX,
    ApiError,
    apply_engine_move,
    client_id,
    engine,
    game_payload,
    json_body,
    locked_game,
    registry,
    search_api_error,
)


def register_game_routes(application: Flask, limiter: Limiter) -> None:
    @application.route("/")
    def index() -> Response:
        return send_from_directory(Path(application.root_path), "index.html")

    @application.get(f"{API_PREFIX}/health")
    @limiter.exempt
    def health() -> Response:
        active_engine = engine()
        strengths = [
            {
                "id": key,
                "name": profile.name,
                "simulations": profile.simulations,
                "description": profile.description,
            }
            for key, profile in active_engine.strengths.items()
        ]
        return jsonify(
            {
                "data": {
                    "ready": True,
                    "engine": "Luna",
                    "checkpoint": active_engine.checkpoint_name,
                    "strengths": strengths,
                }
            }
        )

    @application.post(f"{API_PREFIX}/games")
    @limiter.limit("20 per minute")
    @limiter.limit("8 per minute", key_func=client_id)
    def create_game() -> ResponseReturnValue:
        active_engine = engine()
        payload = json_body()
        mode = _game_mode(payload)
        strength = _strength(payload, active_engine.strengths)
        human_color = _human_color(payload, mode)
        record = registry().create(
            owner_id=client_id(),
            mode=mode,
            human_color=human_color,
            strength=strength,
        )
        with record.lock:
            if record.mode == "human" and record.human_color == "black":
                try:
                    apply_engine_move(record, active_engine)
                except (EngineBusyError, RuntimeError, ValueError) as exc:
                    registry().delete(record.game_id, record.owner_id)
                    if not isinstance(exc, EngineBusyError):
                        logger.exception("Could not calculate the opening move")
                    raise search_api_error(exc) from None
            response = game_payload(record, active_engine)
        return jsonify({"data": response}), 201

    @application.get(f"{API_PREFIX}/games/<game_id>")
    def game_state(game_id: str) -> Response:
        active_engine = engine()
        record = registry().get(game_id, client_id())
        with locked_game(record):
            return jsonify({"data": game_payload(record, active_engine)})

    @application.delete(f"{API_PREFIX}/games/<game_id>")
    def delete_game(game_id: str) -> ResponseReturnValue:
        active_registry = registry()
        owner_id = client_id()
        record = active_registry.get(game_id, owner_id)
        with locked_game(record):
            active_registry.delete(game_id, owner_id)
        return "", 204


def _game_mode(payload: dict[str, Any]) -> GameMode:
    value = payload.get("mode", "human")
    if not isinstance(value, str) or value not in {"human", "selfplay"}:
        raise ApiError(422, "invalid_mode", "Mode must be 'human' or 'selfplay'.")
    return cast(GameMode, value)


def _strength(payload: dict[str, Any], strengths: Mapping[str, StrengthProfile]) -> str:
    value = payload.get("strength", "strong")
    if not isinstance(value, str) or value not in strengths:
        raise ApiError(
            422,
            "invalid_strength",
            "Choose one of the available strength profiles.",
            {"available": list(strengths)},
        )
    return value


def _human_color(payload: dict[str, Any], mode: GameMode) -> ColorName | None:
    if mode != "human":
        return None
    value = payload.get("color", "white")
    if value == "random":
        value = secrets.choice(("white", "black"))
    if not isinstance(value, str) or value not in {"white", "black"}:
        raise ApiError(422, "invalid_color", "Color must be 'white', 'black', or 'random'.")
    return cast(ColorName, value)
