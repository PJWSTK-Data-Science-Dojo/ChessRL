from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import chess
from flask import Flask, Response, jsonify
from flask_limiter import Limiter
from loguru import logger

from web_engine import EngineBusyError, EngineDecision, LunaEngineService
from web_games import (
    API_PREFIX,
    ApiError,
    GameRecord,
    apply_engine_move,
    can_undo,
    client_id,
    color_name,
    engine,
    game_payload,
    history,
    json_body,
    locked_game,
    registry,
    require_revision,
    search_api_error,
)


@dataclass(frozen=True)
class _GameSnapshot:
    board: chess.Board
    revision: int
    engine_metrics: tuple[str | None, int | None, int | None, float | None, float | None]


def register_move_routes(application: Flask, limiter: Limiter) -> None:
    @application.post(f"{API_PREFIX}/games/<game_id>/moves")
    @limiter.limit("30 per minute")
    @limiter.limit("20 per minute", key_func=client_id)
    def human_move(game_id: str) -> Response:
        active_engine = engine()
        record = registry().get(game_id, client_id())
        payload = json_body()
        with locked_game(record):
            require_revision(payload, record)
            _validate_human_turn(record)
            move = _legal_move(record, payload)
            try:
                human_san, decision = _apply_human_turn(record, move, active_engine)
            except (EngineBusyError, RuntimeError, ValueError) as exc:
                if not isinstance(exc, EngineBusyError):
                    logger.exception("Engine search failed after a human move")
                error = search_api_error(exc)
                error.message = _human_search_error(exc)
                raise error from None
            events = [_move_event("human", move, human_san)]
            if decision is not None:
                events.append(_move_event("luna", decision.move, decision.san, decision.think_time_ms))
            return jsonify({"data": game_payload(record, active_engine), "events": events})

    @application.post(f"{API_PREFIX}/games/<game_id>/engine-move")
    @limiter.limit("24 per minute")
    @limiter.limit("16 per minute", key_func=client_id)
    def selfplay_move(game_id: str) -> Response:
        active_engine = engine()
        record = registry().get(game_id, client_id())
        payload = json_body()
        with locked_game(record):
            require_revision(payload, record)
            _validate_selfplay_turn(record)
            snapshot = _snapshot(record)
            try:
                decision = apply_engine_move(record, active_engine)
            except (EngineBusyError, RuntimeError, ValueError) as exc:
                _restore(record, snapshot)
                if not isinstance(exc, EngineBusyError):
                    logger.exception("Self-play search failed")
                raise search_api_error(exc) from None
            event = _move_event("luna", decision.move, decision.san, decision.think_time_ms)
            return jsonify({"data": game_payload(record, active_engine), "events": [event]})

    @application.post(f"{API_PREFIX}/games/<game_id>/hint")
    @limiter.limit("12 per minute")
    @limiter.limit("6 per minute", key_func=client_id)
    def hint(game_id: str) -> Response:
        active_engine = engine()
        record = registry().get(game_id, client_id())
        payload = json_body()
        with locked_game(record):
            require_revision(payload, record)
            _validate_hint(record)
            try:
                decision = active_engine.analyze(record.board, record.strength)
            except (EngineBusyError, RuntimeError, ValueError) as exc:
                if not isinstance(exc, EngineBusyError):
                    logger.exception("Hint search failed")
                raise search_api_error(exc) from None
            _save_hint_metrics(record, decision)
            return jsonify({"data": _decision_payload(decision)})

    @application.post(f"{API_PREFIX}/games/<game_id>/undo")
    @limiter.limit("30 per minute")
    @limiter.limit("20 per minute", key_func=client_id)
    def undo(game_id: str) -> Response:
        active_engine = engine()
        record = registry().get(game_id, client_id())
        payload = json_body()
        with locked_game(record):
            require_revision(payload, record)
            _undo_human_turn(record)
            return jsonify({"data": game_payload(record, active_engine)})


def _validate_human_turn(record: GameRecord) -> None:
    if record.mode != "human" or record.human_color is None:
        raise ApiError(409, "wrong_mode", "Moves can only be submitted in a human game.")
    if record.board.is_game_over(claim_draw=True):
        raise ApiError(409, "game_over", "The game is already over.")
    if color_name(record.board.turn) != record.human_color:
        raise ApiError(409, "not_your_turn", "Wait for Luna to finish its move.")


def _validate_selfplay_turn(record: GameRecord) -> None:
    if record.mode != "selfplay":
        raise ApiError(409, "wrong_mode", "Engine stepping is only available in observatory mode.")
    if record.board.is_game_over(claim_draw=True):
        raise ApiError(409, "game_over", "The game is already over.")


def _validate_hint(record: GameRecord) -> None:
    if record.mode != "human" or record.human_color is None:
        raise ApiError(409, "wrong_mode", "Hints are only available while playing Luna.")
    if record.board.is_game_over(claim_draw=True):
        raise ApiError(409, "game_over", "The game is already over.")
    if color_name(record.board.turn) != record.human_color:
        raise ApiError(409, "not_your_turn", "Hints are available on your turn.")


def _legal_move(record: GameRecord, payload: dict[str, Any]) -> chess.Move:
    move_text = payload.get("move")
    if not isinstance(move_text, str):
        raise ApiError(400, "missing_move", "Provide a move in UCI notation.")
    try:
        move = chess.Move.from_uci(move_text.lower())
    except ValueError as exc:
        raise ApiError(422, "invalid_move", "That is not valid UCI move notation.") from exc
    if move in record.board.legal_moves:
        return move
    promotions = _promotion_options(record.board, move_text)
    if len(move_text) == 4 and promotions:
        raise ApiError(422, "promotion_required", "Choose a promotion piece.", {"moves": promotions})
    raise ApiError(422, "illegal_move", "That move is not legal in this position.")


def _promotion_options(board: chess.Board, move_text: str) -> list[str]:
    prefix = move_text.lower()[:4]
    return sorted(
        move.uci() for move in board.legal_moves if move.uci().startswith(prefix) and move.promotion is not None
    )


def _apply_human_turn(
    record: GameRecord,
    move: chess.Move,
    active_engine: LunaEngineService,
) -> tuple[str, EngineDecision | None]:
    snapshot = _snapshot(record)
    completed = False
    try:
        human_san = record.board.san(move)
        record.board.push(move)
        record.revision += 1
        decision = None if record.board.is_game_over(claim_draw=True) else apply_engine_move(record, active_engine)
        completed = True
        return human_san, decision
    finally:
        if not completed:
            _restore(record, snapshot)


def _snapshot(record: GameRecord) -> _GameSnapshot:
    metrics = (
        record.last_engine_move,
        record.last_think_time_ms,
        record.last_simulations,
        record.last_evaluation_white,
        record.last_confidence,
    )
    return _GameSnapshot(record.board.copy(stack=True), record.revision, metrics)


def _restore(record: GameRecord, snapshot: _GameSnapshot) -> None:
    record.board = snapshot.board
    record.revision = snapshot.revision
    (
        record.last_engine_move,
        record.last_think_time_ms,
        record.last_simulations,
        record.last_evaluation_white,
        record.last_confidence,
    ) = snapshot.engine_metrics


def _human_search_error(error: EngineBusyError | RuntimeError | ValueError) -> str:
    if isinstance(error, EngineBusyError):
        return "Luna is busy; your move was not applied. Please retry shortly."
    return "Luna could not calculate a reply. Your move was not applied."


def _move_event(actor: str, move: chess.Move, san: str, think_time_ms: int | None = None) -> dict[str, Any]:
    event: dict[str, Any] = {"actor": actor, "move": move.uci(), "san": san}
    if think_time_ms is not None:
        event["think_time_ms"] = think_time_ms
    return event


def _save_hint_metrics(record: GameRecord, decision: EngineDecision) -> None:
    record.last_evaluation_white = decision.evaluation_white
    record.last_confidence = decision.confidence
    record.last_think_time_ms = decision.think_time_ms
    record.last_simulations = decision.simulations


def _decision_payload(decision: EngineDecision) -> dict[str, str | int | float]:
    return {
        "move": decision.move.uci(),
        "san": decision.san,
        "confidence": decision.confidence,
        "evaluation_white": decision.evaluation_white,
        "think_time_ms": decision.think_time_ms,
        "simulations": decision.simulations,
    }


def _undo_human_turn(record: GameRecord) -> None:
    entries = history(record.board)
    if not can_undo(record, entries) or record.human_color is None:
        raise ApiError(409, "nothing_to_undo", "There is no completed player move to undo.")
    while record.board.move_stack:
        record.board.pop()
        if color_name(record.board.turn) == record.human_color:
            _clear_engine_metrics(record)
            return
    raise RuntimeError("Undo invariant violated")


def _clear_engine_metrics(record: GameRecord) -> None:
    record.revision += 1
    record.last_engine_move = None
    record.last_think_time_ms = None
    record.last_simulations = None
    record.last_evaluation_white = None
    record.last_confidence = None
    record.updated_at = time.time()
