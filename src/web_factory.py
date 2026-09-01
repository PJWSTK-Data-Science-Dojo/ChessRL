from __future__ import annotations

import os
import secrets
from typing import Any

from flask import Flask, Response, jsonify, make_response, request
from flask.typing import ResponseReturnValue
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from loguru import logger
from werkzeug.exceptions import HTTPException

from web_engine import LunaEngineService
from web_games import API_PREFIX, ApiError, GameRegistry
from web_move_routes import register_move_routes
from web_routes import register_game_routes
from web_security import WebAppConfig, configure_security, secure_response


def create_app(engine: LunaEngineService | None = None, config: WebAppConfig | None = None) -> Flask:
    web_config = config or WebAppConfig()
    application = Flask("web_app", static_folder="static", static_url_path="/static")
    application.config.update(
        SECRET_KEY=os.environ.get("LUNA_WEB_SECRET", secrets.token_hex(32)),
        MAX_CONTENT_LENGTH=16 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=web_config.secure_cookies,
    )
    configure_security(application, web_config)
    application.extensions["luna_engine"] = engine
    application.extensions["luna_games"] = GameRegistry()
    limiter = _limiter(application, web_config)
    _register_hooks(application, web_config)
    register_game_routes(application, limiter)
    register_move_routes(application, limiter)
    return application


def _limiter(application: Flask, config: WebAppConfig) -> Limiter:
    return Limiter(
        key_func=get_remote_address,
        app=application,
        default_limits=["120 per minute"],
        headers_enabled=False,
        storage_uri=config.rate_limit_storage_uri,
        retry_after="delta-seconds",
    )


def _register_hooks(application: Flask, config: WebAppConfig) -> None:
    @application.after_request
    def add_security_headers(response: Response) -> Response:
        return secure_response(response, config)

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
        message = _http_error_message(error)
        response = make_response(jsonify({"error": {"code": code, "message": message}}), error.code or 500)
        if error.code == 429:
            response.headers["Retry-After"] = "60"
        return response

    @application.errorhandler(Exception)
    def handle_unexpected_error(error: Exception) -> ResponseReturnValue:
        logger.opt(exception=error).error("Unhandled web request failure")
        return jsonify({"error": {"code": "internal_error", "message": "Luna encountered an unexpected error."}}), 500


def _http_error_message(error: HTTPException) -> str:
    if error.code == 429:
        return "Request limit reached. Please wait before trying again."
    return error.description or "The request could not be completed."
