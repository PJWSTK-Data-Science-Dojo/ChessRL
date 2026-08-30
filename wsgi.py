"""Fail-closed WSGI entry point for the Luna web application."""

from __future__ import annotations

import os
from pathlib import Path

from src.web_app import LunaEngineService, WebAppConfig, create_app, parse_exact_trusted_hosts


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value")


def _environment_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise RuntimeError(f"{name} cannot be negative")
    return parsed


def _trusted_hosts() -> tuple[str, ...]:
    return parse_exact_trusted_hosts(os.environ.get("LUNA_TRUSTED_HOSTS", ""))


web_secret = os.environ.get("LUNA_WEB_SECRET", "")
if len(web_secret) < 32:
    raise RuntimeError("LUNA_WEB_SECRET must contain at least 32 characters")

checkpoint_path = Path(os.environ.get("CHECKPOINT_PATH", "./runs/luna-main/latest.pth.tar"))
engine = LunaEngineService(
    checkpoint_path,
    device=os.environ.get("DEVICE", "cuda"),
    search_simulations=int(os.environ.get("SEARCH_SIMULATIONS", "96")),
    compile_inference=_environment_bool("COMPILE_INFERENCE", False),
)
app = create_app(
    engine,
    WebAppConfig(
        trusted_hosts=_trusted_hosts(),
        proxy_hops=_environment_int("PROXY_HOPS", 0),
        hsts_max_age_seconds=_environment_int("HSTS_MAX_AGE_SECONDS", 0),
    ),
)
app.config.update(
    SECRET_KEY=web_secret,
    SESSION_COOKIE_NAME="luna_session",
    SESSION_COOKIE_SECURE=_environment_bool("SESSION_COOKIE_SECURE", True),
)
