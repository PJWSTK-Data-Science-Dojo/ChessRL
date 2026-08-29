"""Fail-closed WSGI entry point for the Luna web application."""

from __future__ import annotations

import os
from pathlib import Path

from src.web_app import LunaEngineService, create_app


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


web_secret = os.environ.get("LUNA_WEB_SECRET", "")
if len(web_secret) < 32:
    raise RuntimeError("LUNA_WEB_SECRET must contain at least 32 characters")

checkpoint_path = Path(os.environ.get("CHECKPOINT_PATH", "./temp/latest.pth.tar"))
engine = LunaEngineService(
    checkpoint_path,
    device=os.environ.get("DEVICE", "cuda"),
    search_simulations=int(os.environ.get("SEARCH_SIMULATIONS", "96")),
    compile_inference=_environment_bool("COMPILE_INFERENCE", True),
)
app = create_app(engine)
app.config.update(
    SECRET_KEY=web_secret,
    SESSION_COOKIE_SECURE=_environment_bool("SESSION_COOKIE_SECURE", True),
)


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
