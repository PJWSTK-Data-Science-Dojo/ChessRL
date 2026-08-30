"""Gunicorn settings for one shared in-memory model and game registry."""

import os


def _bounded_environment_int(name: str, default: int, minimum: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return value


bind = f"{os.environ.get('HOST', '127.0.0.1')}:{os.environ.get('PORT', '5000')}"
workers = 1
worker_class = "gthread"
threads = _bounded_environment_int("THREADS", 4, 1)
timeout = _bounded_environment_int("TIMEOUT", 180, 30)
graceful_timeout = _bounded_environment_int("GRACEFUL_TIMEOUT", 30, 10)
keepalive = 5

accesslog = None
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
capture_output = False
proc_name = "luna-chess-web"

limit_request_line = 4094
limit_request_fields = 64
limit_request_field_size = 8190
