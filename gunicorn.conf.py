"""Gunicorn settings for one shared in-memory model and game registry."""

import os

bind = f"{os.environ.get('HOST', '127.0.0.1')}:{os.environ.get('PORT', '5000')}"
workers = 1
worker_class = "gthread"
threads = max(1, int(os.environ.get("THREADS", "4")))
timeout = max(30, int(os.environ.get("TIMEOUT", "180")))
graceful_timeout = max(10, int(os.environ.get("GRACEFUL_TIMEOUT", "30")))
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
capture_output = False
proc_name = "luna-chess-web"

limit_request_line = 4094
limit_request_fields = 64
limit_request_field_size = 8190
