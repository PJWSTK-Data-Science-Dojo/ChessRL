from __future__ import annotations

from dataclasses import dataclass

from flask import Flask, Response, request
from werkzeug.middleware.proxy_fix import ProxyFix

from web_games import API_PREFIX


@dataclass(frozen=True)
class WebAppConfig:
    trusted_hosts: tuple[str, ...] = ()
    proxy_hops: int = 0
    hsts_max_age_seconds: int = 0
    secure_cookies: bool = False
    rate_limit_storage_uri: str = "memory://"


def parse_exact_trusted_hosts(value: str) -> tuple[str, ...]:
    hosts = tuple(host.strip() for host in value.split(",") if host.strip())
    if not hosts:
        raise RuntimeError("LUNA_TRUSTED_HOSTS must contain at least one exact host")
    if any(host.startswith(".") or "*" in host or "://" in host or "/" in host for host in hosts):
        raise RuntimeError("LUNA_TRUSTED_HOSTS accepts exact host names only")
    return hosts


def configure_security(application: Flask, config: WebAppConfig) -> None:
    if config.proxy_hops < 0:
        raise ValueError("proxy_hops cannot be negative")
    if config.hsts_max_age_seconds < 0:
        raise ValueError("hsts_max_age_seconds cannot be negative")
    if not isinstance(config.secure_cookies, bool):
        raise ValueError("secure_cookies must be boolean")
    if config.proxy_hops:
        application.__dict__["wsgi_app"] = ProxyFix(
            application.wsgi_app,
            x_for=config.proxy_hops,
            x_proto=config.proxy_hops,
            x_host=config.proxy_hops,
        )
    if config.trusted_hosts:
        application.config["TRUSTED_HOSTS"] = list(config.trusted_hosts)


def secure_response(response: Response, config: WebAppConfig) -> Response:
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
