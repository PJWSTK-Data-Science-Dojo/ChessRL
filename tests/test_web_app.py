from __future__ import annotations

import threading

import chess
import pytest
from flask import Flask

from web_app import (
    ApiError,
    EngineBusyError,
    EngineDecision,
    GameRegistry,
    LunaEngineService,
    StrengthProfile,
    WebAppConfig,
    create_app,
    parse_exact_trusted_hosts,
)


class _DeterministicEngine(LunaEngineService):
    def __init__(self) -> None:
        self.checkpoint_name = "test-v2.pth.tar"
        self.device = "cpu"
        self.strengths = {
            "quick": StrengthProfile("Quick scan", 8, "test"),
            "strong": StrengthProfile("Deep orbit", 16, "test"),
            "maximum": StrengthProfile("Event horizon", 32, "test"),
        }

    def analyze(self, board: chess.Board, strength: str) -> EngineDecision:
        profile = self.strengths[strength]
        move = sorted(board.legal_moves, key=lambda candidate: candidate.uci())[0]
        return EngineDecision(
            move=move,
            san=board.san(move),
            confidence=0.75,
            evaluation_white=0.1 if board.turn == chess.WHITE else -0.1,
            think_time_ms=3,
            simulations=profile.simulations,
        )


def _app() -> Flask:
    application = create_app(_DeterministicEngine())
    application.config.update(TESTING=True, SECRET_KEY="test-secret")
    return application


def test_health_and_assets_are_served() -> None:
    client = _app().test_client()

    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.get_json()["data"]["checkpoint"] == "test-v2.pth.tar"
    assert client.get("/").status_code == 200
    assert client.get("/static/luna.css").status_code == 200
    assert client.get("/static/luna.js").status_code == 200


def test_human_game_reply_and_undo_are_transactional() -> None:
    client = _app().test_client()
    created = client.post(
        "/api/v1/games",
        json={"mode": "human", "color": "white", "strength": "strong"},
    )
    assert created.status_code == 201
    game_id = created.get_json()["data"]["id"]

    played = client.post(f"/api/v1/games/{game_id}/moves", json={"move": "e2e4", "revision": 0})
    assert played.status_code == 200
    payload = played.get_json()
    assert [event["actor"] for event in payload["events"]] == ["human", "luna"]
    assert len(payload["data"]["history"]) == 2
    assert payload["data"]["turn"] == "white"

    undone = client.post(f"/api/v1/games/{game_id}/undo", json={"revision": 2})
    assert undone.status_code == 200
    undone_state = undone.get_json()["data"]
    assert undone_state["history"] == []
    assert undone_state["revision"] == 3

    stale = client.post(f"/api/v1/games/{game_id}/moves", json={"move": "e2e4", "revision": 0})
    assert stale.status_code == 409
    assert stale.get_json()["error"]["details"] == {"current_revision": 3}


def test_games_are_isolated_by_browser_session() -> None:
    app = _app()
    owner = app.test_client()
    stranger = app.test_client()
    created = owner.post("/api/v1/games", json={"mode": "selfplay", "strength": "quick"})
    game_id = created.get_json()["data"]["id"]

    assert owner.get(f"/api/v1/games/{game_id}").status_code == 200
    hidden = stranger.get(f"/api/v1/games/{game_id}")
    assert hidden.status_code == 404
    assert hidden.get_json()["error"]["code"] == "game_not_found"


def test_per_session_limit_cannot_evict_another_owners_game() -> None:
    registry = GameRegistry(max_games=3, max_games_per_session=1)
    other = registry.create(owner_id="other", mode="selfplay", human_color=None, strength="quick")
    first = registry.create(owner_id="attacker", mode="selfplay", human_color=None, strength="quick")
    second = registry.create(owner_id="attacker", mode="selfplay", human_color=None, strength="quick")

    assert registry.get(other.game_id, "other") is other
    assert registry.get(second.game_id, "attacker") is second
    try:
        registry.get(first.game_id, "attacker")
    except ApiError as error:
        assert error.code == "game_not_found"
    else:
        raise AssertionError("the oldest game from an over-limit session was not evicted")


def test_selfplay_and_structured_validation_errors() -> None:
    client = _app().test_client()
    created = client.post("/api/v1/games", json={"mode": "selfplay", "strength": "quick"})
    game_id = created.get_json()["data"]["id"]

    step = client.post(f"/api/v1/games/{game_id}/engine-move", json={"revision": 0})
    assert step.status_code == 200
    assert len(step.get_json()["data"]["history"]) == 1
    assert step.get_json()["data"]["revision"] == 1

    invalid = client.post("/api/v1/games", json={"mode": "arcade"})
    assert invalid.status_code == 422
    assert invalid.get_json()["error"]["code"] == "invalid_mode"


@pytest.mark.parametrize(
    ("payload", "error_code"),
    [
        ({"mode": []}, "invalid_mode"),
        ({"mode": "human", "strength": []}, "invalid_strength"),
        ({"mode": "human", "color": []}, "invalid_color"),
    ],
)
def test_game_creation_rejects_non_string_options(payload: dict[str, object], error_code: str) -> None:
    response = _app().test_client().post("/api/v1/games", json=payload)

    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == error_code


def test_missing_model_reports_service_unavailable() -> None:
    application = create_app(None)
    application.config.update(TESTING=True, SECRET_KEY="test-secret")
    response = application.test_client().get("/api/v1/health")

    assert response.status_code == 503
    assert response.get_json()["error"]["code"] == "model_unavailable"


def test_mutations_reject_stale_game_revisions() -> None:
    client = _app().test_client()
    created = client.post("/api/v1/games", json={"mode": "human", "color": "white", "strength": "quick"})
    game_id = created.get_json()["data"]["id"]

    response = client.post(f"/api/v1/games/{game_id}/moves", json={"move": "e2e4", "revision": 1})

    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "stale_position"
    assert response.get_json()["error"]["details"] == {"current_revision": 0}


class _BusyEngine(_DeterministicEngine):
    def analyze(self, board: chess.Board, strength: str) -> EngineDecision:
        raise EngineBusyError("test queue is full")


class _UnexpectedEngine(_DeterministicEngine):
    def analyze(self, board: chess.Board, strength: str) -> EngineDecision:
        del board, strength
        raise TypeError("unexpected inference failure")


def test_busy_engine_sheds_load_without_applying_the_move() -> None:
    application = create_app(_BusyEngine())
    application.config.update(TESTING=True, SECRET_KEY="test-secret")
    client = application.test_client()
    created = client.post("/api/v1/games", json={"mode": "human", "color": "white", "strength": "quick"})
    game_id = created.get_json()["data"]["id"]

    response = client.post(f"/api/v1/games/{game_id}/moves", json={"move": "e2e4", "revision": 0})

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "2"
    assert response.get_json()["error"]["code"] == "engine_busy"
    state = client.get(f"/api/v1/games/{game_id}").get_json()["data"]
    assert state["revision"] == 0


def test_unexpected_engine_failure_rolls_back_the_human_move() -> None:
    application = create_app(_UnexpectedEngine())
    application.config.update(TESTING=True, SECRET_KEY="test-secret")
    client = application.test_client()
    created = client.post("/api/v1/games", json={"mode": "human", "color": "white", "strength": "quick"})
    game_id = created.get_json()["data"]["id"]

    response = client.post(f"/api/v1/games/{game_id}/moves", json={"move": "e2e4", "revision": 0})

    assert response.status_code == 500
    state = client.get(f"/api/v1/games/{game_id}").get_json()["data"]
    assert state["revision"] == 0
    assert state["turn"] == "white"


def test_security_headers_and_trusted_hosts() -> None:
    application = create_app(
        _DeterministicEngine(),
        WebAppConfig(
            trusted_hosts=("play.example.test",),
            hsts_max_age_seconds=31_536_000,
            secure_cookies=True,
        ),
    )
    application.config.update(TESTING=True, SECRET_KEY="test-secret")
    client = application.test_client()

    response = client.get("/api/v1/health", headers={"Host": "play.example.test"})

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Content-Security-Policy"].startswith("default-src 'self'")
    assert response.headers["Strict-Transport-Security"].startswith("max-age=31536000")
    assert application.config["SESSION_COOKIE_SECURE"] is True
    assert client.get("/api/v1/health", headers={"Host": "untrusted.example.test"}).status_code == 400


def test_local_web_app_keeps_cookie_usable_over_http() -> None:
    application = create_app(_DeterministicEngine(), WebAppConfig())

    assert application.config["SESSION_COOKIE_SECURE"] is False


@pytest.mark.parametrize("value", ["", "*.example.test", ".example.test", "https://example.test"])
def test_production_trusted_hosts_require_exact_names(value: str) -> None:
    with pytest.raises(RuntimeError, match="LUNA_TRUSTED_HOSTS"):
        parse_exact_trusted_hosts(value)

    assert parse_exact_trusted_hosts("play.example.test,127.0.0.1") == (
        "play.example.test",
        "127.0.0.1",
    )


class _BlockingEngine(_DeterministicEngine):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def analyze(self, board: chess.Board, strength: str) -> EngineDecision:
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise RuntimeError("test search release timed out")
        return super().analyze(board, strength)


def test_concurrent_same_game_hint_is_rejected_without_blocking_http_threads() -> None:
    engine = _BlockingEngine()
    application = create_app(engine)
    application.config.update(TESTING=True, SECRET_KEY="test-secret")
    first_client = application.test_client()
    created = first_client.post(
        "/api/v1/games",
        json={"mode": "human", "color": "white", "strength": "quick"},
    )
    game_id = created.get_json()["data"]["id"]
    session_cookie = first_client.get_cookie("session")
    assert session_cookie is not None
    second_client = application.test_client()
    second_client.set_cookie("session", session_cookie.value)
    first_responses = []

    def request_first_hint() -> None:
        first_responses.append(first_client.post(f"/api/v1/games/{game_id}/hint", json={"revision": 0}))

    thread = threading.Thread(target=request_first_hint)
    thread.start()
    assert engine.started.wait(timeout=1.0)
    try:
        rejected = second_client.post(f"/api/v1/games/{game_id}/hint", json={"revision": 0})
        health = application.test_client().get("/api/v1/health")
    finally:
        engine.release.set()
        thread.join(timeout=2.0)

    assert rejected.status_code == 429
    assert rejected.get_json()["error"]["code"] == "game_busy"
    assert rejected.headers["Retry-After"] == "2"
    assert health.status_code == 200
    assert not thread.is_alive()
    assert [response.status_code for response in first_responses] == [200]


def test_hint_rate_limit_is_session_scoped() -> None:
    client = _app().test_client()
    created = client.post("/api/v1/games", json={"mode": "human", "color": "white", "strength": "quick"})
    game_id = created.get_json()["data"]["id"]

    responses = [client.post(f"/api/v1/games/{game_id}/hint", json={"revision": 0}) for _ in range(7)]

    assert [response.status_code for response in responses[:6]] == [200] * 6
    assert responses[6].status_code == 429
    assert responses[6].get_json()["error"]["code"] == "too_many_requests"
