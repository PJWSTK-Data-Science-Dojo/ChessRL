from __future__ import annotations

import chess

from web_app import ApiError, EngineDecision, GameRegistry, LunaEngineService, StrengthProfile, create_app


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


def _app():
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

    played = client.post(f"/api/v1/games/{game_id}/moves", json={"move": "e2e4"})
    assert played.status_code == 200
    payload = played.get_json()
    assert [event["actor"] for event in payload["events"]] == ["human", "luna"]
    assert len(payload["data"]["history"]) == 2
    assert payload["data"]["turn"] == "white"

    undone = client.post(f"/api/v1/games/{game_id}/undo")
    assert undone.status_code == 200
    assert undone.get_json()["data"]["history"] == []


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

    step = client.post(f"/api/v1/games/{game_id}/engine-move")
    assert step.status_code == 200
    assert len(step.get_json()["data"]["history"]) == 1

    invalid = client.post("/api/v1/games", json={"mode": "arcade"})
    assert invalid.status_code == 422
    assert invalid.get_json()["error"]["code"] == "invalid_mode"


def test_missing_model_reports_service_unavailable() -> None:
    application = create_app(None)
    application.config.update(TESTING=True, SECRET_KEY="test-secret")
    response = application.test_client().get("/api/v1/health")

    assert response.status_code == 503
    assert response.get_json()["error"]["code"] == "model_unavailable"
