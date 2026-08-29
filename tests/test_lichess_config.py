from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import yaml

from luna.lichess_config import build_config, main, write_private_config


def _template() -> dict[str, Any]:
    return {
        "token": "placeholder",
        "url": "https://lichess.org/",
        "engine": {
            "dir": "./engines",
            "name": "example",
            "protocol": "uci",
            "ponder": True,
            "polyglot": {"enabled": True},
            "draw_or_resign": {"resign_enabled": True, "offer_draw_enabled": True},
            "uci_options": {"Hash": 512},
        },
        "challenge": {
            "concurrency": 8,
            "variants": ["standard", "chess960"],
            "time_controls": ["bullet"],
        },
        "correspondence": {"ponder": True},
        "matchmaking": {"allow_matchmaking": True, "allow_during_games": True},
        "greeting": {"hello": "old"},
        "future_upstream_key": {"preserved": True},
    }


def _runtime_files(tmp_path: Path) -> tuple[Path, Path]:
    engine = tmp_path / "venv" / "bin" / "luna-uci"
    engine.parent.mkdir(parents=True)
    engine.write_text("#!/bin/sh\n", encoding="utf-8")
    engine.chmod(0o700)
    checkpoint = tmp_path / "latest.pth.tar"
    checkpoint.write_bytes(b"checkpoint")
    return engine, checkpoint


def test_build_config_uses_current_lichess_bot_shape_and_safe_defaults(tmp_path: Path) -> None:
    engine, checkpoint = _runtime_files(tmp_path)
    config = build_config(
        _template(),
        token="secret-token",
        repository=tmp_path,
        engine_path=engine,
        checkpoint=checkpoint,
        device="cuda",
        cuda_device=0,
        mcts_simulations=96,
        minimum_simulations=12,
        estimated_simulation_ms=3.5,
        compile_inference=True,
    )

    engine_config = config["engine"]
    assert Path(engine_config["dir"]) / engine_config["name"] == engine.resolve()
    assert engine_config["working_dir"] == str(tmp_path.resolve())
    assert engine_config["protocol"] == "uci"
    assert engine_config["ponder"] is False
    assert engine_config["engine_options"] == {
        "checkpoint": str(checkpoint.resolve()),
        "device": "cuda",
        "mcts-sims": 96,
        "minimum-sims": 12,
        "estimated-sim-ms": 3.5,
        "cuda-device": 0,
        "compile-inference": None,
    }
    assert engine_config["uci_options"] == {
        "MCTS Simulations": 96,
        "Minimum Simulations": 12,
        "Estimated Simulation ms": 3.5,
    }
    assert engine_config["polyglot"]["enabled"] is False
    assert engine_config["draw_or_resign"]["resign_enabled"] is False
    assert engine_config["draw_or_resign"]["offer_draw_enabled"] is False

    challenge = config["challenge"]
    assert challenge["concurrency"] == 1
    assert challenge["variants"] == ["standard"]
    assert challenge["time_controls"] == ["blitz", "rapid", "classical"]
    assert "bullet_requires_increment" not in challenge
    assert challenge["max_simultaneous_games_per_user"] == 1
    assert config["correspondence"]["ponder"] is False
    assert config["matchmaking"]["allow_matchmaking"] is False
    assert config["future_upstream_key"] == {"preserved": True}


def test_private_writer_is_atomic_owner_only_and_requires_force(tmp_path: Path) -> None:
    output = tmp_path / "config.yml"
    write_private_config({"token": "first"}, output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == {"token": "first"}

    try:
        write_private_config({"token": "second"}, output)
    except ValueError as exc:
        assert "--force" in str(exc)
    else:
        raise AssertionError("existing private configuration was overwritten without --force")

    write_private_config({"token": "second"}, output, force=True)
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == {"token": "second"}
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_private_writer_never_follows_a_symbolic_link(tmp_path: Path) -> None:
    protected = tmp_path / "protected.yml"
    protected.write_text("owner: user\n", encoding="utf-8")
    output = tmp_path / "config.yml"
    output.symlink_to(protected)

    try:
        write_private_config({"token": "secret"}, output, force=True)
    except ValueError as exc:
        assert "symbolic link" in str(exc)
    else:
        raise AssertionError("private configuration followed a symbolic link")

    assert protected.read_text(encoding="utf-8") == "owner: user\n"


def test_cli_reads_token_only_from_environment_without_printing_it(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    engine, checkpoint = _runtime_files(tmp_path)
    template = tmp_path / "config.yml.default"
    template.write_text(yaml.safe_dump(_template()), encoding="utf-8")
    output = tmp_path / "config.yml"
    token = "do-not-log-this-token"
    monkeypatch.setenv("LICHESS_TOKEN", token)

    result = main(
        [
            "--template",
            str(template),
            "--output",
            str(output),
            "--engine",
            str(engine),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--mcts-sims",
            "32",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert token not in captured.out
    assert token not in captured.err
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["token"] == token
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_cli_refuses_to_write_without_environment_token(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    engine, checkpoint = _runtime_files(tmp_path)
    template = tmp_path / "config.yml.default"
    template.write_text(yaml.safe_dump(_template()), encoding="utf-8")
    output = tmp_path / "config.yml"
    monkeypatch.delenv("LICHESS_TOKEN", raising=False)

    result = main(
        [
            "--template",
            str(template),
            "--output",
            str(output),
            "--engine",
            str(engine),
            "--checkpoint",
            str(checkpoint),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "LICHESS_TOKEN" in captured.err
    assert not output.exists()
