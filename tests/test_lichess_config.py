from __future__ import annotations

import stat
from pathlib import Path

import pytest
import yaml

from luna.lichess_config import build_config, main, write_private_config

type YamlValue = str | int | float | bool | None | list[YamlValue] | dict[str, YamlValue]


def _template() -> dict[str, YamlValue]:
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
    assert config["token"] == ""

    challenge = config["challenge"]
    assert challenge["concurrency"] == 1
    assert challenge["variants"] == ["standard"]
    assert challenge["time_controls"] == ["blitz", "rapid", "classical"]
    assert challenge["modes"] == ["casual"]
    assert "bullet_requires_increment" not in challenge
    assert challenge["max_simultaneous_games_per_user"] == 1
    assert config["correspondence"]["ponder"] is False
    assert config["matchmaking"]["allow_matchmaking"] is False
    assert config["future_upstream_key"] == {"preserved": True}


def test_build_config_rejects_non_finite_timing(tmp_path: Path) -> None:
    engine, checkpoint = _runtime_files(tmp_path)

    with pytest.raises(ValueError, match="finite and positive"):
        build_config(
            _template(),
            repository=tmp_path,
            engine_path=engine,
            checkpoint=checkpoint,
            device="cpu",
            cuda_device=None,
            mcts_simulations=32,
            minimum_simulations=8,
            estimated_simulation_ms=float("nan"),
            compile_inference=False,
        )


def test_private_writer_is_atomic_owner_only_and_requires_force(tmp_path: Path) -> None:
    output = tmp_path / "config.yml"
    write_private_config({"setting": "first"}, output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == {"setting": "first"}

    try:
        write_private_config({"setting": "second"}, output)
    except ValueError as exc:
        assert "--force" in str(exc)
    else:
        raise AssertionError("existing private configuration was overwritten without --force")

    write_private_config({"setting": "second"}, output, force=True)
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == {"setting": "second"}
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_private_writer_never_follows_a_symbolic_link(tmp_path: Path) -> None:
    protected = tmp_path / "protected.yml"
    protected.write_text("owner: user\n", encoding="utf-8")
    output = tmp_path / "config.yml"
    output.symlink_to(protected)

    try:
        write_private_config({"setting": "replacement"}, output, force=True)
    except ValueError as exc:
        assert "symbolic link" in str(exc)
    else:
        raise AssertionError("private configuration followed a symbolic link")

    assert protected.read_text(encoding="utf-8") == "owner: user\n"


def test_cli_never_serializes_or_prints_environment_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine, checkpoint = _runtime_files(tmp_path)
    template = tmp_path / "config.yml.default"
    template.write_text(yaml.safe_dump(_template()), encoding="utf-8")
    output = tmp_path / "config.yml"
    token = "do-not-log-this-token"
    monkeypatch.setenv("LICHESS_BOT_TOKEN", token)

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
    assert token not in output.read_text(encoding="utf-8")
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["token"] == ""
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_cli_writes_credential_free_config_without_environment_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine, checkpoint = _runtime_files(tmp_path)
    template = tmp_path / "config.yml.default"
    template.write_text(yaml.safe_dump(_template()), encoding="utf-8")
    output = tmp_path / "config.yml"
    monkeypatch.delenv("LICHESS_BOT_TOKEN", raising=False)

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
    assert result == 0
    assert "LICHESS_BOT_TOKEN" in captured.out
    assert captured.err == ""
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["token"] == ""
