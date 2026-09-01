"""Generate a secure ``lichess-bot`` configuration for the Luna UCI engine."""

from __future__ import annotations

import argparse
import copy
import math
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


class LichessConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class LichessEngineConfig:
    repository: Path
    engine_path: Path
    checkpoint: Path
    device: str
    cuda_device: int | None
    mcts_simulations: int
    minimum_simulations: int
    estimated_simulation_ms: float
    compile_inference: bool

    def resolved(self) -> LichessEngineConfig:
        return replace(
            self,
            repository=self.repository.resolve(),
            engine_path=self.engine_path.resolve(),
            checkpoint=self.checkpoint.resolve(),
        )


def _mapping_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise LichessConfigurationError(
            f"The lichess-bot template is missing the '{name}' mapping. Use an unmodified current config.yml.default."
        )
    return section


def load_template(path: Path) -> dict[str, Any]:
    """Load and minimally validate an upstream ``config.yml.default`` file."""
    if not path.is_file():
        raise LichessConfigurationError(f"Template not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LichessConfigurationError(f"Could not read lichess-bot template: {path}") from exc
    if not isinstance(loaded, dict):
        raise LichessConfigurationError("The lichess-bot template must contain a YAML mapping.")
    for required_section in ("engine", "challenge", "correspondence", "matchmaking"):
        _mapping_section(loaded, required_section)
    return loaded


def build_config(
    template: Mapping[str, Any],
    settings: LichessEngineConfig,
) -> dict[str, Any]:
    settings = settings.resolved()
    _validate_engine_settings(settings)
    config: dict[str, Any] = copy.deepcopy(dict(template))
    config.update(token="", url="https://lichess.org/", move_overhead=250, max_takebacks_accepted=0)
    _configure_engine(_mapping_section(config, "engine"), settings)
    _configure_challenges(_mapping_section(config, "challenge"))
    _mapping_section(config, "correspondence")["ponder"] = False
    matchmaking = _mapping_section(config, "matchmaking")
    matchmaking.update(allow_matchmaking=False, allow_during_games=False)
    _configure_greeting(config)
    return config


def _validate_engine_settings(settings: LichessEngineConfig) -> None:
    if settings.device not in {"cuda", "mps", "cpu"}:
        raise LichessConfigurationError(f"Unsupported inference device: {settings.device}")
    if settings.cuda_device is not None and settings.cuda_device < 0:
        raise LichessConfigurationError("--cuda-device must be non-negative.")
    if settings.mcts_simulations < 1 or settings.minimum_simulations < 1:
        raise LichessConfigurationError("Simulation counts must be positive.")
    if settings.minimum_simulations > settings.mcts_simulations:
        raise LichessConfigurationError("--minimum-sims cannot exceed --mcts-sims.")
    if not math.isfinite(settings.estimated_simulation_ms) or settings.estimated_simulation_ms <= 0:
        raise LichessConfigurationError("--estimated-sim-ms must be finite and positive.")
    _validate_runtime_paths(settings)


def _validate_runtime_paths(settings: LichessEngineConfig) -> None:
    if not settings.repository.is_dir():
        raise LichessConfigurationError(f"Repository directory not found: {settings.repository}")
    if not settings.engine_path.is_file() or not os.access(settings.engine_path, os.X_OK):
        raise LichessConfigurationError(
            f"UCI executable is missing or not executable: {settings.engine_path}. Run 'uv sync' first."
        )
    if not settings.checkpoint.is_file():
        raise LichessConfigurationError(
            f"Checkpoint not found: {settings.checkpoint}. Train or copy a format-v2 latest.pth.tar first."
        )


def _engine_options(settings: LichessEngineConfig) -> dict[str, Any]:
    engine_options: dict[str, Any] = {
        "checkpoint": str(settings.checkpoint),
        "device": settings.device,
        "mcts-sims": settings.mcts_simulations,
        "minimum-sims": settings.minimum_simulations,
        "estimated-sim-ms": settings.estimated_simulation_ms,
    }
    if settings.cuda_device is not None:
        engine_options["cuda-device"] = settings.cuda_device
    if settings.compile_inference:
        engine_options["compile-inference"] = None
    return engine_options


def _configure_engine(engine: dict[str, Any], settings: LichessEngineConfig) -> None:
    engine.update(
        {
            "dir": str(settings.engine_path.parent),
            "name": settings.engine_path.name,
            "debug": False,
            "working_dir": str(settings.repository),
            "protocol": "uci",
            "ponder": False,
            "engine_options": _engine_options(settings),
            "uci_options": {
                "MCTS Simulations": settings.mcts_simulations,
                "Minimum Simulations": settings.minimum_simulations,
                "Estimated Simulation ms": settings.estimated_simulation_ms,
            },
            "silence_stderr": False,
        }
    )
    _disable_optional_engine_features(engine)


def _disable_optional_engine_features(engine: dict[str, Any]) -> None:
    polyglot = engine.get("polyglot")
    if isinstance(polyglot, dict):
        polyglot["enabled"] = False
    draw_or_resign = engine.get("draw_or_resign")
    if isinstance(draw_or_resign, dict):
        draw_or_resign["resign_enabled"] = False
        draw_or_resign["offer_draw_enabled"] = False


def _configure_challenges(challenge: dict[str, Any]) -> None:
    challenge.update(
        {
            "concurrency": 1,
            "games_reserved_for_humans": 0,
            "accept_bot": True,
            "only_bot": False,
            "max_simultaneous_games_per_user": 1,
            "variants": ["standard"],
            "time_controls": ["blitz", "rapid", "classical"],
            "modes": ["casual"],
        }
    )
    challenge.pop("bullet_requires_increment", None)


def _configure_greeting(config: dict[str, Any]) -> None:
    greeting = config.get("greeting")
    if isinstance(greeting, dict):
        greeting.update(
            {
                "hello": "Hi {opponent}! I am Luna, a neural self-play chess engine. Good luck!",
                "goodbye": "Good game — thanks for playing!",
                "hello_spectators": "Welcome! Luna is thinking with neural tree search.",
                "goodbye_spectators": "Thanks for watching!",
            }
        )


def write_private_config(config: Mapping[str, Any], output: Path, *, force: bool = False) -> None:
    """Atomically write YAML with owner-only permissions."""
    output = output.expanduser().absolute()
    if output.is_symlink():
        raise LichessConfigurationError(f"Refusing to write configuration through a symbolic link: {output}")
    if output.exists() and not force:
        raise LichessConfigurationError(f"Refusing to overwrite existing config: {output}. Pass --force to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(dict(config), stream, sort_keys=False, default_flow_style=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output)
        output.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise LichessConfigurationError(f"Could not write private configuration: {output}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _parser(repository: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a private config.yml for the upstream lichess-bot bridge.")
    parser.add_argument(
        "--lichess-bot-dir",
        type=Path,
        default=repository.parent / "lichess-bot",
        help="Path to a clone of lichess-bot (default: a sibling of this repository).",
    )
    parser.add_argument("--template", type=Path, help="Override the upstream config.yml.default path.")
    parser.add_argument("--output", type=Path, help="Override the generated config.yml path.")
    parser.add_argument("--engine", type=Path, default=repository / ".venv/bin/luna-uci")
    parser.add_argument("--checkpoint", type=Path, default=repository / "runs/luna-main/latest.pth.tar")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--mcts-sims", type=int, default=100)
    parser.add_argument("--minimum-sims", type=int, default=8)
    parser.add_argument("--estimated-sim-ms", type=float, default=4.0)
    parser.add_argument("--compile-inference", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a credential-free config for an environment-authenticated bridge."""
    repository = Path(__file__).resolve().parents[2]
    args = _parser(repository).parse_args(argv)
    bot_directory = args.lichess_bot_dir.resolve()
    template_path = (args.template or bot_directory / "config.yml.default").resolve()
    output_path = (args.output or bot_directory / "config.yml").expanduser().absolute()
    try:
        template = load_template(template_path)
        settings = LichessEngineConfig(
            repository=repository,
            engine_path=args.engine,
            checkpoint=args.checkpoint,
            device=args.device,
            cuda_device=args.cuda_device,
            mcts_simulations=args.mcts_sims,
            minimum_simulations=args.minimum_sims,
            estimated_simulation_ms=args.estimated_sim_ms,
            compile_inference=args.compile_inference,
        )
        config = build_config(template, settings)
        write_private_config(config, output_path, force=args.force)
    except LichessConfigurationError as exc:
        print(f"Configuration failed: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote private lichess-bot configuration to {output_path}")
    print(f"Engine: {args.engine.resolve()}")
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print("Credential: provide LICHESS_BOT_TOKEN only to the running lichess-bot process")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
