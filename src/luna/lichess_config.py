"""Generate a secure ``lichess-bot`` configuration for the Luna UCI engine."""

from __future__ import annotations

import argparse
import copy
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


class LichessConfigurationError(ValueError):
    """Raised when a safe lichess-bot configuration cannot be generated."""


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
    *,
    repository: Path,
    engine_path: Path,
    checkpoint: Path,
    device: str,
    cuda_device: int | None,
    mcts_simulations: int,
    minimum_simulations: int,
    estimated_simulation_ms: float,
    compile_inference: bool,
) -> dict[str, Any]:
    """Return a configured copy of the current lichess-bot template."""
    if device not in {"cuda", "mps", "cpu"}:
        raise LichessConfigurationError(f"Unsupported inference device: {device}")
    if cuda_device is not None and cuda_device < 0:
        raise LichessConfigurationError("--cuda-device must be non-negative.")
    if mcts_simulations < 1 or minimum_simulations < 1:
        raise LichessConfigurationError("Simulation counts must be positive.")
    if minimum_simulations > mcts_simulations:
        raise LichessConfigurationError("--minimum-sims cannot exceed --mcts-sims.")
    if estimated_simulation_ms <= 0:
        raise LichessConfigurationError("--estimated-sim-ms must be positive.")

    repository = repository.resolve()
    engine_path = engine_path.resolve()
    checkpoint = checkpoint.resolve()
    if not engine_path.is_file() or not os.access(engine_path, os.X_OK):
        raise LichessConfigurationError(
            f"UCI executable is missing or not executable: {engine_path}. Run 'uv sync' first."
        )
    if not checkpoint.is_file():
        raise LichessConfigurationError(
            f"Checkpoint not found: {checkpoint}. Train or copy a format-v2 latest.pth.tar first."
        )

    config: dict[str, Any] = copy.deepcopy(dict(template))
    engine = _mapping_section(config, "engine")
    challenge = _mapping_section(config, "challenge")
    correspondence = _mapping_section(config, "correspondence")
    matchmaking = _mapping_section(config, "matchmaking")

    # Current lichess-bot releases read LICHESS_BOT_TOKEN before this field.
    # Keeping the generated file secret-free prevents accidental credential copies.
    config["token"] = ""
    config["url"] = "https://lichess.org/"
    config["move_overhead"] = 250
    config["max_takebacks_accepted"] = 0

    engine_options: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "device": device,
        "mcts-sims": mcts_simulations,
        "minimum-sims": minimum_simulations,
        "estimated-sim-ms": estimated_simulation_ms,
    }
    if cuda_device is not None:
        engine_options["cuda-device"] = cuda_device
    if compile_inference:
        # lichess-bot renders a null engine option as a valueless CLI switch.
        engine_options["compile-inference"] = None

    engine.update(
        {
            "dir": str(engine_path.parent),
            "name": engine_path.name,
            "debug": False,
            "working_dir": str(repository),
            "protocol": "uci",
            "ponder": False,
            "engine_options": engine_options,
            "uci_options": {
                "MCTS Simulations": mcts_simulations,
                "Minimum Simulations": minimum_simulations,
                "Estimated Simulation ms": estimated_simulation_ms,
            },
            "silence_stderr": False,
        }
    )
    polyglot = engine.get("polyglot")
    if isinstance(polyglot, dict):
        polyglot["enabled"] = False
    draw_or_resign = engine.get("draw_or_resign")
    if isinstance(draw_or_resign, dict):
        draw_or_resign["resign_enabled"] = False
        draw_or_resign["offer_draw_enabled"] = False

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
    correspondence["ponder"] = False
    matchmaking["allow_matchmaking"] = False
    matchmaking["allow_during_games"] = False

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
    return config


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
        config = build_config(
            template,
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
