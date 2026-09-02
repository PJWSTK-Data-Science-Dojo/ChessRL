"""Checkpoint lineage, publication, retention, and best-score metadata."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

from loguru import logger

from luna.config import TrainingRunConfig
from luna.game.benchmark_state import BENCHMARK_STATE_NAME
from luna.game.stockfish_eval import (
    StockfishEvalOutcome,
    StockfishEvalScores,
    StockfishEvalSkipped,
    stockfish_evaluation_protocol,
)
from luna.game.stockfish_ladder import LADDER_STATE_NAME
from luna.network import LunaNetwork
from luna.replay_persistence import REPLAY_SNAPSHOT_NAME

if TYPE_CHECKING:
    from luna.coach import Coach

_BEST_EVAL_NAME = "best_eval.json"
_BEST_EVAL_FIELD = "best_evaluation"
_BEST_EVAL_SCHEMA_VERSION = 1


def _managed_checkpoint_conflicts(folder: Path) -> list[str]:
    managed = list(folder.glob("checkpoint_*.pth.tar"))
    managed.extend(
        folder / name
        for name in (
            "latest.pth.tar",
            "best.pth.tar",
            _BEST_EVAL_NAME,
            BENCHMARK_STATE_NAME,
            LADDER_STATE_NAME,
            REPLAY_SNAPSHOT_NAME,
        )
    )
    return sorted(path.name for path in managed if path.exists())


def validate_fresh_checkpoint_target(run: TrainingRunConfig) -> None:
    """Refuse to start a new run in a directory containing managed training state."""
    if not str(run.checkpoint).strip():
        return
    folder = Path(run.checkpoint).resolve()
    conflicts = _managed_checkpoint_conflicts(folder)
    if conflicts:
        raise FileExistsError(
            f"Fresh training would overwrite managed files in {folder}: {conflicts}. "
            "Choose a new --run.checkpoint directory or resume latest.pth.tar."
        )


def validate_resume_checkpoint_target(
    run: TrainingRunConfig,
    source_checkpoint: str | Path,
    *,
    allow_evaluation_artifacts_only: bool = False,
) -> None:
    """Prevent a resume checkpoint from being merged into another managed run."""
    if not str(run.checkpoint).strip():
        return
    target = Path(run.checkpoint).resolve()
    if Path(source_checkpoint).expanduser().resolve().parent == target:
        return
    conflicts = _managed_checkpoint_conflicts(target)
    evaluation_artifacts = {BENCHMARK_STATE_NAME, LADDER_STATE_NAME, "best.pth.tar", _BEST_EVAL_NAME}
    if allow_evaluation_artifacts_only and set(conflicts) <= evaluation_artifacts:
        return
    if conflicts:
        raise FileExistsError(
            f"Resume target {target} contains managed files from another checkpoint lineage: {conflicts}. "
            "Resume in the source directory or choose a new, empty --run.checkpoint directory."
        )


def stockfish_normalized_score(scores: StockfishEvalScores) -> float:
    """Map a Stockfish matchup to a draw-aware score in the unit interval."""
    total = scores.model_wins + scores.draws + scores.stockfish_wins
    if total <= 0:
        raise ValueError("A completed Stockfish evaluation must contain at least one game")
    return (scores.model_wins + 0.5 * scores.draws) / float(total)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_best_record(
    value: object,
    *,
    protocol: dict[str, object],
    best_path: Path,
    trainer_iteration: int,
) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"Best checkpoint has no valid external-evaluation record: {best_path}")
    record = dict(value)
    iteration = record.get("iteration")
    score = record.get("score")
    source_sha256 = record.get("source_checkpoint_sha256")
    if record.get("schema_version") != _BEST_EVAL_SCHEMA_VERSION:
        raise RuntimeError(f"Best checkpoint has an unsupported external-evaluation record: {best_path}")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise RuntimeError(f"Best checkpoint evaluation iteration is invalid: {best_path}")
    if iteration != trainer_iteration:
        raise RuntimeError(f"Best checkpoint evaluation iteration differs from its trainer state: {best_path}")
    if (
        isinstance(score, bool)
        or not isinstance(score, int | float)
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
    ):
        raise RuntimeError(f"Best checkpoint evaluation score must be finite and between zero and one: {best_path}")
    if record.get("protocol") != protocol:
        raise RuntimeError(
            f"External-evaluation protocol differs from the score in {best_path}; "
            "use a new checkpoint directory for this benchmark contract"
        )
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise RuntimeError(f"Best checkpoint source SHA-256 is invalid: {best_path}")
    return record


def write_best_metadata(metadata_path: Path, record: dict[str, object]) -> None:
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, metadata_path)
        directory_fd = os.open(metadata_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def best_evaluation_record(folder: Path, protocol: dict[str, object]) -> dict[str, object] | None:
    best_path = folder / "best.pth.tar"
    metadata_path = folder / _BEST_EVAL_NAME
    if not best_path.exists():
        if metadata_path.exists():
            raise RuntimeError(f"External-evaluation metadata exists without its best checkpoint: {metadata_path}")
        return None
    checkpoint = LunaNetwork._read_checkpoint(best_path)
    trainer_iteration = checkpoint.get("trainer_iteration")
    if isinstance(trainer_iteration, bool) or not isinstance(trainer_iteration, int):
        raise RuntimeError(f"Best checkpoint trainer iteration is invalid: {best_path}")
    record = validate_best_record(
        checkpoint.get(_BEST_EVAL_FIELD),
        protocol=protocol,
        best_path=best_path,
        trainer_iteration=trainer_iteration,
    )
    try:
        metadata: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        metadata = None
    if metadata != record:
        write_best_metadata(metadata_path, record)
    return record


def previous_best_score(folder: Path, protocol: dict[str, object]) -> float:
    record = best_evaluation_record(folder, protocol)
    if record is None:
        return float("-inf")
    return float(cast(float | int, record["score"]))


def validate_best_evaluation_contract(run: TrainingRunConfig) -> dict[str, object] | None:
    """Validate and reconcile the authoritative best-checkpoint record before training."""
    if not str(run.checkpoint).strip():
        return None
    folder = Path(run.checkpoint).expanduser().resolve()
    protocol = asdict(stockfish_evaluation_protocol(run))
    return best_evaluation_record(folder, protocol)


def checkpoint_dir_usable(coach: Coach) -> bool:
    return bool(str(coach.run.checkpoint).strip())


def assert_checkpoint_target(coach: Coach) -> None:
    if coach._checkpoint_target_validated or not checkpoint_dir_usable(coach):
        return
    source_checkpoint = coach.nnet._loaded_checkpoint_path
    if source_checkpoint is None:
        validate_fresh_checkpoint_target(coach.run)
    else:
        validate_resume_checkpoint_target(
            coach.run,
            source_checkpoint,
            allow_evaluation_artifacts_only=coach._initialize_evaluation_state,
        )
    coach._checkpoint_target_validated = True


def numbered_checkpoints(folder: Path) -> list[tuple[int, Path]]:
    numbered: list[tuple[int, Path]] = []
    for path in folder.glob("checkpoint_*.pth.tar"):
        try:
            iteration = int(path.name.removeprefix("checkpoint_").removesuffix(".pth.tar"))
        except ValueError:
            logger.warning("Ignoring checkpoint with an invalid iteration name: {}", path)
            continue
        numbered.append((iteration, path))
    return numbered


def assert_checkpoint_lineage(coach: Coach) -> None:
    if not checkpoint_dir_usable(coach):
        return
    managed_iteration = managed_checkpoint_iteration(coach, Path(coach.run.checkpoint).resolve())
    coach._checkpoint_lineage_iteration = managed_iteration
    if managed_iteration > coach.nnet._trainer_iteration:
        raise RuntimeError(
            "The checkpoint directory contains newer training state than the loaded checkpoint; "
            "load its highest-iteration managed checkpoint or resume into a new directory"
        )


def managed_checkpoint_iteration(coach: Coach, folder: Path) -> int:
    numbered_iteration = max((iteration for iteration, _path in numbered_checkpoints(folder)), default=0)
    latest_path = folder / "latest.pth.tar"
    if not latest_path.exists():
        return numbered_iteration
    latest_iteration = LunaNetwork.checkpoint_trainer_iteration(latest_path)
    return max(numbered_iteration, latest_iteration)


def prune_checkpoint_files(coach: Coach) -> None:
    top_k = coach.run.checkpoint_top_k
    if top_k is None or top_k <= 0 or not checkpoint_dir_usable(coach):
        return
    folder = Path(coach.run.checkpoint).resolve()
    numbered = [path for _, path in sorted(numbered_checkpoints(folder), reverse=True)]
    for path in numbered[max(1, int(top_k)) :]:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove old checkpoint {}: {}", path, exc)


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def publish_bootstrap_checkpoint(network: LunaNetwork, checkpoint_dir: str) -> Path:
    """Durably seed a new online lineage before its first self-play iteration."""
    if not checkpoint_dir.strip():
        raise ValueError("Online bootstrap requires a checkpoint directory")
    if network.global_step != 0 or network.trainer_iteration != 0:
        raise ValueError("Online bootstrap requires zero training counters")
    folder = Path(checkpoint_dir).expanduser().resolve()
    conflicts = _managed_checkpoint_conflicts(folder)
    if conflicts:
        raise FileExistsError(f"Online bootstrap target already contains managed files: {conflicts}")
    checkpoint = folder / "checkpoint_0.pth.tar"
    network.save_checkpoint(str(folder), checkpoint.name)
    atomic_copy(checkpoint, folder / "latest.pth.tar")
    network._loaded_checkpoint_path = checkpoint
    logger.info("Published online bootstrap checkpoint {}", checkpoint)
    return checkpoint


def update_best_from_stockfish(
    coach: Coach,
    iteration: int,
    outcome: StockfishEvalOutcome,
    *,
    checkpoint_path: Path | None = None,
) -> None:
    if isinstance(outcome, StockfishEvalSkipped):
        raise RuntimeError(f"External evaluation did not complete ({outcome.reason}): {outcome.message}")
    if not checkpoint_dir_usable(coach):
        return
    folder = Path(coach.run.checkpoint).resolve()
    source = checkpoint_path if checkpoint_path is not None else folder / f"checkpoint_{iteration}.pth.tar"
    if not source.is_file():
        raise FileNotFoundError(f"Evaluated checkpoint is missing: {source}")
    checkpoint_iteration = LunaNetwork.checkpoint_trainer_iteration(source)
    if checkpoint_iteration != iteration:
        raise RuntimeError(
            f"Evaluated checkpoint iteration {checkpoint_iteration} differs from requested iteration {iteration}: {source}"
        )
    if coach.nnet._trainer_iteration != iteration:
        raise RuntimeError("In-memory model differs from the externally evaluated checkpoint iteration")
    score = stockfish_normalized_score(outcome)
    protocol = asdict(stockfish_evaluation_protocol(coach.run))
    if score <= previous_best_score(folder, protocol):
        return
    record: dict[str, object] = {
        "schema_version": _BEST_EVAL_SCHEMA_VERSION,
        "iteration": iteration,
        "score": score,
        "protocol": protocol,
        "source_checkpoint_sha256": checkpoint_sha256(source),
    }
    coach.nnet.save_checkpoint(
        folder=coach.run.checkpoint,
        filename="best.pth.tar",
        extra_state={_BEST_EVAL_FIELD: record},
    )
    write_best_metadata(folder / _BEST_EVAL_NAME, record)
    logger.info("New best external score {:.3f} at iteration {}", score, iteration)


def publish_checkpoint(coach: Coach, iteration: int) -> None:
    if not checkpoint_dir_usable(coach):
        logger.warning(
            'run.checkpoint "" or unset-like; skipping checkpoint_{} and best.pth.tar writes.',
            iteration,
        )
        return
    checkpoint_name = f"checkpoint_{iteration}.pth.tar"
    folder = Path(coach.run.checkpoint).resolve()
    checkpoint_path = folder / checkpoint_name
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable numbered checkpoint: {checkpoint_path}")
    if coach._checkpoint_lineage_iteration is None:
        coach._checkpoint_lineage_iteration = managed_checkpoint_iteration(coach, folder)
    numbered_iteration = max(
        (saved_iteration for saved_iteration, _path in numbered_checkpoints(folder)),
        default=0,
    )
    latest_existing = max(coach._checkpoint_lineage_iteration, numbered_iteration)
    if latest_existing >= iteration:
        raise FileExistsError(
            f"Refusing non-monotonic checkpoint {checkpoint_path}; "
            f"directory already contains iteration {latest_existing}"
        )
    previous_iteration = coach.nnet._trainer_iteration
    coach.nnet._trainer_iteration = iteration
    numbered_saved = False
    try:
        coach.nnet.save_checkpoint(folder=coach.run.checkpoint, filename=checkpoint_name)
        numbered_saved = True
    finally:
        if not numbered_saved:
            coach.nnet._trainer_iteration = previous_iteration
    atomic_copy(folder / checkpoint_name, folder / "latest.pth.tar")
    coach._checkpoint_lineage_iteration = iteration
    prune_checkpoint_files(coach)
