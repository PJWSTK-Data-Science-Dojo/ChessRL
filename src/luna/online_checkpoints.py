"""Crash-safe discovery and recovery of online training checkpoints."""

from __future__ import annotations

import filecmp
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from luna.coach_checkpoints import atomic_copy
from luna.game.chess_game import ChessGame
from luna.network import LunaNetwork
from luna.network_types import TrainingPhaseProvenance

_BOOTSTRAP_TEMP_PREFIX = "checkpoint_0.pth.tar.tmp-"
_CHECKPOINT_READ_ERRORS = (
    EOFError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    pickle.UnpicklingError,
)


@dataclass(frozen=True, slots=True)
class _CheckpointIdentity:
    iteration: int
    provenance: TrainingPhaseProvenance | None


@dataclass(frozen=True, slots=True)
class _CheckpointCandidate:
    identity: _CheckpointIdentity
    path: Path
    numbered: bool


@dataclass(frozen=True, slots=True)
class _CandidateScan:
    healthy: tuple[_CheckpointCandidate, ...]
    failures: tuple[tuple[Path, str], ...]


def _is_bootstrap_atomic_temporary(path: Path) -> bool:
    suffix = path.name.removeprefix(_BOOTSTRAP_TEMP_PREFIX)
    return (
        path.name.startswith(_BOOTSTRAP_TEMP_PREFIX)
        and suffix.isdecimal()
        and int(suffix) > 0
        and path.is_file()
        and not path.is_symlink()
    )


def validate_new_training_phase_target(checkpoint_dir: str) -> None:
    """Require a dedicated directory, tolerating only interrupted bootstrap writes."""
    if not checkpoint_dir.strip():
        raise ValueError("new_training_phase requires a non-empty --run.checkpoint directory")
    target = Path(checkpoint_dir).expanduser().resolve()
    if not target.exists():
        return
    if not target.is_dir():
        raise FileExistsError(f"New training phase target is not a directory: {target}")
    contents = sorted(target.iterdir())
    conflicts = [path.name for path in contents if not _is_bootstrap_atomic_temporary(path)]
    if conflicts:
        raise FileExistsError(
            f"New training phase requires an empty checkpoint directory, but {target} contains {conflicts}. "
            "Choose a new --run.checkpoint directory."
        )
    for path in contents:
        logger.warning("Preserving stale online-bootstrap temporary file {}", path)


def _numbered_checkpoint_iteration(path: Path) -> int:
    suffix = path.name.removeprefix("checkpoint_").removesuffix(".pth.tar")
    try:
        iteration = int(suffix)
    except ValueError as exc:
        raise ValueError(f"Invalid numbered checkpoint name: {path}") from exc
    if iteration < 0:
        raise ValueError(f"Invalid numbered checkpoint name: {path}")
    return iteration


def _validated_checkpoint_identity(path: Path) -> _CheckpointIdentity:
    network = LunaNetwork.from_checkpoint(ChessGame(), path, device="cpu", load_optimizer=True)
    return _CheckpointIdentity(network.trainer_iteration, network.training_phase_provenance)


def _checkpoint_candidate(path: Path) -> _CheckpointCandidate:
    identity = _validated_checkpoint_identity(path)
    numbered = path.name != "latest.pth.tar"
    if numbered and identity.iteration != _numbered_checkpoint_iteration(path):
        raise ValueError(f"Numbered checkpoint iteration {identity.iteration} differs from its filename: {path}")
    return _CheckpointCandidate(identity, path, numbered)


def _scan_candidates(paths: list[Path]) -> _CandidateScan:
    healthy: list[_CheckpointCandidate] = []
    failures: list[tuple[Path, str]] = []
    for path in paths:
        try:
            healthy.append(_checkpoint_candidate(path))
        except _CHECKPOINT_READ_ERRORS as exc:
            failures.append((path, str(exc)))
            logger.warning("Ignoring unreadable online checkpoint {}: {}", path, exc)
    return _CandidateScan(tuple(healthy), tuple(failures))


def _authoritative_candidates(scan: _CandidateScan) -> tuple[_CheckpointCandidate, ...]:
    numbered = tuple(candidate for candidate in scan.healthy if candidate.numbered)
    if not numbered:
        return scan.healthy
    provenance = numbered[0].identity.provenance
    conflicts = [candidate.path.name for candidate in numbered if candidate.identity.provenance != provenance]
    if conflicts:
        raise RuntimeError(f"Healthy numbered checkpoints contain mixed training-phase lineage: {conflicts}")
    for candidate in scan.healthy:
        if not candidate.numbered and candidate.identity.provenance != provenance:
            logger.warning("Ignoring latest checkpoint from a different training-phase lineage: {}", candidate.path)
    return numbered


def _quarantine_destination(path: Path) -> Path:
    destination = path.with_name(f"{path.name}.invalid")
    collision = 1
    while destination.exists() or destination.is_symlink():
        destination = path.with_name(f"{path.name}.invalid-{collision}")
        collision += 1
    return destination


def _fsync_directory(folder: Path) -> None:
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _quarantine_invalid_numbered(failures: tuple[tuple[Path, str], ...]) -> None:
    for path, reason in failures:
        if path.name == "latest.pth.tar":
            continue
        destination = _quarantine_destination(path)
        os.replace(path, destination)
        _fsync_directory(path.parent)
        logger.warning("Quarantined invalid checkpoint {} as {}: {}", path, destination, reason)


def _select_candidate(scan: _CandidateScan) -> _CheckpointCandidate:
    if not scan.healthy:
        details = "; ".join(f"{path.name}: {message}" for path, message in scan.failures)
        raise RuntimeError(f"No healthy resumable checkpoint found: {details}")
    candidates = _authoritative_candidates(scan)
    return max(candidates, key=lambda candidate: (candidate.identity.iteration, candidate.numbered))


def _heal_latest(selected: Path, latest: Path) -> None:
    if not latest.is_file() or not filecmp.cmp(selected, latest, shallow=False):
        atomic_copy(selected, latest)


def resolve_resume_checkpoint(requested: Path, target: Path) -> Path:
    """Select a healthy immutable checkpoint and reconcile the mutable alias."""
    resolved = requested.expanduser().resolve()
    if resolved.name != "latest.pth.tar" or resolved.parent != target.expanduser().resolve():
        return resolved
    paths = ([resolved] if resolved.is_file() else []) + sorted(resolved.parent.glob("checkpoint_*.pth.tar"))
    if not paths:
        raise FileNotFoundError(f"No resumable checkpoint in {resolved.parent}")
    scan = _scan_candidates(paths)
    selected = _select_candidate(scan)
    _quarantine_invalid_numbered(scan.failures)
    if selected.path != resolved:
        logger.warning('Recovering from immutable checkpoint "{}" instead of "{}"', selected.path, resolved)
        _heal_latest(selected.path, resolved)
    return selected.path
