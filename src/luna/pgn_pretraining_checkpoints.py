"""Checkpoint publication and recovery for PGN pretraining."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from loguru import logger

CHECKPOINT_METADATA_KEY = "pgn_pretraining"
CHECKPOINT_PREFIX = "pretrain_step_"


class CheckpointNetwork(Protocol):
    @property
    def global_step(self) -> int: ...

    def save_checkpoint(
        self,
        folder: str,
        filename: str,
        *,
        extra_state: dict[str, object] | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CheckpointPublication:
    output: Path
    keep: int
    metadata: dict[str, object]


def pretraining_resume_exists(requested: Path, output_dir: Path) -> bool:
    path = requested.expanduser().resolve()
    if path.is_file():
        return True
    target = output_dir.expanduser().resolve()
    return path.name == "latest.pth.tar" and path.parent == target and any(target.glob(f"{CHECKPOINT_PREFIX}*.pth.tar"))


def resolve_pretraining_resume(requested: Path, output_dir: Path) -> Path:
    resolved = requested.expanduser().resolve()
    target = output_dir.expanduser().resolve()
    if resolved.name != "latest.pth.tar" or resolved.parent != target:
        return resolved
    candidates = ([resolved] if resolved.is_file() else []) + sorted(target.glob(f"{CHECKPOINT_PREFIX}*.pth.tar"))
    if not candidates:
        raise FileNotFoundError(f"No resumable pretraining checkpoint in {target}")
    healthy, failures = _healthy_candidates(candidates)
    if not healthy:
        details = "; ".join(f"{path.name}: {message}" for path, message in failures)
        raise RuntimeError(f"No valid pretraining checkpoint in {target}: {details}")
    for path, message in failures:
        logger.warning("Ignoring unreadable pretraining checkpoint {}: {}", path, message)
    return max(healthy, key=lambda candidate: candidate[1])[0]


def _healthy_candidates(candidates: list[Path]) -> tuple[list[tuple[Path, int]], list[tuple[Path, str]]]:
    healthy: list[tuple[Path, int]] = []
    failures: list[tuple[Path, str]] = []
    for candidate in candidates:
        try:
            healthy.append((candidate, _checkpoint_step(candidate)))
        except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
            failures.append((candidate, str(exc)))
    return healthy, failures


def validate_resume_contract(checkpoint: Path, expected: Mapping[str, object]) -> None:
    metadata = _checkpoint_payload(checkpoint).get(CHECKPOINT_METADATA_KEY)
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Resume checkpoint does not match the PGN pretraining contract")


def publish_pretraining_checkpoints(
    network: CheckpointNetwork,
    publication: CheckpointPublication,
) -> None:
    resolved_output = publication.output.expanduser().resolve()
    numbered = resolved_output / f"{CHECKPOINT_PREFIX}{network.global_step:08d}.pth.tar"
    extra_state: dict[str, object] = {CHECKPOINT_METADATA_KEY: publication.metadata}
    if not numbered.exists():
        network.save_checkpoint(str(resolved_output), numbered.name, extra_state=extra_state)
    network.save_checkpoint(str(resolved_output), "latest.pth.tar", extra_state=extra_state)
    _prune_numbered_checkpoints(resolved_output, publication.keep)


def _checkpoint_step(path: Path) -> int:
    payload = _checkpoint_payload(path)
    raw_step = payload.get("global_step")
    if isinstance(raw_step, bool) or not isinstance(raw_step, int) or raw_step < 0:
        raise ValueError(f"Checkpoint has invalid global_step: {path}")
    if path.name.startswith(CHECKPOINT_PREFIX) and raw_step != _numbered_checkpoint_step(path):
        raise ValueError(f"Checkpoint global_step differs from its filename: {path}")
    return raw_step


def _numbered_checkpoint_step(path: Path) -> int:
    suffix = path.name.removeprefix(CHECKPOINT_PREFIX).removesuffix(".pth.tar")
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(f"Invalid pretraining checkpoint filename: {path}") from exc


def _checkpoint_payload(path: Path) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except IndexError as exc:
        raise ValueError(f"Unreadable pretraining checkpoint: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != 2:
        raise ValueError(f"Unsupported pretraining checkpoint: {path}")
    return payload


def _prune_numbered_checkpoints(output: Path, keep: int) -> None:
    checkpoints = sorted(
        output.glob(f"{CHECKPOINT_PREFIX}*.pth.tar"),
        key=_numbered_checkpoint_step,
        reverse=True,
    )
    for checkpoint in checkpoints[keep:]:
        checkpoint.unlink()
