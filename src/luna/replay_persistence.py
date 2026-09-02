"""Durable replay snapshots stored independently from model checkpoints."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import BinaryIO

import zstandard

from luna.replay_buffer import PrioritizedReplayBuffer, ReplaySnapshot

REPLAY_SNAPSHOT_NAME = "replay_buffer.pkl.zst"
_COMPRESSION_LEVEL = 1
_ALLOWED_PICKLE_GLOBALS = {
    ("chess", "Termination"),
    ("luna.replay_buffer", "ReplaySnapshot"),
    ("luna.replay_buffer", "Trajectory"),
    ("numpy", "dtype"),
    ("numpy._core.numeric", "_frombuffer"),
}


class ReplaySnapshotError(RuntimeError):
    """A replay snapshot cannot be decoded or violates its state contract."""


def save_replay_snapshot(
    replay: PrioritizedReplayBuffer,
    checkpoint_dir: str | Path,
    trainer_iteration: int,
) -> Path:
    """Compress and atomically publish the replay state for an iteration."""
    folder = Path(checkpoint_dir).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / REPLAY_SNAPSHOT_NAME
    snapshot = replay.snapshot(trainer_iteration)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.tmp-", dir=folder)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw_stream:
            _write_snapshot(raw_stream, snapshot)
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(folder)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_replay_snapshot(
    replay: PrioritizedReplayBuffer,
    checkpoint_dir: str | Path,
    expected_iteration: int,
) -> int:
    """Load, validate, and install a replay snapshot without partial mutation."""
    path = Path(checkpoint_dir).expanduser().resolve() / REPLAY_SNAPSHOT_NAME
    try:
        with path.open("rb") as raw_stream:
            snapshot = _read_snapshot(raw_stream)
        return replay.restore(snapshot, expected_iteration)
    except FileNotFoundError:
        raise
    except (EOFError, OSError, TypeError, ValueError, pickle.PickleError, zstandard.ZstdError) as exc:
        raise ReplaySnapshotError(f"Invalid replay snapshot {path}: {exc}") from exc


def _write_snapshot(raw_stream: BinaryIO, snapshot: ReplaySnapshot) -> None:
    compressor = zstandard.ZstdCompressor(level=_COMPRESSION_LEVEL, write_checksum=True)
    with compressor.stream_writer(raw_stream, closefd=False) as compressed_stream:
        pickle.dump(snapshot, compressed_stream, protocol=pickle.HIGHEST_PROTOCOL)


def _read_snapshot(raw_stream: BinaryIO) -> ReplaySnapshot:
    decompressor = zstandard.ZstdDecompressor()
    with decompressor.stream_reader(raw_stream, closefd=False) as compressed_stream:
        payload: object = _ReplayUnpickler(compressed_stream).load()
        if compressed_stream.read(1):
            raise ValueError("Replay snapshot contains trailing data")
    if not isinstance(payload, ReplaySnapshot):
        raise TypeError("Replay snapshot has an unsupported payload type")
    return payload


class _ReplayUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        if (module, name) not in _ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"Replay snapshot references forbidden global {module}.{name}")
        resolved: object = super().find_class(module, name)
        return resolved


def _fsync_directory(folder: Path) -> None:
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
