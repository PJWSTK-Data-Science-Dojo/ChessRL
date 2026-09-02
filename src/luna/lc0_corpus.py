from __future__ import annotations

import hashlib
from pathlib import Path

LC0_ADAPTER_VERSION = 1
_CORPUS_FINGERPRINT_VERSION = 1


def dataset_fingerprint(path: Path) -> str:
    resolved = path.expanduser().resolve()
    archives = lc0_archive_paths(resolved)
    if resolved.is_dir():
        return _corpus_fingerprint(archives)
    return _archive_fingerprint(archives[0])


def lc0_archive_paths(path: Path) -> tuple[Path, ...]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.casefold() != ".tar":
            raise ValueError(f"LC0 dataset file must be a .tar archive: {resolved}")
        return (resolved,)
    if not resolved.is_dir():
        raise FileNotFoundError(f"LC0 dataset does not exist: {resolved}")
    archives = tuple(sorted(resolved.glob("*.tar"), key=lambda archive: archive.name))
    if not archives:
        raise ValueError(f"LC0 dataset directory contains no .tar archives: {resolved}")
    return archives


def _archive_fingerprint(path: Path) -> str:
    digest = hashlib.sha256(f"luna-lc0-adapter:{LC0_ADAPTER_VERSION}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _corpus_fingerprint(archives: tuple[Path, ...]) -> str:
    digest = hashlib.sha256(f"luna-lc0-corpus:{_CORPUS_FINGERPRINT_VERSION}\0".encode())
    for archive in archives:
        digest.update(archive.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_archive_fingerprint(archive).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
