from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from luna.game.chess_game import ChessGame
from luna.lc0_corpus import lc0_archive_paths
from luna.lc0_dataset import (
    Lc0Batch,
    Lc0DatasetConfig,
    _iter_archive_records,
    _iter_batches,
    _iter_samples,
    _iter_shard_records,
)


def iter_lc0_shard_batches(path: Path, config: Lc0DatasetConfig, game: ChessGame) -> Iterator[Lc0Batch]:
    archives = lc0_archive_paths(path)
    if len(archives) != 1:
        raise ValueError("LC0 shard iterator requires one .tar archive")
    samples = _iter_samples(_iter_archive_records(archives[0], config, True), config, game)
    yield from _iter_batches(samples, config.batch_size)


def iter_lc0_corpus_batches(
    path: Path,
    config: Lc0DatasetConfig,
    game: ChessGame,
    *,
    archive_offset: int,
    member_window_index: int,
    member_window_count: int,
) -> Iterator[Lc0Batch]:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError("LC0 corpus iterator requires a directory")
    if member_window_count <= 0 or not 0 <= member_window_index < member_window_count:
        raise ValueError("LC0 member window must have a valid index and positive count")
    archives = lc0_archive_paths(resolved)
    offset = archive_offset % len(archives)
    ordered = archives[offset:] + archives[:offset]
    windows = tuple((member_window_index + index) % member_window_count for index in range(member_window_count))
    records = _iter_shard_records(ordered, config, windows, member_window_count)
    samples = _iter_samples(records, config, game)
    yield from _iter_batches(samples, config.batch_size)
