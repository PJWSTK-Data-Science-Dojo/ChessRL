from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from itertools import islice
from pathlib import Path
from unittest.mock import patch

import numpy as np

from luna.game.chess_game import ChessGame
from luna.lc0_dataset import Lc0Batch, Lc0DatasetConfig
from luna.lc0_pretraining import _Context, _training_batches
from luna.lc0_pretraining_config import Lc0PretrainingConfig


def _marked_batches(
    config: Lc0PretrainingConfig,
    dataset: Lc0DatasetConfig,
    produced: list[int],
) -> Iterator[Lc0Batch]:
    size = config.learner.batch_size
    for index in range(config.chunk_steps):
        marker = dataset.epoch * config.chunk_steps + index
        produced.append(marker)
        yield _marked_batch(size, marker)


def _marked_batch(size: int, marker: int) -> Lc0Batch:
    return Lc0Batch(
        observations=np.full((size, 1), marker, dtype=np.float32),
        policies=np.empty((size, 0), dtype=np.float32),
        value_targets=np.empty((size, 0), dtype=np.float32),
        valid_moves=np.empty((size, 0), dtype=np.bool_),
        visits=np.zeros(size, dtype=np.int64),
    )


def test_directory_resume_skips_only_the_current_deterministic_chunk(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.tar").write_bytes(b"archive")
    config = replace(Lc0PretrainingConfig(), dataset_path=corpus, total_steps=20_000, chunk_steps=1_000)
    context = _Context(config, ChessGame(), "a" * 64, "b" * 64)
    calls: list[tuple[int, int, int | None, int, int]] = []
    produced: list[int] = []

    def corpus_batches(
        _path: Path,
        dataset: Lc0DatasetConfig,
        _game: ChessGame,
        *,
        archive_offset: int,
        member_window_index: int,
        member_window_count: int,
    ) -> Iterator[Lc0Batch]:
        calls.append((dataset.epoch, archive_offset, dataset.max_samples, member_window_index, member_window_count))
        return _marked_batches(config, dataset, produced)

    with patch("luna.lc0_pretraining.iter_lc0_corpus_batches", side_effect=corpus_batches):
        expected = list(islice(_training_batches(context, 0), 19_017, 19_022))
        calls.clear()
        produced.clear()
        resumed = list(islice(_training_batches(context, 19_017), 5))

    expected_markers = [int(batch.observations[0, 0]) for batch in expected]
    assert [int(batch.observations[0, 0]) for batch in resumed] == expected_markers
    assert calls == [(19, 19, config.chunk_steps * config.learner.batch_size, 19, 20)]
    assert produced == list(range(19_000, 19_022))


def test_single_archive_resume_keeps_the_legacy_global_cursor(tmp_path: Path) -> None:
    archive = tmp_path / "training.tar"
    archive.write_bytes(b"archive")
    config = replace(Lc0PretrainingConfig(), dataset_path=archive)
    context = _Context(config, ChessGame(), "a" * 64, "b" * 64)
    calls: list[int] = []

    def archive_batches(
        _path: Path,
        dataset: Lc0DatasetConfig,
        _game: ChessGame,
    ) -> Iterator[Lc0Batch]:
        calls.append(dataset.epoch)
        size = config.learner.batch_size
        return iter((_marked_batch(size, dataset.epoch * 10), _marked_batch(size, dataset.epoch * 10 + 1)))

    with patch("luna.lc0_pretraining.iter_lc0_batches", side_effect=archive_batches):
        expected = list(islice(_training_batches(context, 0), 5, 8))
        calls.clear()
        resumed = list(islice(_training_batches(context, 5), 3))

    expected_markers = [int(batch.observations[0, 0]) for batch in expected]
    assert [int(batch.observations[0, 0]) for batch in resumed] == expected_markers
    assert calls == [0, 1, 2, 3]
