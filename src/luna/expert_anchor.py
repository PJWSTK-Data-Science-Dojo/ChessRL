"""Persistent LC0 supervision used to anchor online self-play training."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from luna.config import EzV2LearnerConfig
from luna.game.chess_game import ChessGame
from luna.lc0_batch_stream import iter_lc0_corpus_batches, iter_lc0_shard_batches
from luna.lc0_corpus import dataset_fingerprint, lc0_archive_paths
from luna.lc0_dataset import Lc0Batch, Lc0DatasetConfig

_CHUNK_STEPS = 1_000
_WINDOWS_PER_SHARD = 5


def expert_anchor_fingerprint(path: Path) -> str:
    return dataset_fingerprint(path)


class ExpertAnchorBatchSource:
    def __init__(
        self,
        learner: EzV2LearnerConfig,
        game: ChessGame,
        *,
        seed: int,
        starting_step: int,
    ) -> None:
        self._path = Path(learner.expert_anchor_path).expanduser().resolve()
        self._shards = lc0_archive_paths(self._path)
        actual = expert_anchor_fingerprint(self._path)
        if actual != learner.expert_anchor_fingerprint:
            raise ValueError(
                f"expert anchor fingerprint mismatch: expected {learner.expert_anchor_fingerprint}, got {actual}"
            )
        self._game = game
        self._batch_size = max(1, math.ceil(learner.batch_size * learner.expert_anchor_fraction))
        self._base_config = Lc0DatasetConfig(
            batch_size=self._batch_size,
            split="train",
            split_seed=seed,
            min_visits=1,
            shuffle_buffer_size=max(2048, self._batch_size * 4),
            value_source="root",
        )
        if isinstance(starting_step, bool) or not isinstance(starting_step, int) or starting_step < 0:
            raise ValueError("starting_step must be a non-negative integer")
        if self._path.is_dir():
            self._batches = self._corpus_batches(starting_step)
        else:
            self._batches = self._single_archive_batches()
            self._skip_batches(starting_step)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def next_batch(self) -> Lc0Batch:
        return self._next_batch()

    def _next_batch(self) -> Lc0Batch:
        try:
            return next(self._batches)
        except StopIteration:
            raise ValueError("expert anchor training split contains no accepted samples") from None

    def _skip_batches(self, count: int) -> None:
        for _ in range(count):
            self._next_batch()

    def _single_archive_batches(self) -> Iterator[Lc0Batch]:
        epoch = 0
        while True:
            yielded = False
            for batch in iter_lc0_shard_batches(self._shards[0], replace(self._base_config, epoch=epoch), self._game):
                yielded = True
                yield batch
            if not yielded:
                raise ValueError("expert anchor training split contains no accepted samples")
            epoch += 1

    def _corpus_batches(self, starting_step: int) -> Iterator[Lc0Batch]:
        chunk_index, offset = divmod(starting_step, _CHUNK_STEPS)
        window_count = len(self._shards) * _WINDOWS_PER_SHARD
        while True:
            config = replace(
                self._base_config,
                epoch=chunk_index,
                max_samples=_CHUNK_STEPS * self._batch_size,
            )
            batches = iter_lc0_corpus_batches(
                self._path,
                config,
                self._game,
                archive_offset=chunk_index,
                member_window_index=chunk_index % window_count,
                member_window_count=window_count,
            )
            yield from self._batch_window(batches, offset)
            chunk_index += 1
            offset = 0

    def _batch_window(self, batches: Iterator[Lc0Batch], offset: int) -> Iterator[Lc0Batch]:
        for index in range(_CHUNK_STEPS):
            try:
                batch = next(batches)
            except StopIteration:
                raise ValueError("expert anchor corpus cannot fill a deterministic chunk") from None
            if len(batch.observations) != self._batch_size:
                raise ValueError("expert anchor corpus produced a partial deterministic batch")
            if index >= offset:
                yield batch


def build_expert_anchor_source(
    learner: EzV2LearnerConfig,
    game: ChessGame,
    *,
    seed: int,
    starting_step: int,
) -> ExpertAnchorBatchSource | None:
    if learner.expert_anchor_loss_weight == 0.0:
        return None
    return ExpertAnchorBatchSource(learner, game, seed=seed, starting_step=starting_step)
