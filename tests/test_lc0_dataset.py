from __future__ import annotations

import gzip
import hashlib
import io
import struct
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import cast

import chess
import numpy as np
import pytest

from luna.game.chess_game import ACTION_SIZE, SIDE_TO_MOVE_PLANE, ChessGame, move_to_action
from luna.lc0_batch_stream import iter_lc0_corpus_batches, iter_lc0_shard_batches
from luna.lc0_corpus import dataset_fingerprint, lc0_archive_paths
from luna.lc0_dataset import (
    Lc0DatasetConfig,
    Lc0DatasetError,
    Lc0Split,
    Lc0ValueSource,
    _member_window,
    iter_lc0_batches,
    iter_lc0_samples,
)

_POLICY_SIZE = 1858
_START = chess.Board()


def _policy_moves() -> tuple[chess.Move, ...]:
    moves: list[chess.Move] = []
    for source in chess.SQUARES:
        for target in chess.SQUARES:
            file_delta = abs(chess.square_file(target) - chess.square_file(source))
            rank_delta = abs(chess.square_rank(target) - chess.square_rank(source))
            geometry = file_delta == 0 or rank_delta == 0 or file_delta == rank_delta
            if source != target and (geometry or sorted((file_delta, rank_delta)) == [1, 2]):
                moves.append(chess.Move(source, target))
    for source_file in range(8):
        for target_file in range(max(0, source_file - 1), min(7, source_file + 1) + 1):
            for promotion in (chess.QUEEN, chess.ROOK, chess.BISHOP):
                moves.append(
                    chess.Move(chess.square(source_file, 6), chess.square(target_file, 7), promotion=promotion)
                )
    return tuple(moves)


_MOVES = _policy_moves()
_MOVE_TO_INDEX = {move.uci(): index for index, move in enumerate(_MOVES)}


def _transform_square(square: chess.Square, transform: int) -> chess.Square:
    file, rank = chess.square_file(square), chess.square_rank(square)
    if transform & 5:
        file = 7 - file
    if transform & 6:
        rank = 7 - rank
    return chess.square(file, rank)


def _lc0_index(board: chess.Board, move: chess.Move, transform: int) -> int:
    target = move.to_square
    if board.is_castling(move):
        target = chess.H1 if chess.square_file(target) > chess.square_file(move.from_square) else chess.A1
    promotion = move.promotion
    if promotion == chess.KNIGHT:
        promotion = None
    transformed = chess.Move(
        _transform_square(move.from_square, transform),
        _transform_square(target, transform),
        promotion=promotion,
    )
    return _MOVE_TO_INDEX[transformed.uci()]


def _policy(board: chess.Board, weights: dict[chess.Move, float], transform: int) -> np.ndarray:
    probabilities = np.full(_POLICY_SIZE, -1.0, dtype="<f4")
    for move in board.legal_moves:
        probabilities[_lc0_index(board, move, transform)] = 0.0
    for move, weight in weights.items():
        probabilities[_lc0_index(board, move, transform)] = weight
    return probabilities


def _forward_transform(planes: np.ndarray, transform: int) -> np.ndarray:
    if transform & 1:
        planes = planes[:, :, ::-1]
    if transform & 2:
        planes = planes[:, ::-1, :]
    if transform & 4:
        planes = planes.transpose(0, 2, 1)[:, ::-1, ::-1]
    return planes


def _planes(board: chess.Board, previous: chess.Board | None, transform: int) -> bytes:
    encoded = np.zeros((104, 8, 8), dtype=np.uint8)
    for history_index, position in enumerate(position for position in (board, previous) if position is not None):
        for square, piece in position.piece_map().items():
            color_offset = 0 if piece.color else 6
            plane = history_index * 13 + color_offset + piece.piece_type - 1
            encoded[plane, chess.square_rank(square), chess.square_file(square)] = 1
    transformed = _forward_transform(encoded, transform)
    return np.packbits(transformed.reshape(-1)).tobytes()


def _record(
    board: chess.Board,
    weights: dict[chess.Move, float],
    *,
    version: int = 6,
    input_format: int = 1,
    transform: int = 0,
    actual_black: bool = False,
    previous: chess.Board | None = None,
    result: tuple[float, float] = (0.4, 0.2),
    root: tuple[float, float] = (0.2, 0.4),
    visits: int = 800,
) -> bytes:
    size = 8356 if version == 6 else 8396
    data = bytearray(size)
    struct.pack_into("<II", data, 0, version, input_format)
    data[8:7440] = _policy(board, weights, transform).tobytes()
    data[7440:8272] = _planes(board, previous, transform)
    rights = (
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.BLACK),
        board.has_kingside_castling_rights(chess.BLACK),
    )
    castling = (
        tuple(int(value) for value in rights)
        if input_format == 1
        else (int(rights[0]), 128 * int(rights[1]), int(rights[2]), 128 * int(rights[3]))
    )
    ep_mask = 0 if board.ep_square is None else 1 << chess.square_file(board.ep_square)
    side_or_ep = int(actual_black) if input_format < 3 else ep_mask
    invariance = transform | ((1 << 7) if actual_black and input_format >= 3 else 0)
    data[8272:8280] = bytes((*castling, side_or_ep, min(board.halfmove_clock, 255), invariance, 0))
    values = (
        root[0],
        root[0],
        root[1],
        root[1],
        0.0,
        0.0,
        0.0,
        result[0],
        result[1],
        root[0],
        root[1],
        0.0,
        0.0,
        0.0,
        0.0,
    )
    struct.pack_into("<15fIHH2f", data, 8280, *values, visits, 0, 0, 0.0, 0.0)
    return bytes(data)


def _write_archive(tmp_path: Path, members: dict[str, list[bytes]]) -> Path:
    path = tmp_path / "lc0.tar"
    with tarfile.open(path, mode="w") as archive:
        for name, records in members.items():
            compressed = gzip.compress(b"".join(records))
            info = tarfile.TarInfo(name)
            info.size = len(compressed)
            archive.addfile(info, io.BytesIO(compressed))
    return path


def _config(batch_size: int = 8) -> Lc0DatasetConfig:
    return Lc0DatasetConfig(batch_size=batch_size, validation_fraction=0.0, shuffle_buffer_size=1)


def _archive_with_position(tmp_path: Path, record: bytes) -> Path:
    start = _record(_START, {chess.Move.from_uci("e2e4"): 1.0})
    return _write_archive(tmp_path, {"training.1.gz": [start, record]})


def test_config_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="split"):
        Lc0DatasetConfig(split=cast(Lc0Split, "test"))


def test_config_rejects_unknown_value_source() -> None:
    with pytest.raises(ValueError, match="value source"):
        Lc0DatasetConfig(value_source=cast(Lc0ValueSource, "search"))


def test_decodes_official_move_indices_and_wdl_order(tmp_path: Path, chess_game: ChessGame) -> None:
    assert _MOVE_TO_INDEX["e2e4"] == 322
    assert _MOVE_TO_INDEX["e1h1"] == 103
    assert _MOVE_TO_INDEX["a7a8q"] == 1792
    path = _write_archive(tmp_path, {"training.1.gz": [_record(_START, {chess.Move.from_uci("e2e4"): 1.0})]})

    sample = next(iter_lc0_samples(path, _config(), chess_game))

    action = move_to_action(chess.Move.from_uci("e2e4"))
    assert sample.policy[action] == pytest.approx(1.0)
    assert sample.valid_moves[action]
    np.testing.assert_allclose(sample.value_target, [0.2, 0.2, 0.6])
    assert sample.observation.shape == (8, 8, 119)
    assert np.all(sample.observation[:, :, SIDE_TO_MOVE_PLANE] == 1.0)


def test_restores_transform_and_black_side_canonicalization(tmp_path: Path, chess_game: ChessGame) -> None:
    board = chess.Board("3r4/4k3/8/1K6/8/8/8/8 w - - 0 1")
    move = chess.Move.from_uci("b5c5")
    assert _lc0_index(board, move, 7) == 857
    transformed = _record(board, {move: 1.0}, input_format=3, transform=7, actual_black=True)
    sample = list(iter_lc0_samples(_archive_with_position(tmp_path, transformed), _config(), chess_game))[-1]

    assert sample.observation[4, 1, chess.KING - 1] == 1.0
    assert sample.policy[move_to_action(move)] == pytest.approx(1.0)
    np.testing.assert_allclose(sample.value_target, [0.2, 0.2, 0.6])


def test_converts_castling_and_every_promotion(tmp_path: Path, chess_game: ChessGame) -> None:
    castling_board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    kingside, queenside = chess.Move.from_uci("e1g1"), chess.Move.from_uci("e1c1")
    castling = _record(castling_board, {kingside: 0.75, queenside: 0.25})
    castling_sample = list(iter_lc0_samples(_archive_with_position(tmp_path, castling), _config(), chess_game))[-1]
    assert castling_sample.policy[move_to_action(kingside)] == pytest.approx(0.75)
    assert castling_sample.policy[move_to_action(queenside)] == pytest.approx(0.25)

    promotion_board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    promotions = [chess.Move.from_uci(f"a7a8{piece}") for piece in "qrbn"]
    promotion = _record(promotion_board, {move: 0.25 for move in promotions}, version=7)
    samples = list(iter_lc0_samples(_archive_with_position(tmp_path, promotion), _config(), chess_game))
    for move in promotions:
        assert samples[-1].policy[move_to_action(move)] == pytest.approx(0.25)


def test_infers_classical_en_passant_from_history(tmp_path: Path, chess_game: ChessGame) -> None:
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    previous = chess.Board("4k3/3p4/8/4P3/8/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("e5d6")
    record = _record(board, {move: 1.0}, previous=previous)

    sample = list(iter_lc0_samples(_archive_with_position(tmp_path, record), _config(), chess_game))[-1]

    assert sample.observation[5, 3, 116] == 1.0
    assert sample.policy[move_to_action(move)] == pytest.approx(1.0)


def test_batches_v6_v7_filters_chess960_and_fingerprints(tmp_path: Path, chess_game: ChessGame) -> None:
    standard_v6 = _record(_START, {chess.Move.from_uci("e2e4"): 1.0}, visits=7)
    standard_v7 = _record(_START, {chess.Move.from_uci("d2d4"): 1.0}, version=7, visits=9)
    chess960 = chess.Board.from_chess960_pos(0)
    chess960.castling_rights = 0
    frc = _record(chess960, {next(iter(chess960.legal_moves)): 1.0})
    path = _write_archive(tmp_path, {"standard.gz": [standard_v6, standard_v7], "frc.gz": [frc]})

    batches = list(iter_lc0_batches(path, _config(batch_size=2), chess_game))

    assert len(batches) == 1
    assert batches[0].observations.shape == (2, 8, 8, 119)
    assert batches[0].policies.shape == (2, ACTION_SIZE)
    assert batches[0].value_targets.shape == (2, 3)
    np.testing.assert_array_equal(batches[0].visits, [7, 9])
    fingerprint = dataset_fingerprint(path)
    assert fingerprint == dataset_fingerprint(path)
    path.write_bytes(path.read_bytes() + b"changed")
    assert fingerprint != dataset_fingerprint(path)


def test_streams_a_deterministic_multi_archive_corpus(tmp_path: Path, chess_game: ChessGame) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _write_archive(
        first_dir,
        {"first.gz": [_record(_START, {chess.Move.from_uci("e2e4"): 1.0}, visits=7)]},
    )
    second = _write_archive(
        second_dir,
        {"second.gz": [_record(_START, {chess.Move.from_uci("d2d4"): 1.0}, visits=9)]},
    )
    (corpus / "b.tar").write_bytes(second.read_bytes())
    (corpus / "a.tar").write_bytes(first.read_bytes())

    batches = list(iter_lc0_batches(corpus, _config(batch_size=2), chess_game))
    rotated = list(
        iter_lc0_corpus_batches(
            corpus,
            _config(batch_size=2),
            chess_game,
            archive_offset=1,
            member_window_index=0,
            member_window_count=1,
        )
    )

    assert [path.name for path in lc0_archive_paths(corpus)] == ["a.tar", "b.tar"]
    np.testing.assert_array_equal(batches[0].visits, [7, 9])
    np.testing.assert_array_equal(rotated[0].visits, [9, 7])
    original = dataset_fingerprint(corpus)
    (corpus / "a.tar").rename(corpus / "c.tar")
    assert dataset_fingerprint(corpus) != original


def test_member_windows_sample_every_shard_without_reusing_prefixes(tmp_path: Path, chess_game: ChessGame) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for shard, base_visit in (("a.tar", 10), ("b.tar", 20)):
        members: dict[str, list[bytes]] = {}
        for window in range(2):
            index = next(
                candidate
                for candidate in range(1_000)
                if _member_window(f"{shard}/game-{candidate}.gz", 0, 2) == window
            )
            members[f"game-{index}.gz"] = [
                _record(_START, {chess.Move.from_uci("e2e4"): 1.0}, visits=base_visit + window)
            ]
        source_dir = tmp_path / shard.removesuffix(".tar")
        source_dir.mkdir()
        archive = _write_archive(source_dir, members)
        (corpus / shard).write_bytes(archive.read_bytes())

    def visits(window: int) -> list[int]:
        config = replace(_config(batch_size=2), max_samples=2)
        batches = iter_lc0_corpus_batches(
            corpus,
            config,
            chess_game,
            archive_offset=0,
            member_window_index=window,
            member_window_count=2,
        )
        return next(batches).visits.tolist()

    assert visits(0) == [10, 20]
    assert visits(1) == [11, 21]


def test_shard_iterator_preserves_directory_split_identity(tmp_path: Path, chess_game: ChessGame) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    archive = _write_archive(
        tmp_path,
        {"training.gz": [_record(_START, {chess.Move.from_uci("e2e4"): 1.0})]},
    )
    shard = corpus / "a.tar"
    shard.write_bytes(archive.read_bytes())

    def split_fraction(seed: int, identity: str) -> float:
        material = f"{seed}\0{identity}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2**64

    seed = next(
        candidate
        for candidate in range(10_000)
        if split_fraction(candidate, "a.tar/training.gz") < 0.5 and split_fraction(candidate, "training.gz") >= 0.5
    )
    config = Lc0DatasetConfig(
        batch_size=1,
        split="train",
        validation_fraction=0.5,
        split_seed=seed,
        shuffle_buffer_size=1,
    )

    assert list(iter_lc0_batches(corpus, config, chess_game)) == []
    assert list(iter_lc0_shard_batches(shard, config, chess_game)) == []
    assert len(list(iter_lc0_batches(shard, config, chess_game))) == 1


def test_rejects_an_empty_multi_archive_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "empty"
    corpus.mkdir()

    with pytest.raises(ValueError, match=r"contains no \.tar archives"):
        lc0_archive_paths(corpus)


def test_rejects_truncated_record(tmp_path: Path, chess_game: ChessGame) -> None:
    path = _write_archive(tmp_path, {"broken.gz": [_record(_START, {chess.Move.from_uci("e2e4"): 1.0})[:-1]]})

    with pytest.raises(Lc0DatasetError, match="Truncated Lc0 record"):
        list(iter_lc0_samples(path, _config(), chess_game))
