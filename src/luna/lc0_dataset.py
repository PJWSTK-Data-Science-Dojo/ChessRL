from __future__ import annotations

import gzip
import hashlib
import math
import random
import struct
import tarfile
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chess
import numpy as np
from numpy.typing import NDArray

from luna.game.chess_game import ACTION_SIZE, OBS_PLANES, ChessGame, move_to_action
from luna.lc0_corpus import lc0_archive_paths
from luna.lc0_policy import LC0_POLICY_SIZE, decode_lc0_policy_move

type FloatArray = NDArray[np.float32]
type BoolArray = NDArray[np.bool_]
type IntArray = NDArray[np.int64]
type Lc0Split = Literal["train", "validation"]
type Lc0ValueSource = Literal["result", "root"]

_V6_SIZE, _V7_SIZE = 8356, 8396
_PLANES_OFFSET, _METADATA_OFFSET, _VISITS_OFFSET = 7440, 8272, 8340
_TRANSFORM_MASK, _DELETED_MASK = 0b111, 1 << 6
_CANONICAL_FORMATS = frozenset({3, 4})
_SUPPORTED_INPUT_FORMATS = frozenset({1, 2, 3, 4})
_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


class Lc0DatasetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Lc0DatasetConfig:
    batch_size: int = 256
    split: Lc0Split = "train"
    validation_fraction: float = 0.02
    split_seed: int = 0
    epoch: int = 0
    min_visits: int = 1
    max_samples: int | None = None
    shuffle_buffer_size: int = 2048
    value_source: Lc0ValueSource = "result"

    def __post_init__(self) -> None:
        if min(self.batch_size, self.shuffle_buffer_size) <= 0:
            raise ValueError("batch_size and shuffle_buffer_size must be positive")
        if not math.isfinite(self.validation_fraction) or not 0.0 <= self.validation_fraction <= 1.0:
            raise ValueError("validation_fraction must be finite and between zero and one")
        if min(self.epoch, self.min_visits) < 0:
            raise ValueError("epoch and min_visits must be non-negative")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when set")
        if self.split not in ("train", "validation"):
            raise ValueError(f"Unsupported LC0 split: {self.split}")
        if self.value_source not in ("result", "root"):
            raise ValueError(f"Unsupported LC0 value source: {self.value_source}")


@dataclass(frozen=True, slots=True)
class Lc0Sample:
    observation: FloatArray
    policy: FloatArray
    value_target: FloatArray
    valid_moves: BoolArray
    visits: int


@dataclass(frozen=True, slots=True)
class Lc0Batch:
    observations: FloatArray
    policies: FloatArray
    value_targets: FloatArray
    valid_moves: BoolArray
    visits: IntArray


@dataclass(frozen=True, slots=True)
class _RawRecord:
    data: bytes
    member: str
    index: int


def iter_lc0_samples(path: Path, config: Lc0DatasetConfig, game: ChessGame) -> Iterator[Lc0Sample]:
    yield from _iter_samples(_iter_records(path, config), config, game)


def _iter_samples(
    records: Iterator[_RawRecord],
    config: Lc0DatasetConfig,
    game: ChessGame,
) -> Iterator[Lc0Sample]:
    if game.get_action_size() != ACTION_SIZE:
        raise ValueError(f"Lc0 adapter requires Luna's {ACTION_SIZE}-action encoding")
    yielded = 0
    for record in _shuffled_records(records, config):
        visits = int(struct.unpack_from("<I", record.data, _VISITS_OFFSET)[0])
        if visits < config.min_visits or record.data[_METADATA_OFFSET + 6] & _DELETED_MASK:
            continue
        yield _decode_sample(record, config)
        yielded += 1
        if config.max_samples is not None and yielded >= config.max_samples:
            return


def iter_lc0_batches(path: Path, config: Lc0DatasetConfig, game: ChessGame) -> Iterator[Lc0Batch]:
    yield from _iter_batches(iter_lc0_samples(path, config, game), config.batch_size)


def _iter_batches(stream: Iterator[Lc0Sample], batch_size: int) -> Iterator[Lc0Batch]:
    samples: list[Lc0Sample] = []
    for sample in stream:
        samples.append(sample)
        if len(samples) == batch_size:
            yield _collate(samples)
            samples.clear()
    if samples:
        yield _collate(samples)


def _collate(samples: list[Lc0Sample]) -> Lc0Batch:
    return Lc0Batch(
        observations=np.stack([sample.observation for sample in samples]),
        policies=np.stack([sample.policy for sample in samples]),
        value_targets=np.stack([sample.value_target for sample in samples]),
        valid_moves=np.stack([sample.valid_moves for sample in samples]),
        visits=np.asarray([sample.visits for sample in samples], dtype=np.int64),
    )


def _iter_records(path: Path, config: Lc0DatasetConfig) -> Iterator[_RawRecord]:
    resolved = path.expanduser().resolve()
    qualify_members = resolved.is_dir()
    archives = lc0_archive_paths(resolved)
    if qualify_members:
        yield from _iter_shard_records(archives, config)
        return
    yield from _iter_archive_records(archives[0], config, False)


def _iter_shard_records(
    archives: tuple[Path, ...],
    config: Lc0DatasetConfig,
    windows: tuple[int, ...] = (),
    window_count: int = 0,
) -> Iterator[_RawRecord]:
    for window in windows or (None,):
        for archive in archives:
            yield from _iter_archive_records(archive, config, True, window, window_count)


def _iter_archive_records(
    path: Path,
    config: Lc0DatasetConfig,
    qualify_members: bool,
    member_window: int | None = None,
    member_window_count: int = 0,
) -> Iterator[_RawRecord]:
    try:
        with tarfile.open(path, mode="r|*") as archive:
            for member in archive:
                if not member.isfile() or Path(member.name).name == "LICENSE":
                    continue
                if not member.name.casefold().endswith(".gz"):
                    raise Lc0DatasetError(f"Unexpected non-gzip member in {path}: {member.name}")
                identity = f"{path.name}/{member.name}" if qualify_members else member.name
                if _member_split(identity, config) != config.split:
                    continue
                if (
                    member_window is not None
                    and _member_window(identity, config.split_seed, member_window_count) != member_window
                ):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise Lc0DatasetError(f"Cannot read tar member {member.name}")
                with extracted, gzip.GzipFile(fileobj=extracted, mode="rb") as stream:
                    yield from _read_game(stream, identity)
    except (OSError, EOFError, tarfile.TarError, gzip.BadGzipFile, zlib.error) as exc:
        raise Lc0DatasetError(f"Cannot stream Lc0 archive {path}: {exc}") from exc


def _read_game(stream: gzip.GzipFile, member: str) -> Iterator[_RawRecord]:
    frames = _read_frames(stream, member)
    try:
        first_data = next(frames)
    except StopIteration:
        raise Lc0DatasetError(f"Empty Lc0 game member: {member}") from None
    first_planes = _record_planes(first_data)
    if _piece_board(first_planes).board_fen() != chess.Board().board_fen():
        return
    yield _RawRecord(first_data, member, 0)
    for index, data in enumerate(frames, start=1):
        yield _RawRecord(data, member, index)


def _read_frames(stream: gzip.GzipFile, member: str) -> Iterator[bytes]:
    index = 0
    while header := stream.read(4):
        if len(header) != 4:
            raise Lc0DatasetError(f"Truncated version field in {member} at record {index}")
        version = struct.unpack("<I", header)[0]
        size = {6: _V6_SIZE, 7: _V7_SIZE}.get(version)
        if size is None:
            raise Lc0DatasetError(f"Unsupported Lc0 schema V{version} in {member} at record {index}")
        payload = stream.read(size - 4)
        if len(payload) != size - 4:
            raise Lc0DatasetError(f"Truncated Lc0 record in {member} at index {index}")
        yield header + payload
        index += 1


def _member_split(member: str, config: Lc0DatasetConfig) -> Lc0Split:
    material = f"{config.split_seed}\0{member}".encode()
    fraction = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2**64
    return "validation" if fraction < config.validation_fraction else "train"


def _member_window(member: str, seed: int, count: int) -> int:
    material = f"luna-lc0-window\0{seed}\0{member}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % count


def _shuffled_records(records: Iterator[_RawRecord], config: Lc0DatasetConfig) -> Iterator[_RawRecord]:
    rng = random.Random((config.split_seed << 32) ^ config.epoch)
    buffer: list[_RawRecord] = []
    for record in records:
        if len(buffer) < config.shuffle_buffer_size:
            buffer.append(record)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = record
    rng.shuffle(buffer)
    yield from buffer


def _decode_sample(record: _RawRecord, config: Lc0DatasetConfig) -> Lc0Sample:
    planes = _record_planes(record.data)
    board = _board_from_record(record.data, planes)
    policy, valid_moves = _policy_targets(record.data, board)
    return Lc0Sample(
        _observation(record.data, planes),
        policy,
        _value_target(record.data, config.value_source),
        valid_moves,
        int(struct.unpack_from("<I", record.data, _VISITS_OFFSET)[0]),
    )


def _record_planes(data: bytes) -> BoolArray:
    input_format = struct.unpack_from("<I", data, 4)[0]
    if input_format not in _SUPPORTED_INPUT_FORMATS:
        raise Lc0DatasetError(f"Unsupported Lc0 input format {input_format}")
    raw = np.frombuffer(data, dtype=np.uint8, count=104 * 8, offset=_PLANES_OFFSET)
    # Lc0 reverses bits in every stored byte, so big-endian unpack restores a1..h8 order.
    planes = np.unpackbits(raw).reshape(104, 8, 8).astype(np.bool_)
    transform = data[_METADATA_OFFSET + 6] & _TRANSFORM_MASK
    if input_format not in _CANONICAL_FORMATS and transform:
        raise Lc0DatasetError(f"Non-canonical input format {input_format} carries transform {transform}")
    if transform & 4:
        planes = planes.transpose(0, 2, 1)[:, ::-1, ::-1]
    if transform & 2:
        planes = planes[:, ::-1, :]
    if transform & 1:
        planes = planes[:, :, ::-1]
    return planes


def _board_from_record(data: bytes, planes: BoolArray) -> chess.Board:
    board = _piece_board(planes)
    board.halfmove_clock = data[_METADATA_OFFSET + 5]
    board.castling_rights = _castling_rights(data)
    board.ep_square = _en_passant_square(data, planes)
    if not board.is_valid():
        raise Lc0DatasetError(f"Lc0 record decodes to an invalid canonical board: {board.fen()}")
    return board


def _piece_board(planes: BoolArray) -> chess.Board:
    board = chess.Board(None)
    for color, offset in ((chess.WHITE, 0), (chess.BLACK, 6)):
        for plane, piece_type in enumerate(_PIECE_TYPES, offset):
            for square in np.flatnonzero(planes[plane]):
                board.set_piece_at(int(square), chess.Piece(piece_type, color))
    board.turn = chess.WHITE
    return board


def _castling_rights(data: bytes) -> chess.Bitboard:
    input_format = struct.unpack_from("<I", data, 4)[0]
    values = data[_METADATA_OFFSET : _METADATA_OFFSET + 4]
    expected = (chess.BB_A1, chess.BB_H1, chess.BB_A8, chess.BB_H8)
    rights = 0
    for value, square_mask in zip(values, expected, strict=True):
        if input_format >= 2 and value not in (0, 1, 128):
            raise Lc0DatasetError("Chess960 castling masks are incompatible with Luna standard chess")
        if value:
            rights |= square_mask
    return rights


def _en_passant_square(data: bytes, planes: BoolArray) -> chess.Square | None:
    input_format = struct.unpack_from("<I", data, 4)[0]
    if input_format in _CANONICAL_FORMATS:
        mask = data[_METADATA_OFFSET + 4]
        if mask == 0:
            return None
        transform = data[_METADATA_OFFSET + 6] & _TRANSFORM_MASK
        files = [file for file in range(8) if mask & (1 << file)]
        if len(files) != 1 or transform & 6:
            raise Lc0DatasetError("Invalid transformed en-passant mask")
        file = 7 - files[0] if transform & 1 else files[0]
        return chess.square(file, 5)
    current, previous = planes[6], planes[13 + 6]
    removed = np.flatnonzero(previous & ~current)
    added = np.flatnonzero(current & ~previous)
    if len(removed) != 1 or len(added) != 1:
        return None
    source, target = int(removed[0]), int(added[0])
    if chess.square_file(source) != chess.square_file(target) or abs(source - target) != 16:
        return None
    return (source + target) // 2


def _observation(data: bytes, planes: BoolArray) -> FloatArray:
    observation = np.zeros((8, 8, OBS_PLANES), dtype=np.float32)
    history = planes.reshape(8, 13, 8, 8)
    for index in range(8):
        offset = index * 14
        observation[:, :, offset : offset + 12] = np.moveaxis(history[index, :12], 0, -1)
        observation[:, :, offset + 12] = history[index, 12]
    us_ooo, us_oo, them_ooo, them_oo = data[_METADATA_OFFSET : _METADATA_OFFSET + 4]
    observation[:, :, 112:116] = (bool(us_oo), bool(us_ooo), bool(them_oo), bool(them_ooo))
    ep_square = _en_passant_square(data, planes)
    if ep_square is not None:
        observation[chess.square_rank(ep_square), chess.square_file(ep_square), 116] = 1.0
    observation[:, :, 117] = min(data[_METADATA_OFFSET + 5] / 100.0, 1.0)
    observation[:, :, 118] = 1.0
    return observation


def _policy_targets(data: bytes, board: chess.Board) -> tuple[FloatArray, BoolArray]:
    probabilities = np.frombuffer(data, dtype="<f4", count=LC0_POLICY_SIZE, offset=8)
    if not np.isfinite(probabilities).all():
        raise Lc0DatasetError("Lc0 policy contains a non-finite probability")
    transform = data[_METADATA_OFFSET + 6] & _TRANSFORM_MASK
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    valid = np.zeros(ACTION_SIZE, dtype=np.bool_)
    legal_moves = set(board.legal_moves)
    for index in np.flatnonzero(probabilities >= 0.0):
        move = decode_lc0_policy_move(int(index), transform, board)
        if move not in legal_moves:
            raise Lc0DatasetError(f"Lc0 policy index {index} decodes to illegal move {move.uci()}")
        action = move_to_action(move)
        valid[action] = True
        policy[action] += probabilities[index]
    expected_actions = {move_to_action(move) for move in legal_moves}
    if set(np.flatnonzero(valid)) != expected_actions:
        raise Lc0DatasetError("Lc0 policy legality mask disagrees with the decoded board")
    mass = float(policy.sum())
    if not math.isfinite(mass) or mass <= 0.0:
        raise Lc0DatasetError("Lc0 policy has no positive probability mass")
    policy /= mass
    return policy, valid


def _value_target(data: bytes, source: Lc0ValueSource) -> FloatArray:
    q_offset, d_offset = (8308, 8312) if source == "result" else (8280, 8288)
    q, draw = struct.unpack_from("<f", data, q_offset)[0], struct.unpack_from("<f", data, d_offset)[0]
    values = np.asarray(((1.0 - draw - q) / 2.0, draw, (1.0 - draw + q) / 2.0), dtype=np.float32)
    if not np.isfinite(values).all() or float(values.min()) < -1e-4:
        raise Lc0DatasetError(f"Invalid Lc0 WDL target q={q}, d={draw}")
    np.clip(values, 0.0, 1.0, out=values)
    values /= values.sum()
    return values
