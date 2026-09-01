"""Translate the fixed LCZero policy index into Luna's canonical move space."""

from __future__ import annotations

import itertools

import chess

LC0_POLICY_SIZE = 1858


def decode_lc0_policy_move(index: int, transform: int, board: chess.Board) -> chess.Move:
    if not 0 <= index < LC0_POLICY_SIZE:
        raise ValueError(f"LC0 policy index must be in [0, {LC0_POLICY_SIZE}), got {index}")
    move = _LC0_MOVES[index]
    inverse = _inverse_transform(transform)
    source = _transform_square(move.from_square, inverse)
    target = _transform_square(move.to_square, inverse)
    piece = board.piece_at(source)
    promotion = _promotion(move, piece, target)
    decoded = chess.Move(source, target, promotion=promotion)
    if piece is not None and piece.piece_type == chess.KING and _is_castling_target(board, target):
        destination = chess.G1 if chess.square_file(target) > chess.square_file(source) else chess.C1
        return chess.Move(source, destination)
    return decoded


def _inverse_transform(transform: int) -> int:
    if transform & 4:
        return 4 | (2 if transform & 1 else 0) | (1 if transform & 2 else 0)
    return transform


def _promotion(
    move: chess.Move,
    piece: chess.Piece | None,
    target: chess.Square,
) -> chess.PieceType | None:
    if move.promotion is not None:
        return move.promotion
    if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(target) == 7:
        return chess.KNIGHT
    return None


def _is_castling_target(board: chess.Board, target: chess.Square) -> bool:
    return board.piece_at(target) == chess.Piece(chess.ROOK, chess.WHITE)


def _transform_square(square: chess.Square, transform: int) -> chess.Square:
    file, rank = chess.square_file(square), chess.square_rank(square)
    if transform & (1 | 4):
        file = 7 - file
    if transform & (2 | 4):
        rank = 7 - rank
    return chess.square(file, rank)


def _is_policy_geometry(source: int, target: int) -> bool:
    file_delta = abs(chess.square_file(target) - chess.square_file(source))
    rank_delta = abs(chess.square_rank(target) - chess.square_rank(source))
    return source != target and (
        file_delta == 0 or rank_delta == 0 or file_delta == rank_delta or sorted((file_delta, rank_delta)) == [1, 2]
    )


_LC0_MOVES = tuple(
    itertools.chain(
        (
            chess.Move(source, target)
            for source in chess.SQUARES
            for target in chess.SQUARES
            if _is_policy_geometry(source, target)
        ),
        (
            chess.Move(chess.square(source, 6), chess.square(target, 7), promotion=promotion)
            for source in range(8)
            for target in range(max(0, source - 1), min(7, source + 1) + 1)
            for promotion in (chess.QUEEN, chess.ROOK, chess.BISHOP)
        ),
    )
)
if len(_LC0_MOVES) != LC0_POLICY_SIZE:
    raise RuntimeError(f"LC0 move table has {len(_LC0_MOVES)} entries, expected {LC0_POLICY_SIZE}")
