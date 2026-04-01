"""python-chess luna wrapper."""

from __future__ import annotations

import chess
import numpy as np


def board_to_numpy(board: chess.Board) -> np.ndarray:
    encoded_board = [0] * (8 * 8 * 6)
    for square_index, piece in board.piece_map().items():
        encoded_board[square_index * 6 + piece.piece_type - 1] = 1 if piece.color else -1
    return np.array(encoded_board, dtype=np.float32)


def move_to_action(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def action_to_move(action: int) -> chess.Move:
    to_sq = action % 64
    from_sq = int(action / 64)
    return chess.Move(from_sq, to_sq)


def player_from_turn(turn: bool) -> int:
    """1 for white, -1 for black."""
    return 1 if turn else -1


def mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square))


ACTION_SIZE = 64 * 64


class ChessGame:
    """python-chess wrapper."""

    def __init__(self) -> None:
        pass

    def get_init_board(self) -> chess.Board:
        return chess.Board()

    def get_board_size(self) -> tuple[int, int, int]:
        return (8, 8, 6)

    def to_array(self, board: chess.Board) -> np.ndarray:
        return board_to_numpy(board)

    def get_action_size(self) -> int:
        return ACTION_SIZE

    def get_next_state(self, board: chess.Board, player: int, action: int) -> tuple[chess.Board, int]:
        assert player_from_turn(board.turn) == player
        move = action_to_move(action)
        if not board.turn:
            move = mirror_move(move)
        if move not in board.legal_moves:
            move = chess.Move.from_uci(move.uci() + "q")
            if move not in board.legal_moves:
                raise ValueError(f"{move} not in {list(board.legal_moves)}")
        board = board.copy()
        board.push(move)
        return (board, player_from_turn(board.turn))

    def get_valid_moves(self, board: chess.Board, player: int) -> np.ndarray:
        assert player_from_turn(board.turn) == player
        acts = np.zeros(self.get_action_size(), dtype=np.float32)
        for move in board.legal_moves:
            acts[move_to_action(move)] = 1.0
        return acts

    def get_game_ended(self, board: chess.Board, player: int) -> float:
        """Return 0 if not ended. Otherwise return reward from *player*'s perspective."""
        outcome = board.outcome()
        if outcome is None:
            return 0.0
        if outcome.winner is None:
            return 1e-4
        winner_int = player_from_turn(outcome.winner)
        return 1.0 if winner_int == player else -1.0

    def get_canonical_form(self, board: chess.Board, player: int) -> chess.Board:
        assert player_from_turn(board.turn) == player
        if board.turn:
            return board
        else:
            return board.mirror()

    def get_symmetries(self, board: chess.Board, pi: list | np.ndarray) -> list[tuple]:
        return [(board, pi)]

    def string_representation(self, board: chess.Board) -> str:
        return board.fen()

    @staticmethod
    def display(board: chess.Board) -> None:
        print(board)
