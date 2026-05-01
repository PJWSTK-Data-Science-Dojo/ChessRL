"""python-chess luna wrapper with spatial action encoding.

Action encoding (4288 total):
    indices   0..4095  -- normal moves + queen promotions: from_square * 64 + to_square
    indices 4096..4159 -- knight promotions: from_file * 8 + to_file (ranks implied 7th->8th)
    indices 4160..4223 -- rook promotions
    indices 4224..4287 -- bishop promotions
Queen promotions share the base 0..4095 range (default promotion type).
"""

import random

import chess
import numpy as np
from loguru import logger

_PROMOTION_OFFSETS: dict[int, int] = {
    chess.KNIGHT: 4096,
    chess.ROOK: 4160,
    chess.BISHOP: 4224,
}
_OFFSET_TO_PROMO: dict[int, int] = {v: k for k, v in _PROMOTION_OFFSETS.items()}

ACTION_SIZE = 4288
DRAW_VALUE = 1e-4

NUM_PIECE_PLANES = 6
NUM_AUX_PLANES = 7  # castling (4) + en-passant (1) + halfmove clock (1) + color (1)
OBS_PLANES = NUM_PIECE_PLANES + NUM_AUX_PLANES


def board_to_numpy(board: chess.Board) -> np.ndarray:
    """Encode board as (8, 8, OBS_PLANES) with piece planes + auxiliary planes."""
    arr = np.zeros((8, 8, OBS_PLANES), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        rank, file = divmod(sq, 8)
        arr[rank, file, piece.piece_type - 1] = 1.0 if piece.color else -1.0

    arr[:, :, 6] = float(board.has_kingside_castling_rights(chess.WHITE))
    arr[:, :, 7] = float(board.has_queenside_castling_rights(chess.WHITE))
    arr[:, :, 8] = float(board.has_kingside_castling_rights(chess.BLACK))
    arr[:, :, 9] = float(board.has_queenside_castling_rights(chess.BLACK))

    if board.ep_square is not None:
        ep_rank, ep_file = divmod(board.ep_square, 8)
        arr[ep_rank, ep_file, 10] = 1.0

    arr[:, :, 11] = board.halfmove_clock / 100.0
    arr[:, :, 12] = 1.0 if board.turn else 0.0

    return arr


def move_to_action(move: chess.Move) -> int:
    base = move.from_square * 64 + move.to_square
    if move.promotion is not None and move.promotion in _PROMOTION_OFFSETS:
        from_file = chess.square_file(move.from_square)
        to_file = chess.square_file(move.to_square)
        return _PROMOTION_OFFSETS[move.promotion] + from_file * 8 + to_file
    return base


def action_to_move(action: int) -> chess.Move:
    for offset, promo_type in _OFFSET_TO_PROMO.items():
        if action >= offset and action < offset + 64:
            idx = action - offset
            from_file = idx // 8
            to_file = idx % 8
            from_sq = chess.square(from_file, 6)
            to_sq = chess.square(to_file, 7)
            return chess.Move(from_sq, to_sq, promotion=promo_type)
    to_sq = action % 64
    from_sq = action // 64
    return chess.Move(from_sq, to_sq)


def action_to_planes(action: int) -> tuple[int, int]:
    """Decompose an action index into (from_square, to_square)."""
    move = action_to_move(action)
    return move.from_square, move.to_square


def player_from_turn(turn: bool) -> int:
    """1 for white, -1 for black."""
    return 1 if turn else -1


def mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


class ChessGame:
    """python-chess wrapper."""

    def __init__(self) -> None:
        self._illegal_move_fallback_count = 0

    def get_init_board(self) -> chess.Board:
        return chess.Board()

    def replay_board_player(self, actions: np.ndarray, pos_idx: int) -> tuple[chess.Board, int]:
        """Replay ``actions[:pos_idx]`` from the start position; return board and side to move."""
        board = self.get_init_board()
        player = 1
        for t in range(pos_idx):
            board, player = self.get_next_state(board, player, int(actions[t]))
        return board, player

    def get_board_size(self) -> tuple[int, int, int]:
        return (8, 8, OBS_PLANES)

    def to_array(self, board: chess.Board) -> np.ndarray:
        return board_to_numpy(board)

    def get_action_size(self) -> int:
        return ACTION_SIZE

    def get_illegal_move_count(self) -> int:
        """Return the number of times illegal moves were handled with fallback."""
        return self._illegal_move_fallback_count

    def reset_illegal_move_count(self) -> None:
        """Reset the illegal move fallback counter."""
        self._illegal_move_fallback_count = 0

    def get_next_state(self, board: chess.Board, player: int, action: int) -> tuple[chess.Board, int]:
        """Execute action and return next (board, player).

        If the action is illegal, tries promotion fallback, then picks a random legal move
        as a last resort to prevent crashes during self-play.
        """
        assert player_from_turn(board.turn) == player
        move = action_to_move(action)
        if not board.turn:
            move = mirror_move(move)

        # Try to execute the move
        if move not in board.legal_moves:
            # Fallback 1: Try queen promotion if it's a pawn-to-back-rank move
            if move.promotion is None:
                promo_move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
                if promo_move in board.legal_moves:
                    move = promo_move

            # Fallback 2: If still illegal, pick a random legal move as last resort
            if move not in board.legal_moves:
                legal_move_list = list(board.legal_moves)
                if len(legal_move_list) > 0:
                    self._illegal_move_fallback_count += 1
                    move = random.choice(legal_move_list)
                    logger.warning(
                        "Illegal action {} selected. Falling back to random legal move: {} (total fallbacks: {})",
                        action,
                        move,
                        self._illegal_move_fallback_count,
                    )
                else:
                    # No legal moves - should never happen in valid chess positions
                    raise ValueError(f"No legal moves available. Position: {board.fen()}")

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
            return DRAW_VALUE
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
        logger.info("\n{}", board)
