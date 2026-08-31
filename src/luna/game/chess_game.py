"""Chess rules, temporal observations, and spatial action encoding.

Action encoding (4288 total):
    indices   0..4095  -- normal moves + queen promotions: from_square * 64 + to_square
    indices 4096..4159 -- knight promotions: from_file * 8 + to_file (ranks implied 7th->8th)
    indices 4160..4223 -- rook promotions
    indices 4224..4287 -- bishop promotions
Queen promotions share the base 0..4095 range (default promotion type).
"""

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

HISTORY_LENGTH = 8
PIECE_PLANES_PER_POSITION = 12
REPETITION_PLANES_PER_POSITION = 2
PLANES_PER_POSITION = PIECE_PLANES_PER_POSITION + REPETITION_PLANES_PER_POSITION
NUM_AUX_PLANES = 7
OBS_PLANES = HISTORY_LENGTH * PLANES_PER_POSITION + NUM_AUX_PLANES

CASTLING_PLANES_START = HISTORY_LENGTH * PLANES_PER_POSITION
EN_PASSANT_PLANE = CASTLING_PLANES_START + 4
HALFMOVE_CLOCK_PLANE = CASTLING_PLANES_START + 5
SIDE_TO_MOVE_PLANE = CASTLING_PLANES_START + 6
HALFMOVE_CLOCK_NORMALIZER = 100.0
_MIN_THREEFOLD_CLAIM_HALFMOVES = 7


def _observation_history_plies(board: chess.Board) -> int:
    """Bound the stack while preserving all eight temporal repetition planes."""
    temporal_plies = min(HISTORY_LENGTH - 1, len(board.move_stack))
    if temporal_plies == len(board.move_stack):
        return temporal_plies

    oldest_observation = board.copy(stack=temporal_plies)
    for _ in range(temporal_plies):
        oldest_observation.pop()
    return min(len(board.move_stack), temporal_plies + oldest_observation.halfmove_clock)


def board_to_numpy(board: chess.Board) -> np.ndarray:
    """Encode a position and its history as an AlphaZero-style 119-plane tensor.

    The first 112 planes contain eight temporal positions, newest first. Each
    position uses six white piece planes, six black piece planes, and two
    repetition planes. The final seven planes encode the current position's four
    castling rights, en-passant square, normalized halfmove clock, and side to move.

    A board constructed from FEN has no recoverable history, so all unavailable
    temporal slots remain zero. The returned layout is HWC to match the rest of the
    training pipeline.
    """
    arr = np.zeros((8, 8, OBS_PLANES), dtype=np.float32)

    historical_board = board.copy(stack=_observation_history_plies(board))
    for history_index in range(HISTORY_LENGTH):
        plane_offset = history_index * PLANES_PER_POSITION
        for square, piece in historical_board.piece_map().items():
            rank, file = divmod(square, 8)
            color_offset = 0 if piece.color == chess.WHITE else 6
            piece_plane = plane_offset + color_offset + piece.piece_type - 1
            arr[rank, file, piece_plane] = 1.0

        arr[:, :, plane_offset + PIECE_PLANES_PER_POSITION] = float(historical_board.is_repetition(2))
        arr[:, :, plane_offset + PIECE_PLANES_PER_POSITION + 1] = float(historical_board.is_repetition(3))

        if not historical_board.move_stack:
            break
        historical_board.pop()

    arr[:, :, CASTLING_PLANES_START] = float(board.has_kingside_castling_rights(chess.WHITE))
    arr[:, :, CASTLING_PLANES_START + 1] = float(board.has_queenside_castling_rights(chess.WHITE))
    arr[:, :, CASTLING_PLANES_START + 2] = float(board.has_kingside_castling_rights(chess.BLACK))
    arr[:, :, CASTLING_PLANES_START + 3] = float(board.has_queenside_castling_rights(chess.BLACK))

    if board.ep_square is not None:
        ep_rank, ep_file = divmod(board.ep_square, 8)
        arr[ep_rank, ep_file, EN_PASSANT_PLANE] = 1.0

    arr[:, :, HALFMOVE_CLOCK_PLANE] = min(board.halfmove_clock / HALFMOVE_CLOCK_NORMALIZER, 1.0)
    arr[:, :, SIDE_TO_MOVE_PLANE] = float(board.turn == chess.WHITE)

    return arr


def move_to_action(move: chess.Move) -> int:
    if move.drop is not None:
        raise ValueError("Drop moves are not part of standard chess action encoding")
    base = move.from_square * 64 + move.to_square
    if move.promotion is not None and move.promotion != chess.QUEEN:
        if move.promotion not in _PROMOTION_OFFSETS:
            raise ValueError(f"Unsupported promotion piece: {move.promotion}")
        from_file = chess.square_file(move.from_square)
        to_file = chess.square_file(move.to_square)
        return _PROMOTION_OFFSETS[move.promotion] + from_file * 8 + to_file
    return base


def action_to_move(action: int) -> chess.Move:
    if not 0 <= action < ACTION_SIZE:
        raise ValueError(f"Action index must be in [0, {ACTION_SIZE}), got {action}")
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


def player_from_turn(turn: bool) -> int:
    """1 for white, -1 for black."""
    return 1 if turn else -1


def _validate_player_turn(board: chess.Board, player: int) -> None:
    expected_player = player_from_turn(board.turn)
    if player != expected_player:
        raise ValueError(
            f"Player {player} does not match the side to move ({expected_player}) in position {board.fen()}"
        )


def mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def mirror_board(board: chess.Board) -> chess.Board:
    """Mirror a position while retaining all semantically relevant history.

    ``python-chess`` intentionally drops the stack in :meth:`Board.mirror`. Search
    needs reversible history for repetition adjudication and the latest seven
    plies for temporal observations. Earlier moves cannot affect either result.
    """
    history_plies = _observation_history_plies(board)
    source = board if len(board.move_stack) <= history_plies else board.copy(stack=history_plies)
    mirrored = source.root().mirror()
    for move in source.move_stack:
        mirrored.push(mirror_move(move))
    mirrored.fullmove_number = board.fullmove_number
    return mirrored


class ChessGame:
    """Rules adapter used by training, search, and protocol front ends."""

    def __init__(self, *, claim_draw: bool = True) -> None:
        # Self-play claims available draws to bound repeated trajectories. Protocol
        # adapters can disable this when the remote server still expects a move.
        self.claim_draw = claim_draw

    def get_init_board(self) -> chess.Board:
        return chess.Board()

    def replay_board_player(self, actions: np.ndarray, pos_idx: int) -> tuple[chess.Board, int]:
        """Replay ``actions[:pos_idx]`` from the start position; return board and side to move."""
        board = self.get_init_board()
        player = 1
        for t in range(pos_idx):
            player = self.push_action(board, player, int(actions[t]))
        return board, player

    def get_board_size(self) -> tuple[int, int, int]:
        return (8, 8, OBS_PLANES)

    def to_array(self, board: chess.Board) -> np.ndarray:
        return board_to_numpy(board)

    def get_action_size(self) -> int:
        return ACTION_SIZE

    def get_next_state(self, board: chess.Board, player: int, action: int) -> tuple[chess.Board, int]:
        """Execute action and return next (board, player).

        Queen promotions encoded without an explicit promotion type are completed
        automatically. Any other illegal action raises instead of silently executing a
        different move and corrupting replay data.
        """
        next_board = board.copy(stack=True)
        next_player = self.push_action(next_board, player, action)
        return next_board, next_player

    def get_next_search_state(self, board: chess.Board, player: int, action: int) -> tuple[chess.Board, int]:
        """Execute an MCTS edge while retaining exactly the rule-relevant history.

        Recurrent MCTS nodes are never encoded as temporal observations. Their
        move stack is needed only for repetition adjudication, which cannot cross
        the most recent zeroing move represented by ``halfmove_clock``.
        """
        next_board = board.copy(stack=board.halfmove_clock)
        next_player = self.push_action(next_board, player, action)
        if next_board.halfmove_clock == 0:
            next_board.clear_stack()
        return next_board, next_player

    def push_action(self, board: chess.Board, player: int, action: int) -> int:
        """Validate and apply an action to a board owned by the caller."""
        move = self._legal_move(board, player, action)
        board.push(move)
        return player_from_turn(board.turn)

    @staticmethod
    def _legal_move(board: chess.Board, player: int, action: int) -> chess.Move:
        _validate_player_turn(board, player)
        try:
            move: chess.Move | None = action_to_move(action)
        except ValueError:
            move = None
        if move is not None and not board.turn:
            move = mirror_move(move)

        if move is None or move not in board.legal_moves:
            if move is not None and move.promotion is None:
                promo_move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
                if promo_move in board.legal_moves:
                    move = promo_move

            if move is None or move not in board.legal_moves:
                raise ValueError(f"Illegal action {action} for position {board.fen()}")
        return move

    def get_valid_moves(self, board: chess.Board, player: int) -> np.ndarray:
        _validate_player_turn(board, player)
        acts = np.zeros(self.get_action_size(), dtype=np.float32)
        for move in board.legal_moves:
            canonical_move = move if player == 1 else mirror_move(move)
            acts[move_to_action(canonical_move)] = 1.0
        return acts

    def get_game_outcome(self, board: chess.Board, player: int) -> float | None:
        """Return the exact terminal value from ``player``'s perspective.

        Returns ``None`` while the game is ongoing, ``0.0`` for a draw, and ``+1`` or
        ``-1`` for a decisive result.
        """
        # In legal standard chess, the earliest claim by a move occurs after
        # seven reversible plies. Skipping the expensive history replay below
        # that exact bound preserves fifty-move and repetition semantics.
        claim_draw = self.claim_draw and board.halfmove_clock >= _MIN_THREEFOLD_CLAIM_HALFMOVES
        outcome = board.outcome(claim_draw=claim_draw)
        if outcome is None:
            return None
        if outcome.winner is None:
            return 0.0
        winner_int = player_from_turn(outcome.winner)
        return 1.0 if winner_int == player else -1.0

    def get_canonical_form(self, board: chess.Board, player: int) -> chess.Board:
        _validate_player_turn(board, player)
        if board.turn:
            return board
        return mirror_board(board)

    @staticmethod
    def display(board: chess.Board) -> None:
        logger.info("\n{}", board)
