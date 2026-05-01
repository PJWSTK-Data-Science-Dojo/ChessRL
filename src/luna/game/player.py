"""Chess Players."""

import random

import chess
import numpy as np
from loguru import logger

from .chess_game import ChessGame, mirror_move, move_to_action, player_from_turn


def move_from_uci(board: chess.Board, uci: str) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        logger.warning("expected an UCI move")
        return None
    if move not in board.legal_moves:
        logger.warning("expected a valid move")
        return None
    return move


class RandomPlayer:
    """Always plays random legal moves."""

    def __init__(self, game: ChessGame) -> None:
        self.game = game

    def play(self, board: chess.Board) -> int:
        valids = self.game.get_valid_moves(board, player_from_turn(board.turn))
        moves = np.argwhere(valids == 1)
        return int(random.choice(moves)[0])


class HumanChessPlayer:
    """Interactive human player via stdin."""

    def play(self, board: chess.Board) -> int:
        mboard = board
        if not board.turn:
            mboard = board.mirror()

        legal_uci = ", ".join(m.uci() for m in mboard.legal_moves)
        logger.info("Valid moves: {}", legal_uci)
        human_move = input()

        parsed = move_from_uci(mboard, human_move.strip())
        if parsed is None:
            ex = random.choice(list(mboard.legal_moves)).uci()
            logger.info("try again, e.g., {}", ex)
            return self.play(board)

        move = parsed
        if not board.turn:
            move = mirror_move(move)
        return move_to_action(move)


class StockFishPlayer:
    """Stockfish wrapper (requires `stockfish` package and binary)."""

    def __init__(self, elo: int = 1000, skill_level: int = 10, depth: int = 10, think_time: int = 30) -> None:
        from stockfish import Stockfish

        self.stockfish = Stockfish(parameters={"Threads": 2, "Minimum Thinking Time": think_time})
        self.stockfish.set_elo_rating(elo)
        self.stockfish.set_skill_level(skill_level)
        self.stockfish.set_depth(depth)

    def play(self, board: chess.Board) -> int:
        self.stockfish.set_fen_position(board.fen())
        uci_move = self.stockfish.get_best_move()
        if uci_move is None:
            legal = list(board.legal_moves)
            if not legal:
                raise RuntimeError("Stockfish returned None and no legal moves exist")
            logger.warning("Stockfish returned None, falling back to first legal move")
            return move_to_action(legal[0])
        move = move_from_uci(board, uci_move.strip())
        if move is None:
            raise ValueError(f"Stockfish suggested illegal move {uci_move} in position {board.fen()}")
        return move_to_action(move)
