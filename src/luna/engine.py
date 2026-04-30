"""Luna-Chess: main inference interface for the chess engine."""

from __future__ import annotations

import chess
import numpy as np
from loguru import logger

from .config import MCTSParams
from .game.chess_game import ChessGame, player_from_turn
from .game.state import LunaState
from .mcts import MCTS
from .network import LunaNetwork


class Luna:
    """Main interface for Luna Chess Engine (EZV2)."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.game = ChessGame()
        self.luna_eval = LunaNetwork(self.game)

        self.mcts_params = MCTSParams(num_mcts_sims=100, cpuct=1.25, dir_noise=False, dir_alpha=0.3)

        try:
            self.luna_eval.load_checkpoint("./temp/", "best.pth.tar")
            if self.verbose:
                logger.info("Loaded pre-trained model")
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            if self.verbose:
                logger.warning("Failed to load model ({}), using untrained network", exc)

        self.mcts = MCTS(self.game, self.luna_eval, self.mcts_params)
        self.board = chess.Board()

    def computer_move(self, state: LunaState) -> int:
        """Have Luna make a move on *state.board*."""
        if self.verbose:
            logger.info("Luna thinking about position {}", state.board.fen())

        current_player = player_from_turn(state.board.turn)
        canonical_board = self.game.get_canonical_form(state.board, current_player)

        action_probs = self.mcts.get_action_prob(canonical_board, temp=0)
        action = int(np.argmax(action_probs))

        next_board, _ = self.game.get_next_state(state.board, current_player, action)
        state.board = next_board
        self.board = next_board

        if self.verbose:
            logger.info("Luna played {}", state.board.peek())

        return action

    def is_game_over(self) -> bool:
        return self.board.is_game_over()
