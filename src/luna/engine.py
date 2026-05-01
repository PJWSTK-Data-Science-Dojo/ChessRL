"""Luna-Chess: main inference interface for the chess engine."""

import chess
import numpy as np
from loguru import logger

from .config import EzV2LearnerConfig, MCTSParams
from .game.chess_game import ChessGame, player_from_turn
from .game.state import LunaState
from .mcts import MCTS
from .network import LunaNetwork


class Luna:
    """Main interface for Luna Chess Engine (EZV2)."""

    def __init__(
        self,
        verbose: bool = False,
        device: str = "cuda",
        num_mcts_sims: int = 100,
        checkpoint_dir: str = "./temp/",
        checkpoint_file: str = "best.pth.tar",
    ) -> None:
        """Initialize Luna chess engine.

        Args:
            verbose: Enable verbose logging
            device: Compute device (cuda, mps, or cpu)
            num_mcts_sims: Number of MCTS simulations per move
            checkpoint_dir: Directory containing model checkpoint
            checkpoint_file: Checkpoint filename
        """
        self.verbose = verbose
        self.game = ChessGame()

        # Configure learner with specified device
        learner_cfg = EzV2LearnerConfig(device=device, compile_inference=False)
        self.luna_eval = LunaNetwork(self.game, learner_cfg)

        self.mcts_params = MCTSParams(num_mcts_sims=num_mcts_sims, cpuct=1.25, dir_noise=False, dir_alpha=0.3)

        try:
            self.luna_eval.load_checkpoint(checkpoint_dir, checkpoint_file)
            if self.verbose:
                logger.info(f"Loaded pre-trained model from {checkpoint_dir}/{checkpoint_file} on {device}")
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
