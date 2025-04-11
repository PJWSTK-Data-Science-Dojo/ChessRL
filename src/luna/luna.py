"""
    Luna-Chess, main interface for the chess engine
"""

import chess
import numpy as np
import torch

from .NNet import Luna_Network
from .game.luna_game import ChessGame, who
from .mcts import MCTS
from .utils import dotdict
from .game.state import LunaState

class Luna:
    """Main interface for Luna Chess Engine"""
    
    def __init__(self, verbose=False) -> None:
        self.verbose = verbose
        
        # Game environment
        self.game = ChessGame()
        
        # Neural net
        self.luna_eval = Luna_Network(self.game)
        
        # Model hyperparameters
        self.args = dotdict({
            'numMCTSSims': 100,  # Number of MCTS simulations per move
            'cpuct': 1,          # Exploration constant
            'dir_noise': False,  # No Dirichlet noise when playing
            'dir_alpha': 1.4,    # Dirichlet alpha parameter
        })
        
        # Load pre-trained model if available
        try:
            self.luna_eval.load_checkpoint('./temp/', 'best.pth.tar')
            if self.verbose: print("[LUNA] Loaded pre-trained model")
        except:
            if self.verbose: print("[LUNA] Failed to load model, using untrained network")
        
        # Initialize MCTS search
        self.mcts = MCTS(self.game, self.luna_eval, self.args)
        
        # Current state
        self.board = chess.Board()
        self.board_state = LunaState()
    
    def computer_move(self, state, model):
        """Have Luna make a move"""
        if self.verbose: print(f"[LUNA THINKS] Luna is thinking about position {state.board.fen()}")
        
        # Get current player based on the board's turn
        current_player = who(state.board.turn)
        
        if self.verbose: print(f"[LUNA PLAYER] Current player is {'white' if current_player == 1 else 'black'}")
        
        # Get canonical form for current player's perspective
        canonical_board = self.game.getCanonicalForm(state.board, current_player)
        
        # Run MCTS and get action probabilities
        action_probs = self.mcts.getActionProb(canonical_board, temp=0)
        
        # Select best action
        action = np.argmax(action_probs)
        
        # Convert action to move and make it
        next_board, _ = self.game.getNextState(state.board, current_player, action)
        
        # Update the board state
        state.board = next_board
        self.board = next_board  # Make sure the main board state is updated too
        
        if self.verbose: print(f"[LUNA MOVES] Luna played {state.board.peek()}")
        
        return action

    
    def is_game_over(self):
        """Check if the game is over"""
        return self.board.is_game_over()