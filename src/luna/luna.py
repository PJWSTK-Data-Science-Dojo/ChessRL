import chess
import numpy as np
import torch
import logging
import os
from omegaconf import OmegaConf, DictConfig
import yaml
from collections import deque
from .NNet import Luna_Network
from .game.luna_game import ChessGame, who, _flip_action_index
from .mcts import MCTS

log = logging.getLogger(__name__)

# Define default config path relative to the expected current working directory (src)
CONFIG_PATH = 'src/config/config.yaml'

class Luna:
    """Main interface for Luna Chess Engine - Updated for OmegaConf and Inference Loading"""

    # Change __init__ signature to accept optional cfg
    def __init__(self, cfg: DictConfig | None = None, config_path: str = CONFIG_PATH, verbose: bool = False) -> None:
        super(Luna, self).__init__()
        self.verbose = verbose
        self.cfg = None # Will store OmegaConf config

        # --- Load Configuration ---
        try:
            if cfg is None:
                # Load config from file if not provided
                log.info(f"Attempting to load config from file: {os.path.abspath(config_path)}")
                full_config = OmegaConf.load(config_path)
                self.cfg = full_config # Store loaded config

                # Add CUDA availability check
                self.cfg.cuda = torch.cuda.is_available()

                # Note: Path resolution relative to project root should ideally happen *before* passing cfg to Luna
                # if Luna needs to access file paths specified in the config.
                # For the HTML wrapper, the path resolution is handled in the __main__ block.
                # For playground, path resolution is handled in its main block.
                # Assuming cfg paths are already resolved when passed or accessed here.

            else:
                # Use provided config (assumed to be already loaded, merged, and paths resolved)
                self.cfg = cfg # Store provided config
                log.info("Using provided configuration.")

            if self.verbose: log.info(f"Configuration for Luna Instance:\n{OmegaConf.to_yaml(self.cfg)}")

        except Exception as e:
            log.error(f"Error loading or processing configuration: {e}")
            raise e # Re-raise the exception if config loading fails

        # --- Game environment ---
        # Initialize Game with cfg (it extracts game section)
        self.game = ChessGame(cfg=self.cfg)

        # --- Neural net ---
        # Initialize Luna_Network with game and cfg (it extracts model section)
        self.luna_eval = Luna_Network(self.game, self.cfg)

        # --- Load inference model if specified in config.inference ---
        # Access loading config from cfg.inference
        if hasattr(self.cfg, 'inference') and self.cfg.inference.get('load_model', False):
            load_folder = self.cfg.inference.get('load_folder', None) # Use .get with default None
            load_file = self.cfg.inference.get('load_file', None)     # Use .get with default None

            if load_folder is not None and load_file is not None:
                try:
                    # Luna_Network.load_checkpoint uses its internal model_cfg for cuda device
                    self.luna_eval.load_checkpoint(load_folder, load_file)
                    if self.verbose: log.info(f"[LUNA] Loaded inference model from {load_folder}/{load_file} (Inference config)")
                except FileNotFoundError:
                    if self.verbose: log.warning(f"[LUNA] Failed to load inference model {load_folder}/{load_file}. Using untrained network.")
                except Exception as e:
                    log.error(f"[LUNA] Error loading inference model from {load_folder}/{load_file}: {e}", exc_info=True)
                    if self.verbose: log.warning(f"[LUNA] Error loading inference model. Using untrained network.")
            else:
                 log.warning("[LUNA] cfg.inference.load_model is True, but load_folder or load_file are not specified. Starting with an untrained network.")

        else:
            log.info("[LUNA] cfg.inference.load_model is False or missing. Starting with an untrained network for inference.")


        # --- Initialize MCTS search ---
        # Initialize MCTS with game, nnet, and cfg (it extracts mcts or inference section)
        # MCTS constructor now expects the specific config subset for MCTS params.
        # Pass the inference config subset which includes MCTS search parameters for inference.
        mcts_cfg_inference = OmegaConf.create({
             'numMCTSSims': self.cfg.inference.get('numMCTSSims', 100), # Default if missing
             'cpuct': self.cfg.inference.get('cpuct', 1.0),             # Default if missing
             'dir_noise': self.cfg.inference.get('dir_noise', False),   # Default if missing
             'dirichlet_alpha': self.cfg.inference.get('dirichlet_alpha', 0.3), # Default if missing
             'dirichlet_epsilon': self.cfg.inference.get('dirichlet_epsilon', 0.25) # Default if missing
        })
        self.mcts = MCTS(self.game, self.luna_eval, mcts_cfg_inference)


        # Current state (chess.Board)
        self.board = chess.Board()

        # Initialize history for the GUI's board instance
        # Use history_len from the game instance (which gets it from cfg)
        self.history = deque(maxlen=self.game.history_len)
        initial_array = self.game._board_to_feature_array(self.board)
        for _ in range(self.game.history_len):
            self.history.append(initial_array)

        # Store the GUI's chosen orientation (white or black)
        # This is not part of the core engine state, but used by the wrapper
        self.currentOrientation = 'white' # Default, will be set by /reset


    # computer_move method remains largely the same, it interacts with self.mcts and self.game
    # It uses temp=0 for deterministic moves, which is hardcoded and appropriate here.
    def computer_move(self, state=None, model=None): # Keep state/model as optional args for compatibility, but use self.board/history
        """Have Luna make a move"""
        # Use the internal self.board and self.history managed by this Luna instance

        if self.verbose: log.info(f"[LUNA THINKS] Luna is thinking about position {self.board.fen()}")

        current_player = who(self.board.turn)

        if self.verbose: log.info(f"[LUNA PLAYER] Current player is {'white' if current_player == 1 else 'black'}")

        # Run MCTS and get action probabilities from the root state (self.board, self.history)
        # MCTS instance uses its internal cfg for sims, cpuct, etc.
        # Pass actual board and history, temp=0 for deterministic inference
        action_probs = self.mcts.getActionProb(self.board, self.history, temp=0) # temp=0 for deterministic arena

        # Select best action (argmax on canonical action space probs)
        action_canonical = np.argmax(action_probs)

        # Translate canonical action index back to actual action index for the *current* board
        action_actual = action_canonical
        if current_player == -1: # If the actual board is Black's turn
             action_actual = self.game._flip_action_index(action_canonical) # Use game helper

        # --- Determine the exact legal move corresponding to action_actual ---
        temp_board = self.board.copy()
        found_legal_move = None
        try:
            # Iterate through all legal moves on a copy of the board to find the one
            # that maps to the selected action_actual index.
            for legal_m in temp_board.legal_moves:
                 # Use game.from_move to get the 64*64 index
                 if self.game.from_move(legal_m) == action_actual:
                      found_legal_move = legal_m
                      break

            if found_legal_move is None:
                 log.error(f"[LUNA MOVES] Luna selected action {action_actual} which does not correspond to any legal move on board {self.board.fen()}.")
                 return None # Indicate failure to move

            # Push the found legal move to the temp board to get the next state and the move UCI
            # Note: We push to temp_board just to get the UCI of the actual move made.
            # The main update to self.board and self.history happens manually below,
            # mirroring what getNextState does, to ensure history is consistent.
            temp_board.push(found_legal_move)
            luna_move_uci = temp_board.peek().uci() # Get the UCI of the last move pushed

            # Manually update Luna's internal board and history
            # This ensures history is updated correctly based on the specific move found.
            self.board.push(found_legal_move) # Apply the actual move to the main board instance

            # Update internal history: get array for the new board and append
            next_board_array = self.game._board_to_feature_array(self.board)
            # The deque self.history has maxlen set in __init__ based on game.history_len
            self.history.append(next_board_array) # Append the new state

            if self.verbose: log.info(f"[LUNA MOVES] Luna played {luna_move_uci}. New board: {self.board.fen()}")
            return luna_move_uci # Return the UCI string of the actual move made

        except Exception as e:
            log.error(f"[LUNA MOVES] Error determining Luna's move from action {action_actual} on board {self.board.fen()}: {e}", exc_info=True)
            raise e # Re-raise the exception


    def is_game_over(self):
        """Check if the game is over"""
        return self.board.is_game_over()

    def reset(self):
        """Reset the game board and history to the initial state."""
        self.board = chess.Board()
        # Re-initialize history deque with the correct maxlen
        self.history = deque(maxlen=self.game.history_len) # Use game.history_len
        initial_array = self.game._board_to_feature_array(self.board)
        for _ in range(self.game.history_len):
            self.history.append(initial_array)
        if self.verbose: log.info("[LUNA] Board and history reset.")