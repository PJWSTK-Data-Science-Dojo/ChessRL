import logging
from tqdm import tqdm
import numpy as np
import time
import chess
import os
import torch
import multiprocessing as mp
from collections import deque
from omegaconf import DictConfig, OmegaConf # Import OmegaConf types

# Luna
from .luna_game import ChessGame # Assuming correct class name
from ..NNet import Luna_Network # Corrected relative import
from ..mcts import MCTS # Corrected relative import
from .luna_game import who, _flip_action_index # Import helpers

log = logging.getLogger(__name__)

# Define a small epsilon for floating point comparisons, consistent with ChessGame
EPS = 1e-8


# --- Worker function for parallel arena games ---
# This function runs a single game in a separate process.
# It initializes necessary components (Game, NNets, MCTS, Arena instance)
# using the configuration passed to it.
# Accept cfg: DictConfig directly as part of the args_tuple.
def run_arena_game_worker(args_tuple):
    """
    Worker function to play a single arena game between two models.
    Takes a tuple: (game_id, cfg, pnet_checkpoint_path, nnet_checkpoint_path, player1_is_white)
    Returns the game outcome relative to Player 1 (Prev Model) and player1_is_white flag.
    Returns None if loading fails.
    """
    game_id, cfg, pnet_checkpoint_path, nnet_checkpoint_path, player1_is_white = args_tuple
    log.debug(f"Arena Worker {game_id} starting. P1 (Prev) is White: {player1_is_white}. Config type: {type(cfg)}")

    try:
        # --- Initialize components within the worker process ---
        # Each worker needs its own game instance initialized with cfg
        g = ChessGame(cfg=cfg)

        # Initialize NNet instances within the worker using appropriate config subsets
        # model_cfg is needed for NNet architecture and device setup
        model_cfg_worker = OmegaConf.create({
            'model': cfg.model,
            'cuda': cfg.cuda # Add cuda flag to the model_cfg subset
        })
        pnet_worker = Luna_Network(g, model_cfg_worker) # Pass game and model_cfg subset
        nnet_worker = Luna_Network(g, model_cfg_worker) # Pass game and model_cfg subset

        # Determine device based on cfg.cuda
        map_location = torch.device('cuda') if cfg.cuda and torch.cuda.is_available() else torch.device('cpu')

        # Load Previous Model (pnet) from checkpoint path
        if os.path.exists(pnet_checkpoint_path):
             try:
                 # Load checkpoint (weights_only=False is good practice if other states might be saved)
                 p_checkpoint = torch.load(pnet_checkpoint_path, map_location=map_location, weights_only=False)
                 if 'state_dict' in p_checkpoint:
                     pnet_worker.nnet.load_state_dict(p_checkpoint['state_dict'], strict=True)
                 else:
                      log.warning(f"Arena Worker {game_id}: Prev Checkpoint missing 'state_dict': {pnet_checkpoint_path}. Cannot load.")
                      return None # Indicate failure if model state_dict is missing
             except Exception as load_err:
                  log.error(f"Arena Worker {game_id}: Failed to load Prev Checkpoint {pnet_checkpoint_path}: {load_err}", exc_info=True)
                  return None # Indicate failure on load error
        else:
             log.error(f"Arena Worker {game_id}: Prev Checkpoint not found at {pnet_checkpoint_path}. Cannot load.")
             return None # Indicate failure if file not found


        # Load New Model (nnet) from checkpoint path
        if os.path.exists(nnet_checkpoint_path):
             try:
                 n_checkpoint = torch.load(nnet_checkpoint_path, map_location=map_location, weights_only=False)
                 if 'state_dict' in n_checkpoint:
                     nnet_worker.nnet.load_state_dict(n_checkpoint['state_dict'], strict=True)
                 else:
                      log.warning(f"Arena Worker {game_id}: New Checkpoint missing 'state_dict': {nnet_checkpoint_path}. Cannot load.")
                      return None # Indicate failure
             except Exception as load_err:
                  log.error(f"Arena Worker {game_id}: Failed to load New Checkpoint {nnet_checkpoint_path}: {load_err}", exc_info=True)
                  return None # Indicate failure on load error
        else:
             log.error(f"Arena Worker {game_id}: New Checkpoint not found at {nnet_checkpoint_path}. Cannot load.")
             return None # Indicate failure if file not found


        # Initialize MCTS instances with game, nnet, and appropriate mcts_cfg subset
        # For arena games, we typically use inference MCTS parameters (fewer sims, temp=0 usually)
        # Create an mcts_cfg subset using inference parameters from cfg
        mcts_cfg_inference = OmegaConf.create({
            'numMCTSSims': cfg.inference.numMCTSSims, # Use inference sims
            'cpuct': cfg.inference.cpuct,           # Use inference cpuct
            'dir_noise': cfg.inference.dir_noise,     # Use inference dir_noise (usually False)
            # Use inference specific alpha/epsilon if available, otherwise fallback to mcts
            'dirichlet_alpha': cfg.inference.get('dirichlet_alpha', cfg.mcts.dirichlet_alpha),
            'dirichlet_epsilon': cfg.inference.get('dirichlet_epsilon', cfg.mcts.dirichlet_epsilon)
             # history_len is obtained from game instance
        })

        pmcts_worker = MCTS(g, pnet_worker, mcts_cfg_inference) # Pass game, nnet_worker, and mcts_cfg_inference
        nmcts_worker = MCTS(g, nnet_worker, mcts_cfg_inference) # Pass game, nnet_worker, and mcts_cfg_inference


        # Define the player functions that wrap the worker's MCTS instances
        # These functions take (board, history) and return the actual action index for the board.
        def create_worker_arena_player_func(mcts_instance_worker):
             def arena_player_worker(board: chess.Board, history: deque) -> int:
                  current_player = who(board.turn)
                  try:
                      # Call MCTS.getActionProb with actual board, history, and temp=0 (deterministic)
                      pi_canonical = mcts_instance_worker.getActionProb(board, history, temp=0)
                  except Exception as e:
                       log.error(f"Arena Worker {game_id}: Error in MCTS.getActionProb for player {current_player}: {e}", exc_info=True)
                       # Fallback: Return a uniform random move over valid actions on error
                       valids_actual = g.getValidMoves(board, current_player)
                       valid_indices = np.where(valids_actual == 1)[0]
                       if valid_indices.size > 0:
                           return int(np.random.choice(valid_indices))
                       else:
                           log.error(f"Arena Worker {game_id}: No valid moves found for board {board.fen()} in fallback.")
                           return -1 # Indicate failure to move


                  pi_array = np.array(pi_canonical)
                  pi_sum = np.sum(pi_array)

                  if pi_sum < EPS: # Use defined EPS
                       log.warning(f"Arena Worker {game_id}: Policy sum ({pi_sum}) zero. Fallback to uniform over valids for board {board.fen()}.")
                       valids_actual = g.getValidMoves(board, current_player)
                       valid_indices = np.where(valids_actual == 1)[0]
                       if valid_indices.size > 0:
                           return int(np.random.choice(valid_indices))
                       else:
                            log.error(f"Arena Worker {game_id}: No valid moves found for board {board.fen()} during fallback.")
                            return -1 # Indicate failure


                  # Select the action index with the highest probability from the canonical policy
                  action_canonical = np.argmax(pi_array)

                  # Translate canonical action index back to actual action index for the current board/player
                  action_actual = action_canonical
                  if current_player == -1: # If the actual board is Black's turn
                       action_actual = _flip_action_index(action_canonical) # Use game helper

                  return action_actual # Return the action index for the actual board

             return arena_player_worker


        # Create the player functions for this specific game instance
        player1_arena_func = create_worker_arena_player_func(pmcts_worker) # Player 1 (Prev)
        player2_arena_func = create_worker_arena_player_func(nmcts_worker) # Player 2 (New)


        # Determine which player function plays as White and which as Black for *this* game
        if player1_is_white:
            white_player_func = player1_arena_func # P1 (Prev) is White
            black_player_func = player2_arena_func # P2 (New) is Black
        else:
            white_player_func = player2_arena_func # P2 (New) is White
            black_player_func = player1_arena_func # P1 (Prev) is Black


        # Create a local Arena instance to play the game (it runs sequentially)
        # Arena expects player1 func to be White, player2 func to be Black
        local_arena = Arena(white_player_func, black_player_func, g)

        # Play the game. playGame returns outcome relative to White (+1).
        # verbose is False for parallel workers.
        game_outcome_relative_to_white = local_arena.playGame(verbose=False)

        # Return the outcome relative to White and the starting player information
        # The main process will use this to aggregate results relative to Player 1 (Prev).
        return (game_outcome_relative_to_white, player1_is_white)

    except Exception as e:
        log.error(f"Arena Worker {game_id}: Unexpected error during game execution: {e}", exc_info=True)
        # Return None to indicate failure for this specific game
        return None

# -------------------------------------


class Arena():
    """Arena class - Utility Class for playing a single game sequentially."""

    def __init__(self, player1, player2, game: ChessGame):
        """
        Input:
            player1, player2: Functions that take (board: chess.Board, history: deque),
                              return the chosen action index (int 0-4095) for the actual board.
            game: ChessGame object (already initialized with history_len).
        """
        super(Arena, self).__init__()
        self.player1 = player1 # Function for player 1 (plays as White in playGame)
        self.player2 = player2 # Function for player 2 (plays as Black in playGame)
        self.game = game # Store the game instance


    def playGame(self, verbose=False) -> float:
        """
        Executes one game between player1 (White) and player2 (Black).
        Returns outcome relative to White (+1 for White win, -1 for Black win, 1e-4 for draw).
        Does NOT use multiprocessing.
        """
        # player_funcs list indexes: 0 -> player2 (Black), 1 -> None (unused), 2 -> player1 (White)
        player_funcs = [self.player2, None, self.player1]
        curPlayer = 1 # Game starts with White (+1)
        # Get initial board AND history from the game instance
        board, history = self.game.getInitBoard() # CORRECTED: Get history here
        it = 0
        move_times = []

        while True:
            it += 1

            if verbose:
                print(f"\n--- Turn {it} ---")
                try:
                   # Display board from the current player's perspective
                   # Pass board to game.display directly
                   if curPlayer == 1:
                        self.game.display(board)
                   else: # Display mirrored board for Black's perspective
                        self.game.display(board.mirror()) # Mirror for display only
                except Exception as e:
                   print(f"Error displaying board: {e}")
                   print(board.fen() if isinstance(board, chess.Board) else "Invalid board object")

                print(f"Current Player: {'White' if curPlayer == 1 else 'Black'}")


            # Get action from the correct player function, passing the actual board AND history
            player_func = player_funcs[curPlayer + 1] # Index 0 for P2 (Black), 2 for P1 (White)
            start_time = time.time()
            # Pass the actual board and current history to the player function
            action = player_func(board, history) # Player function returns action index (0-4095)
            move_times.append(time.time() - start_time)

            # Validate action index against the actual legal moves for the current board/player
            # The action index `action` is expected to be in the 0-4095 range corresponding to the
            # move that should be made on the *actual* board.
            is_valid_action = False
            if 0 <= action < self.game.getActionSize():
                 # Check if the action index corresponds to any legal move (including promotions handled by getValidMoves mapping)
                 # Get valids vector for actual board using game instance
                 actual_valids_vector = self.game.getValidMoves(board, curPlayer)
                 if actual_valids_vector[action] == 1:
                      is_valid_action = True
                 else:
                     # Action index is valid in range, but not a legal move from this board state.
                     log.warning(f'Arena Warning: Player {curPlayer} ({("White" if curPlayer==1 else "Black")}) returned action index {action} which is NOT legal for actual board {board.fen()}. Action will be treated as invalid.')


            if not is_valid_action:
                log.error(f'Arena Error: Player {curPlayer} ({("White" if curPlayer==1 else "Black")}) returned invalid action index {action} for actual board {board.fen()}!')
                # Penalize the player returning invalid move.
                return -curPlayer # Opponent wins due to invalid move by current player


            # Store the board state *before* calling getNextState for illegal move double-check
            previous_board_fen = board.fen()

            # Get next state (board, player, history) using the *actual* board and player
            # Pass the TRANSLATED action index (action) and the *current* history
            # getNextState will return the updated board, next player, and updated history (or original if illegal)
            board, nextPlayer, history = self.game.getNextState(board, curPlayer, action, history) # Pass current history, update history

            # Check if getNextState indicated an illegal move (should be caught by validation above, but double-check)
            # This is a critical error if validation passed but getNextState signals illegal.
            if nextPlayer == curPlayer and board.fen() == previous_board_fen:
                 log.error(f"Arena Error: getNextState returned original state after move {action} on board {previous_board_fen}, indicating illegal move AFTER validation! This is an internal inconsistency.")
                 # Critical inconsistency. Treat as an error for the current player.
                 return -curPlayer # Opponent wins


            # Update current player and history based on getNextState return
            curPlayer = nextPlayer
            # History is already updated by the assignment above: `history = next_history` (although variable name is the same)


            # Check game end state relative to player 1 (White) using game instance
            game_ended_result_p1 = self.game.getGameEnded(board, 1)
            if game_ended_result_p1 != 0:
                if verbose:
                    print("\n--- Game Over! ---")
                    try:
                       # Display final board (usually White's perspective implicitly)
                       self.game.display(board)
                    except Exception as e:
                       print(f"Error displaying board: {e}")
                       print(board.fen() if isinstance(board, chess.Board) else "Invalid board object")

                    # Use a small tolerance for draw check
                    if abs(game_ended_result_p1) > EPS: # Win/loss (relative to White)
                        winner_msg = "White" if game_ended_result_p1 > 0 else "Black"
                        print(f"Result: {winner_msg} wins (relative to White)")
                    else:
                        print("Result: Draw")
                    avg_move_time = np.mean(move_times) if move_times else 0
                    print(f"Average move time: {avg_move_time:.3f}s")

                # Return result relative to player 1 (White)
                return game_ended_result_p1

    # Removed the plural playGames method from Arena class.
    # The parallel execution logic for multiple games is handled in Coach.learn.