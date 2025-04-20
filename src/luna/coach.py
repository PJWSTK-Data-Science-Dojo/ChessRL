import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import numpy as np
from tqdm import tqdm
import torch
import wandb
import multiprocessing as mp
import time
import re
from omegaconf import DictConfig, OmegaConf # Import OmegaConf types

# Import necessary components
from .game.arena import Arena, run_arena_game_worker # Import Arena class and the worker function
from .mcts import MCTS
from .game.luna_game import ChessGame, who, _flip_action_index
from .NNet import Luna_Network
from .utils import AverageMeter
# Add path utility import
from .utils import find_project_root, get_absolute_path # Assuming utils.py is in src/luna/utils.py relative to src/

log = logging.getLogger(__name__)
EPS = 1e-8

# --- Worker function for multiprocessing (Self-Play) ---
# This function runs a single self-play episode in a separate process.
# It initializes necessary components (Game, NNet, MCTS instance)
# using the configuration passed to it.
# Accept cfg: DictConfig directly as part of the args_tuple.
def run_episode_worker(args_tuple):
    """
    Worker function to execute one episode of self-play with history.
    Takes a tuple: (iteration, worker_id, cfg, model_checkpoint_path)
    Returns a list of training examples collected during the episode.
    Returns an empty list on failure.
    """
    iteration, worker_id, cfg, model_checkpoint_path = args_tuple # Accept cfg

    # trainExamples is a list of tuples: (canonical_NN_input, pi_canonical_target, value_canonical_target, canonical_valid_mask)
    trainExamples = []

    try:
        # --- Initialize components within the worker process ---
        # Each worker needs its own game instance initialized with cfg
        g = ChessGame(cfg=cfg)

        # Initialize NNet worker instance with appropriate config subset
        # model_cfg is needed for NNet architecture and device setup
        model_cfg_worker = OmegaConf.create({
            'model': cfg.model,
            'cuda': cfg.cuda # Add cuda flag to the model_cfg subset
        })
        # NNet constructor expects game and model_cfg subset
        nnet_worker = Luna_Network(g, model_cfg_worker) # Pass game and model_cfg subset

        # Load model weights from checkpoint path if provided and exists
        # This worker does not need optimizer/scheduler state, only weights
        if model_checkpoint_path and os.path.exists(model_checkpoint_path):
             # Determine device based on cfg.cuda
             map_location = torch.device('cuda') if cfg.cuda and torch.cuda.is_available() else torch.device('cpu')
             try:
                 # Load checkpoint (weights_only=False is good practice if other states might be saved)
                 checkpoint = torch.load(model_checkpoint_path, map_location=map_location, weights_only=False)
                 if 'state_dict' in checkpoint:
                     # Use strict=True for safety if model architecture is fixed
                     nnet_worker.nnet.load_state_dict(checkpoint['state_dict'], strict=True)
                 else:
                      log.warning(f"Worker {worker_id}: Checkpoint '{model_checkpoint_path}' missing 'state_dict'. Cannot load model.")
             except Exception as load_err:
                  log.error(f"Worker {worker_id} failed to load checkpoint {model_checkpoint_path}: {load_err}", exc_info=True)
                  # Log error but continue with untrained network if loading fails?
                  # Or terminate episode? Let's terminate the episode to avoid bad examples.
                  return [] # Indicate failure to load model weights


        # Initialize MCTS worker instance with game, nnet, and appropriate config subset
        # For self-play, use the MCTS parameters specified in the 'mcts' section of the config
        mcts_cfg_selfplay_worker = OmegaConf.create({
            'numMCTSSims': cfg.mcts.numMCTSSims, # Use self-play sims
            'cpuct': cfg.mcts.cpuct,           # Use self-play cpuct
            'dir_noise': cfg.mcts.dir_noise,     # Use self-play dir_noise
            'dirichlet_alpha': cfg.mcts.dirichlet_alpha, # Use self-play alpha
            'dirichlet_epsilon': cfg.mcts.dirichlet_epsilon # Use self-play epsilon
             # history_len is obtained from game instance
        })
        # MCTS constructor expects game, nnet, and mcts_cfg subset
        mcts_worker = MCTS(g, nnet_worker, mcts_cfg_selfplay_worker) # Pass game, nnet_worker, and mcts_cfg subset


        # --- Run the self-play episode ---
        board, current_history = g.getInitBoard() # Get initial board and history
        curPlayer = who(board.turn) # Current player (+1 for White, -1 for Black)
        episodeStep = 0 # Current move number in the episode (ply)

        # Access tempThreshold from cfg
        temp_threshold = cfg.mcts.tempThreshold # Temperature threshold for action selection

        while True:
            episodeStep += 1 # Increment move counter

            # Check game end state relative to Player 1 (White) using game instance
            game_outcome_p1 = g.getGameEnded(board, 1) # Outcome relative to Player 1
            if game_outcome_p1 != 0:
                # Game is over. Calculate final value targets (z) for collected examples.
                # Value is relative to the player whose turn it was at that specific state
                final_examples = []
                # Iterate through the temporarily stored example data (input, pi, value_placeholder, valids)
                for ex in trainExamples:
                    canonical_nn_input, pi_canonical_target, _, canonical_valid_mask = ex # Unpack
                    # The value target 'z' should be relative to Player 1.
                    # The outcome 'game_outcome_p1' is already relative to Player 1.
                    # Value is +1 for Player 1 win, -1 for Player 1 loss, 1e-4 for draw (relative to P1).
                    value_canonical_target = game_outcome_p1 # Value target is the game outcome relative to Player 1
                    final_examples.append((canonical_nn_input, pi_canonical_target, value_canonical_target, canonical_valid_mask)) # Store complete tuple
                return final_examples # Return list of (input, pi, z, valids) tuples


            # Determine temperature for action selection (higher temp early for exploration)
            temp = int(episodeStep < temp_threshold) # temp=1 if episodeStep < threshold, else temp=0 (deterministic)

            # Get action probabilities from MCTS for the current state (board, history)
            # MCTS.getActionProb expects actual board and history at the root.
            # MCTS instance uses its internal cfg for number of sims and cpuct.
            # MCTS returns a policy vector (numpy array) over the 4096 *canonical* actions.
            pi_canonical_mcts = mcts_worker.getActionProb(board, current_history, temp=temp)


            # Prepare data for the training example tuple (s, pi, v, valids)
            # s: NN input array (stacked canonical history)
            # pi: MCTS policy (numpy array, canonical)
            # v: Value target (placeholder for now, determined at end of episode)
            # valids: Valid moves mask (numpy array, canonical)

            # Get NN input for the current state (stacked canonical history)
            # g.getCanonicalHistory needs actual history and current player
            canonical_nn_input = g.toArray(g.getCanonicalHistory(current_history, curPlayer))

            # Get canonical valid moves mask for the current board state
            # g.getCanonicalForm needs actual board and current player
            canonical_board_for_valids = g.getCanonicalForm(board, curPlayer)
            canonical_valid_mask = g.getValidMoves(canonical_board_for_valids, 1) # Get valids for player 1 perspective on canonical board


            # Store original example data temporarily. Value target (z) is None for now.
            # Tuple: (canonical_NN_input_array, pi_canonical_target, value_placeholder, canonical_valid_mask)
            pi_array = np.array(pi_canonical_mcts) # Ensure pi is a numpy array
            trainExamples.append([canonical_nn_input, pi_array, None, canonical_valid_mask])


            # Generate symmetric example data (vertical flip symmetry)
            # Symmetry applies to the board representation (NN input) and the policy target (pi).
            # It also applies to the valid moves mask.

            # 1. Create flipped canonical history stack for the NN input
            flipped_canonical_history = deque(maxlen=g.history_len)
            # g.getCanonicalHistory gets the history relative to curPlayer, resulting in a canonical stack.
            canonical_history_at_step = g.getCanonicalHistory(current_history, curPlayer)
            # Iterate through each frame in the canonical history stack and flip it vertically
            for frame in canonical_history_at_step:
                 flipped_canonical_history.append(g._flip_feature_array(frame)) # Use game helper to flip single frame array
            flipped_canonical_nn_input = g.toArray(flipped_canonical_history) # Stack the flipped history frames


            # 2. Flip the canonical MCTS policy (pi_array is for the canonical board)
            pi_array_flipped = np.zeros_like(pi_array)
            # Iterate through all possible action indices (0-4095) in the canonical space
            for action_index in range(g.getActionSize()):
                # Calculate the corresponding action index on the vertically flipped board
                flipped_action = _flip_action_index(action_index) # Use game helper to flip action index
                # The probability of the flipped action on the flipped board is the probability
                # of the original action on the original board.
                pi_array_flipped[flipped_action] = pi_array[action_index]

            # Note: If the sum is slightly off after flipping due to floating point errors,
            # re-normalizing pi_array_flipped might be needed, but usually not critical
            # if the original pi_array was properly normalized.

            # 3. Flip the canonical valid moves mask
            flipped_canonical_valid_mask = np.zeros_like(canonical_valid_mask)
            # Iterate through all possible action indices in the canonical space
            for action_index in range(g.getActionSize()):
                # Calculate the corresponding action index on the vertically flipped board
                flipped_action = _flip_action_index(action_index) # Use game helper
                # If the original action was valid, the flipped action is valid on the flipped board.
                flipped_canonical_valid_mask[flipped_action] = canonical_valid_mask[action_index]


            # Store symmetric example data temporarily. Value target (z) is None for now.
            # The value target (z) is relative to Player 1.
            # The symmetric canonical input represents the board from Player 1's perspective,
            # but the effective player who would make the move *from this flipped state* is -curPlayer relative to the original board's player.
            # However, the NN input itself is ALWAYS from Player 1's perspective (the canonical form).
            # The value target is the game outcome relative to Player 1, regardless of symmetry.
            # The training example tuple is (s, pi, v) where v is relative to Player 1.
            # So the value target for the symmetric example is the SAME as the original example.

            # Add symmetric example data: (flipped_canonical_input, flipped_canonical_pi_target, value_placeholder, flipped_canonical_valid_mask)
            trainExamples.append([flipped_canonical_nn_input, pi_array_flipped, None, flipped_canonical_valid_mask])


            # --- Choose action to take based on MCTS policy ---
            # Select action index from MCTS policy (canonical perspective, potentially with temperature/noise).
            # This policy `pi_array` is for the canonical board.
            # We need to choose an action from the *actual* legal moves on the current board.
            # The MCTS policy should, ideally, only have probability on actions that correspond to legal moves.
            # The canonical_valid_mask identifies valid actions in the canonical space.
            # We should sample from the MCTS policy *restricted to* the canonical valid actions.

            pi_mcts_array_normalized = pi_array / np.sum(pi_array) # Ensure normalization
            valid_action_indices = np.where(canonical_valid_mask == 1)[0] # Indices of valid actions in the canonical space

            if valid_action_indices.size > 0:
                 # Filter policy to only valid canonical actions and re-normalize for sampling
                 pi_valid_only = pi_mcts_array_normalized[valid_action_indices]
                 sum_pi_valid_only = np.sum(pi_valid_only)

                 if sum_pi_valid_only < EPS: # Handle case where MCTS policy puts zero weight on all valid moves
                      log.warning(f"Worker {worker_id}: Validated policy sum zero for canonical board {canonical_board_for_valids.fen()}. Falling back to uniform sampling over valid actions.")
                      # Fallback to uniform sampling over valid actions
                      pi_valid_only_normalized = np.ones_like(pi_valid_only) / valid_action_indices.size
                 else:
                      pi_valid_only_normalized = pi_valid_only / sum_pi_valid_only # Re-normalize over valid actions

                 # Choose action index from the valid canonical actions based on the normalized policy
                 action_canonical_chosen = np.random.choice(valid_action_indices, p=pi_valid_only_normalized)
            else:
                 log.error(f"Worker {worker_id}: No valid canonical actions found for board {canonical_board_for_valids.fen()}! MCTS policy sum: {np.sum(pi_array)}. Ending episode due to no valid moves.")
                 # Handle error: return examples collected so far with neutral value
                 log.warning(f"Worker {worker_id}: Assigning neutral value (0.0) to {len(trainExamples)} examples collected before no valid actions.")
                 final_examples_partial = []
                 for ex in trainExamples:
                     input_arr, pi_target, _, valids_mask = ex
                     final_examples_partial.append((input_arr, pi_target, 0.0, valids_mask))
                 return final_examples_partial


            # Translate the chosen canonical action index back to the actual action index for the current board
            action_actual = action_canonical_chosen # Start with the canonical index
            if curPlayer == -1: # If the actual turn is Black
                 action_actual = _flip_action_index(action_canonical_chosen) # Flip the action index


            # --- Get next state by applying the chosen action ---
            previous_board_fen = board.fen() # Store current board FEN before the move

            # Use game.getNextState to apply the action and get the new state and history
            # Pass the actual board, actual current player, chosen actual action index, and current history
            board, nextPlayer, next_history = g.getNextState(board, curPlayer, action_actual, current_history)

            # --- Check for illegal move ---
            # getNextState is designed to return the original state if the move is illegal.
            # Although MCTS selected a valid action index, getNextState might still
            # flag it as illegal if there's a subtle issue (e.g., promotion type mismatch, although our mapping handles Queen).
            # This check acts as a safety net and should ideally never trigger if MCTS respects valid moves.
            if nextPlayer == curPlayer and board.fen() == previous_board_fen:
                log.error(f"Worker {worker_id}: getNextState returned original state after action {action_actual} on actual board {previous_board_fen}. This action was selected by MCTS from canonical state {g.stringRepresentation(canonical_board_for_valids)}! Illegal move detected. Ending episode path.")
                # If an illegal move was chosen despite MCTS/valids, terminate this path.
                # Return examples collected so far with a neutral value.
                log.warning(f"Worker {worker_id}: Assigning neutral value (0.0) to {len(trainExamples)} examples collected before illegal move.")
                final_examples_partial = []
                for ex in trainExamples:
                    input_arr, pi_target, _, valids_mask = ex
                    final_examples_partial.append((input_arr, pi_target, 0.0, valids_mask))
                return final_examples_partial


            # If getNextState was successful, nextPlayer should be the opposite of curPlayer
            assert nextPlayer == -curPlayer, f"Worker {worker_id}: Player did not flip after move! Current: {curPlayer}, Next: {nextPlayer}. Board: {board.fen()}"

            # Update current player and history for the next loop iteration
            curPlayer = nextPlayer
            current_history = next_history # Use the history returned by getNextState

    # --- Error Handling for unexpected exceptions during episode ---
    except Exception as e:
        log.error(f"Error in worker {worker_id} for iteration {iteration}: {e}", exc_info=True)
        # If any unexpected exception occurs, terminate the episode.
        # Return examples collected so far with a neutral value.
        log.warning(f"Worker {worker_id}: Assigning neutral value (0.0) to {len(trainExamples)} examples collected before unexpected error.")
        final_examples_partial = []
        for ex in trainExamples:
            input_arr, pi_target, _, valids_mask = ex
            final_examples_partial.append((input_arr, pi_target, 0.0, valids_mask))
        return final_examples_partial

# -------------------------------------


class Coach():
    """Coach class - Manages the training loop (Self-Play, Training, Arena)."""

    # Coach __init__ accepts the initialized game instance and the full configuration object.
    def __init__(self, game: ChessGame, cfg: DictConfig, wandb_enabled: bool = False):
        super(Coach, self).__init__()
        self.game = game # Store the game instance (already initialized with history_len from cfg)
        self.cfg = cfg # Store the full configuration object

        # --- Initialize NNet instances ---
        # nnet is the network currently being trained.
        # pnet is the previous best network, used for arena comparison.
        # Both are initialized with the same architecture (from model_cfg) and device (cuda flag).
        # Create a model_cfg subset to pass to NNet constructor
        model_cfg = OmegaConf.create({
            'model': cfg.model,
            'cuda': cfg.cuda # Add cuda flag to the model_cfg
        })
        # NNet constructor expects game and model_cfg subset
        self.nnet = Luna_Network(self.game, model_cfg)
        self.pnet = Luna_Network(self.game, model_cfg) # pnet initialized with same architecture


        # --- Setup Optimizers/Schedulers for Training Instances ---
        # The nnet instance (the one being trained) needs optimizer and scheduler.
        # Call the setup method on nnet, passing the relevant config subsets.
        # pnet does NOT need optimizer/scheduler as it's only used for prediction (evaluation).
        self.nnet.setup_optimizer_scheduler(cfg.optimizer, cfg.training) # Pass optimizer and training config subsets


        # trainExamplesHistory stores collected self-play examples across iterations.
        # It is a list of deques. Each deque contains examples from one self-play iteration.
        # Each example tuple: (canonical_NN_input, pi_canonical_target, value_canonical_target, canonical_valid_mask)
        self.trainExamplesHistory = []
        # skipFirstSelfPlay flag controls whether to run self-play in the first iteration.
        # This is typically True if examples are loaded.
        self.skipFirstSelfPlay = False
        # Flag to indicate if WandB logging is enabled
        self.wandb_enabled = wandb_enabled


    def learn(self) -> None:
        """Main training loop overseeing self-play, training, and arena comparison."""

        # --- Determine Starting Iteration ---
        # Based on whether examples or a model are being loaded.
        # Access loading config from cfg
        load_examples = self.cfg.loading.load_examples
        load_model = self.cfg.loading.load_model
        # Use resolved paths from cfg (handled by Luna.__init__ if called first, or should be handled by main/playground)
        # Let's assume paths in cfg are already absolute or relative to project root as setup in main/playground.
        load_folder = self.cfg.loading.load_folder
        load_file = self.cfg.loading.load_file
        load_examples_folder = self.cfg.loading.load_examples_folder # Folder to load examples FROM

        start_iter = 1 # Default starting iteration

        # If loading examples, find the latest examples file and determine the iteration
        if load_examples:
             latest_examples_file = self._find_latest_examples_file(load_examples_folder) # Use load_examples_folder
             if latest_examples_file:
                  # Parse the iteration number from the filename
                  match = re.search(r'_iter_(\d+)\.pkl$', latest_examples_file)
                  if match:
                      start_iter = int(match.group(1)) + 1 # Start from the iteration AFTER the loaded ones
                      log.info(f"Resuming training from iteration {start_iter} based on loaded examples file: {latest_examples_file}")
                      # Load the training examples history
                      self.loadTrainExamples() # loadTrainExamples uses self.cfg.loading.load_examples_folder internally
                  else:
                       log.warning(f"Could not parse iteration from example file name {latest_examples_file}. Starting from iteration 1.")
                       start_iter = 1 # Fallback to starting from 1 if parsing fails
                       self.trainExamplesHistory = [] # Ensure history is empty if starting fresh
                       self.skipFirstSelfPlay = False # Must perform self-play if no examples loaded
             else:
                  log.warning(f"load_examples=True but no examples file found in {load_examples_folder}. Starting from iteration 1.")
                  start_iter = 1 # Fallback
                  self.trainExamplesHistory = []
                  self.skipFirstSelfPlay = False # Must perform self-play


        # If loading a model but not examples, start from iteration 1 and potentially skip first self-play
        elif load_model:
             log.info("Loading model but not examples. Starting from iteration 1.")
             start_iter = 1
             # Load the model checkpoint into both nnet and pnet.
             # This allows pitting the loaded model against itself in the first arena,
             # and then training from this loaded state.
             try:
                 # NNet.load_checkpoint uses its internal model_cfg for device
                 self.nnet.load_checkpoint(load_folder, load_file) # Load into nnet (which has optimizer/scheduler setup)
                 self.pnet.load_checkpoint(load_folder, load_file) # Load into pnet (evaluation only)
                 log.info(f"Loaded model '{load_folder}/{load_file}' into both nnet and pnet.")
             except FileNotFoundError:
                 log.error(f"Checkpoint not found at {load_folder}/{load_file}! Starting with untrained models.")
                 # Continue with untrained models if loading fails
             except Exception as e:
                 log.error(f"Error loading checkpoint {load_folder}/{load_file}: {e}! Starting with untrained models.", exc_info=True)


             self.trainExamplesHistory = [] # History is empty if not loaded separately
             # skipFirstSelfPlay should be explicitly configured, not guessed from load_model alone.
             # Access skipFirstSelfPlay from training config, default to False.
             # It only applies if a model *was* successfully loaded.
             self.skipFirstSelfPlay = self.cfg.training.get('skipFirstSelfPlay', False) and load_model # Check cfg and if load_model was true
             if self.skipFirstSelfPlay:
                  log.info("Skipping first self-play due to cfg.training.skipFirstSelfPlay.")
             else:
                  log.info("Performing first self-play.")


        # If not loading examples and not loading a model, start fresh from iteration 1
        else:
             log.info("Starting fresh training from iteration 1.")
             start_iter = 1
             self.trainExamplesHistory = []
             self.skipFirstSelfPlay = False # Must perform self-play


        # --- Training Loop ---
        # Access total number of iterations from training config
        num_iters = self.cfg.training.numIters

        for i in range(start_iter, num_iters + 1):
            log.info(f'Starting Iteration #{i}/{num_iters} ...')
            iteration_start_time = time.time()

            # --- Self Play ---
            # Determine if self-play should run this iteration
            perform_self_play = not (self.skipFirstSelfPlay and i == start_iter) # Skip only the first iteration if flag is set

            if perform_self_play:
                 # Run self-play episodes in parallel
                 # run_parallel_episodes passes self.cfg to workers
                 iterationTrainExamples = self.run_parallel_episodes(i)
                 num_new_examples = len(iterationTrainExamples) # Number of (input, pi, z, valids) tuples

                 # Check if any examples were generated or if history is available
                 if num_new_examples == 0 and not self.trainExamplesHistory:
                      log.error("No new training examples generated and no history available. Stopping.")
                      # Save history collected so far before stopping
                      self.saveTrainExamples(i)
                      break # Exit the training loop

                 # Add new examples to history if any were generated
                 if num_new_examples > 0:
                    # trainExamplesHistory stores deques, each with maxlen from cfg.training.maxlenOfQueue
                    self.trainExamplesHistory.append(iterationTrainExamples)
                    log.info(f"Iteration {i}: Added {num_new_examples} new examples to history.")
                 else:
                      log.warning(f"Iteration {i}: No new examples generated in self-play. Training will use only historical data.")


                 # Log self-play duration and number of examples generated
                 if self.wandb_enabled:
                      wandb.log({'iteration': i, 'self_play_examples_generated': num_new_examples, 'self_play_duration_sec': time.time() - iteration_start_time})
            else:
                # If skipping self-play, create an empty deque for this iteration's examples
                # Use maxlenOfQueue from cfg
                iterationTrainExamples = deque([], maxlen=self.cfg.training.maxlenOfQueue)
                log.info(f"Iteration {i}: Skipped self-play. Using {len(self.trainExamplesHistory)} iterations of historical examples.")
                if self.wandb_enabled:
                    wandb.log({'iteration': i, 'self_play_examples_generated': 0, 'self_play_duration_sec': 0})


            # --- Manage Training Examples History ---
            # Trim the history to keep only the most recent iterations as specified in config
            # Access numItersForTrainExamplesHistory from cfg
            while len(self.trainExamplesHistory) > self.cfg.training.numItersForTrainExamplesHistory:
                if not isinstance(self.trainExamplesHistory, list) or not self.trainExamplesHistory:
                    log.error("trainExamplesHistory is not a list or is empty unexpectedly during trimming.")
                    break # Prevent infinite loop if structure is wrong
                log.info(f"Iteration {i}: Removing oldest examples. History length before: {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0) # Remove the oldest deque


            # Combine examples from all iterations in history for training
            trainExamples = []
            for examples_deque in self.trainExamplesHistory:
                 if isinstance(examples_deque, deque):
                     # The examples tuples now contain [input, pi, z, valids] - z is already calculated by worker
                     trainExamples.extend(examples_deque) # Extend with (input, pi, z, valids) tuples
                 else:
                      log.error(f"Unexpected element type in trainExamplesHistory: {type(examples_deque)}. Skipping element.")


            # Check if there are any examples to train on
            if not trainExamples:
                 log.warning(f"Iteration {i}: No training examples available from history. Skipping training and arena.")
                 # Save history (which might contain 0 new examples)
                 self.saveTrainExamples(i)
                 continue # Skip to the next iteration


            shuffle(trainExamples) # Shuffle the combined examples
            log.info(f"Iteration {i}: Training on {len(trainExamples)} examples from history.")

            # --- Training ---
            training_start_time = time.time()
            self.nnet.train(trainExamples) # <--- TUTAJ NASTĘPUJE FAKTYCZNY TRENING!
            training_duration = time.time() - training_start_time
            log.info(f"Iteration {i}: Training finished in {training_duration:.2f}s.")
            if self.wandb_enabled:
                 wandb.log({'iteration': i, 'training_duration_sec': training_duration})

            # --- Save the newly trained model ---
            # Zapisz model po treningu (będzie to model "nowy" w arenie)
            # Access checkpoint_dir from cfg (already resolved path)
            temp_checkpoint_path = os.path.join(self.cfg.checkpoint_dir, 'temp.pth.tar')
            # NNet.save_checkpoint uses its internal model_cfg, optimizer_cfg, training_cfg
            self.nnet.save_checkpoint(folder=self.cfg.checkpoint_dir, filename='temp.pth.tar')


            # --- Parallel Arena Comparison ---
            arena_start_time = time.time()
            log.info('PITTING AGAINST PREVIOUS VERSION (Parallel Arena)')

            # Access arenaCompare from cfg for the number of games
            num_arena_games = self.cfg.arena.arenaCompare
            # Access num_workers from training config (used by the parallel pool)
            num_workers = self.cfg.training.num_workers


            # Define checkpoint paths for the previous and new models
            # The 'previous' model is typically the 'best.pth.tar' from the last accepted iteration.
            # The 'new' model is the 'temp.pth.tar' saved just before training.
            # Access checkpoint_dir from cfg (already resolved path)
            prev_checkpoint_path = os.path.join(self.cfg.checkpoint_dir, 'best.pth.tar')
            # If best.pth.tar doesn't exist (e.g., first iteration, or loading models failed),
            # pit the new model against itself ('temp.pth.tar' vs 'temp.pth.tar').
            # This handles the case where loading failed but training might still happen.
            if not os.path.exists(prev_checkpoint_path):
                 log.warning(f"'best.pth.tar' not found at {prev_checkpoint_path}. Pitting the new model ('temp.pth.tar') against itself.")
                 prev_checkpoint_path = temp_checkpoint_path # Pit against itself

            # Check if necessary checkpoint files exist before starting the arena
            arena_possible = True
            if num_arena_games <= 0 or num_workers <= 0:
                 log.warning(f"Arena games requested ({num_arena_games}) or workers ({num_workers}) <= 0. Skipping arena.")
                 arena_possible = False
            # Check if both the 'previous' and 'new' checkpoint files exist
            elif not os.path.exists(prev_checkpoint_path): # prev_checkpoint_path might point to temp if best is missing
                  log.warning(f"Previous model checkpoint not found at {prev_checkpoint_path}. Skipping arena.")
                  arena_possible = False
            elif not os.path.exists(temp_checkpoint_path):
                 log.warning(f"New model checkpoint not found at {temp_checkpoint_path}. Skipping arena.")
                 arena_possible = False


            if arena_possible:
                 log.info(f"Starting {num_arena_games} parallel arena games using {num_workers} workers...")

                 # Prepare tasks for the arena worker function
                 arena_tasks = []
                 for game_id in range(num_arena_games):
                     # Alternate starting player every game for fairness
                     # Game 0: P1 (Prev) White, P2 (New) Black
                     # Game 1: P2 (New) White, P1 (Prev) Black
                     # etc.
                     player1_is_white = (game_id % 2 == 0) # Player 1 (Prev) is White if game_id is even
                     # Pass the full OmegaConf cfg object to the worker
                     # Pass the paths to the previous and new model checkpoints
                     arena_tasks.append((game_id, self.cfg, prev_checkpoint_path, temp_checkpoint_path, player1_is_white))

                 # Execute arena games in parallel using the Pool
                 arena_results = [] # Collect results
                 try:
                     # Use a new Pool for arena games
                     with mp.Pool(processes=num_workers) as pool:
                          # imap_unordered allows results to come back as games finish
                          # run_arena_game_worker returns (outcome_relative_to_white, player1_is_white_in_game) or None
                          results_iterator = pool.imap_unordered(run_arena_game_worker, arena_tasks)
                          # Wrap with tqdm for a progress bar
                          arena_results = list(tqdm(results_iterator, total=num_arena_games, desc="Parallel Arena Games"))

                 except Exception as e:
                      log.error(f"Error during parallel arena pool execution: {e}", exc_info=True)
                      arena_possible = False # Mark arena as failed


            # --- Aggregate Arena Results ---
            if arena_possible:
                 # Aggregate results from the completed games
                 prev_wins = 0 # Corresponds to player1 (Previous model) wins
                 new_wins = 0 # Corresponds to player2 (New model) wins
                 draws = 0
                 successful_games = 0 # Count games that completed without worker failure

                 for result_tuple in arena_results:
                      # Check if the worker returned a valid result tuple (not None)
                      if result_tuple is None:
                           log.warning("An arena worker failed or returned None. Skipping result for this game.")
                           continue # Skip failed games

                      # Unpack the result tuple returned by the worker
                      game_outcome_relative_to_white, player1_is_white_in_game = result_tuple
                      successful_games += 1 # This game finished successfully

                      # Interpret the outcome (relative to White) based on who was White
                      # Game outcome is > 0 if White won, < -EPS if Black won, ~0 if Draw
                      if abs(game_outcome_relative_to_white) < EPS: # Use defined EPS for draw tolerance
                           draws += 1
                      elif game_outcome_relative_to_white > 0: # White won the game
                           if player1_is_white_in_game:
                                prev_wins += 1 # Player 1 (Prev) was White and won
                           else:
                                new_wins += 1  # Player 2 (New) was White and won
                      else: # Black won the game (game_outcome_relative_to_white < -EPS)
                           if player1_is_white_in_game:
                                new_wins += 1  # Player 2 (New, Black) won
                           else:
                                prev_wins += 1 # Player 1 (Prev, Black) won


                 arena_duration = time.time() - arena_start_time
                 total_scored_games = prev_wins + new_wins + draws # Only count games that finished and weren't skipped
                 log.info(f'ARENA RESULTS ({successful_games}/{num_arena_games} games played) - NEW/PREV WINS : {new_wins} / {prev_wins} ; DRAWS : {draws}')
                 log.info(f"Parallel arena comparison finished in {arena_duration:.2f}s.")

                 # Log arena results to WandB
                 if self.wandb_enabled:
                     wandb.log({
                         'iteration': i,
                         'arena_new_wins': new_wins,
                         'arena_prev_wins': prev_wins,
                         'arena_draws': draws,
                         'arena_duration_sec': arena_duration,
                         'arena_games_played_total': num_arena_games,
                         'arena_games_played_successful': successful_games # Log actual games completed
                         })


                 # --- Model Acceptance Decision ---
                 # Decide whether to accept the new model based on arena results
                 # Access updateThreshold and save_anyway from arena config
                 update_threshold = self.cfg.arena.updateThreshold
                 save_anyway = self.cfg.arena.save_anyway

                 # Calculate win rate (including draws as half wins) based on *successful* games
                 new_score = new_wins + 0.5 * draws
                 prev_score = prev_wins + 0.5 * draws
                 total_score = new_score + prev_score # Total score for normalization
                 # Avoid division by zero if no games were scored (e.g., all failed)
                 win_rate_plus_draws = new_score / total_score if total_score > 0 else 0.5 # Default to 0.5 if no games scored

                 log.info(f"New model score (wins + 0.5*draws) on {total_scored_games} scored games: {new_score:.2f}")
                 log.info(f"Win rate + 0.5*draw rate: {win_rate_plus_draws:.3f} (Threshold: {update_threshold:.3f})")

                 accepted = False # Flag to indicate if the new model is accepted
                 if save_anyway:
                     log.info('cfg.arena.save_anyway is True: ACCEPTING NEW MODEL REGARDLESS OF ARENA')
                     accepted = True
                 elif total_scored_games == 0:
                      log.warning("No successful arena games played. Cannot determine model acceptance. Rejecting new model by default.")
                      accepted = False # Reject if no valid games were played
                 elif win_rate_plus_draws >= update_threshold:
                     log.info(f'ACCEPTING NEW MODEL (WinRate+Draws {win_rate_plus_draws:.3f} >= Threshold {update_threshold:.3f})')
                     accepted = True
                 else:
                     log.info(f'REJECTING NEW MODEL (WinRate+Draws {win_rate_plus_draws:.3f} < Threshold {update_threshold:.3f})')
                     accepted = False

                 # --- Save or Reload Model ---
                 if accepted:
                      # Save the newly trained model ('temp.pth.tar') as the new 'best.pth.tar'.
                      # Also save a dated checkpoint for tracking.
                      # Access checkpoint_dir from cfg (already resolved path)
                      # NNet.save_checkpoint saves model, optimizer, scheduler, and config subsets
                      self.nnet.save_checkpoint(folder=self.cfg.checkpoint_dir, filename=self.getCheckpointFile(i)) # Save dated checkpoint
                      self.nnet.save_checkpoint(folder=self.cfg.checkpoint_dir, filename='best.pth.tar') # Overwrite best.pth.tar
                      log.info(f"New model saved as 'best.pth.tar' and '{self.getCheckpointFile(i)}'.")

                      # The 'pnet' instance (previous best) is not updated here.
                      # It will be loaded from 'best.pth.tar' in the next iteration.

                 else:
                     # Reject the new model. Reload the previous best model ('best.pth.tar') into nnet.
                     # This ensures nnet is the previous best model for the next iteration's self-play.
                     log.info("REJECTING NEW MODEL. Reloading previous best model into NNet.")
                     try:
                         # Ensure 'best.pth.tar' exists before trying to load
                         prev_best_path_actual = os.path.join(self.cfg.checkpoint_dir, 'best.pth.tar')
                         if os.path.exists(prev_best_path_actual):
                             # NNet.load_checkpoint uses its internal model_cfg for device
                             self.nnet.load_checkpoint(folder=self.cfg.checkpoint_dir, filename='best.pth.tar')
                             log.info("Previous best model successfully reloaded into NNet.")
                         else:
                              log.error("Previous 'best.pth.tar' not found after rejecting new model! Cannot reload. NNet state remains the rejected one.")
                     except Exception as e:
                         log.error(f"Error reloading previous 'best.pth.tar': {e}. NNet state might be inconsistent.", exc_info=True)

                 # Log acceptance decision to WandB
                 if self.wandb_enabled: wandb.log({'iteration': i, 'model_accepted': int(accepted), 'new_model_win_rate_plus_draws': win_rate_plus_draws})

            else: # If arena_possible was False (due to setup error, pool failure, or no games)
                 log.warning("Skipping Arena comparison and model acceptance due to setup or pool failure.")
                 # If arena was skipped, the newly trained model is effectively rejected.
                 # Reload the previous best model into nnet to ensure consistency for the next iteration.
                 log.info("Reloading previous best model into NNet because arena was skipped.")
                 try:
                    prev_best_path_actual = os.path.join(self.cfg.checkpoint_dir, 'best.pth.tar')
                    if os.path.exists(prev_best_path_actual):
                        self.nnet.load_checkpoint(folder=self.cfg.checkpoint_dir, filename='best.pth.tar')
                        log.info("Previous best model successfully reloaded after skipped arena.")
                    else:
                         log.warning("Previous 'best.pth.tar' not found after skipped arena. Cannot reload.")
                 except Exception as e:
                    log.error(f"Error reloading previous 'best.pth.tar' after skipped arena: {e}. NNet state might be inconsistent.", exc_info=True)

                 # Log rejection to WandB
                 if self.wandb_enabled: wandb.log({'iteration': i, 'model_accepted': 0, 'new_model_win_rate_plus_draws': 0.5}) # Log as rejected with neutral score


            # --- Save Training Examples History ---
            # Save history *after* training and arena for this iteration
            # Use checkpoint_dir from cfg (already resolved path) for saving examples
            self.saveTrainExamples(i)


            # --- End of Iteration ---
            iteration_duration = time.time() - iteration_start_time
            log.info(f"Iteration {i} finished in {iteration_duration:.2f} seconds.")
            if self.wandb_enabled: wandb.log({'iteration': i, 'iteration_duration_sec': iteration_duration})


        # --- End of Training Loop ---
        log.info("Training process completed.")


    # --- Helper methods for Self-Play Episodes ---
    # This method runs multiple self-play episodes in parallel using run_episode_worker.
    def run_parallel_episodes(self, iteration: int) -> deque:
        """Runs self-play episodes in parallel using multiprocessing."""
        # Access numEps and num_workers from cfg
        num_episodes = self.cfg.training.numEps
        # Determine the number of workers to use, capped by CPU count, numEps, and config setting
        try:
            available_cpus = mp.cpu_count()
            # Leave some cores free for the main process and OS
            recommended_workers = max(1, available_cpus - 2)
            num_workers = min(recommended_workers, num_episodes, self.cfg.training.num_workers)
        except NotImplementedError:
             # Fallback if cpu_count is not implemented on the system
             log.warning("Could not determine CPU count. Using cfg.training.num_workers directly.")
             num_workers = min(self.cfg.training.num_workers, num_episodes)


        if num_workers <= 0:
             log.warning("Number of workers <= 0. Cannot run parallel episodes.")
             # Return an empty deque with the correct maxlen
             return deque([], maxlen=self.cfg.training.maxlenOfQueue) # Use cfg.training.maxlenOfQueue


        log.info(f"Starting {num_episodes} self-play episodes using {num_workers} workers...")

        # Save the current state of the new network (nnet) so workers can load it.
        # Use checkpoint_dir from cfg (already resolved path)
        model_checkpoint_path = os.path.join(self.cfg.checkpoint_dir, 'temp_selfplay.pth.tar') # Use a separate temp file for self-play if needed
        try:
            # NNet.save_checkpoint saves model, optimizer, scheduler, and config subsets
            self.nnet.save_checkpoint(folder=self.cfg.checkpoint_dir, filename='temp_selfplay.pth.tar')
            log.info(f"Current model saved to {model_checkpoint_path} for workers.")
        except Exception as save_err:
             # If saving fails, workers will not be able to load the latest model.
             log.error(f"Failed to save temporary model for workers at {model_checkpoint_path}: {save_err}. Workers might use outdated model.", exc_info=True)
             # Decide how to handle this: proceed with warning, or stop?
             # Let's proceed, but the workers might fail to load, which will be caught by the worker itself.
             model_checkpoint_path = None # Indicate that saving failed


        # Create tasks for the worker function (run_episode_worker)
        # Each task tuple: (iteration, worker_id, cfg, model_checkpoint_path)
        # Pass the full OmegaConf cfg object to the worker.
        # Worker IDs are just for logging/debugging within the worker.
        tasks = [(iteration, i, self.cfg, model_checkpoint_path) for i in range(num_episodes)]

        # Use multiprocessing Pool to run workers in parallel
        all_examples_list = [] # Collect examples from all successful episodes into a list first
        try:
            # Use Pool for parallel execution. imap_unordered is good for processing results as they finish.
            with mp.Pool(processes=num_workers) as pool:
                 # imap_unordered applies run_episode_worker to each item in tasks and returns results as they complete.
                 # run_episode_worker returns a list of examples for one episode.
                 results_iterator = pool.imap_unordered(run_episode_worker, tasks)
                 # Wrap the iterator with tqdm for a progress bar
                 episode_results = list(tqdm(results_iterator, total=num_episodes, desc="Self Play Episodes"))

            # Aggregate examples from all completed episodes
            successful_episodes = 0 # Count episodes that returned a list of examples
            for episode_examples in episode_results:
                 # Check if the worker returned a list (successful execution, could be empty list)
                 if isinstance(episode_examples, list):
                     all_examples_list.extend(episode_examples) # Add examples from this episode to the main list
                     # Consider an episode successful if it returned *any* examples
                     if len(episode_examples) > 0:
                          successful_episodes += 1
                 else:
                      # This would catch cases where a worker returned something other than a list (e.g., None on hard failure)
                      log.error(f"Unexpected return type from worker: {type(episode_examples)}")
                      # Optionally handle non-list returns (e.g., log more info, count as failed episode)

            log.info(f"Finished parallel self-play. Collected {len(all_examples_list)} examples from {successful_episodes}/{num_episodes} episodes with results.")

        except Exception as e:
            log.error(f"Error during parallel self-play pool execution: {e}", exc_info=True)
            # If the pool execution itself fails, all_examples_list might be incomplete or empty.

        # Convert the collected examples list to a deque with the configured maxlen
        # Use maxlenOfQueue from cfg
        return deque(all_examples_list, maxlen=self.cfg.training.maxlenOfQueue)


    # --- Helper methods for Checkpoints and Examples ---
    def getCheckpointFile(self, iteration: int) -> str:
        """Generate checkpoint filename based on iteration."""
        return f'checkpoint_{iteration:04d}.pth.tar' # Padded iteration number

    def saveTrainExamples(self, iteration: int) -> None:
        """Save training examples history with iteration number."""
        # Access checkpoint_dir from cfg (already resolved path)
        folder = self.cfg.checkpoint_dir # Use checkpoint_dir for saving examples
        if not os.path.exists(folder): os.makedirs(folder) # Ensure the directory exists
        # Filename includes iteration number
        filename = os.path.join(folder, f"train_examples_iter_{iteration:04d}.pkl")
        try:
            # Save the entire list of deques (trainExamplesHistory)
            with open(filename, "wb+") as f: Pickler(f).dump(self.trainExamplesHistory)
            log.info(f"Saved training examples history ({len(self.trainExamplesHistory)} iterations) to {filename}")

            # --- Optional: Log examples artifact to WandB ---
            if self.wandb_enabled:
                 try:
                     log.info(f"Logging examples artifact to WandB: {filename}")
                     # Artifact name helps identify the run and iteration
                     artifact = wandb.Artifact(f'train_examples-{wandb.run.id}-{iteration:04d}', type='train_examples')
                     # Add the saved file to the artifact
                     artifact.add_file(filename)
                     # Log the artifact
                     wandb.log_artifact(artifact)
                 except Exception as e:
                     log.error(f"Failed to log examples artifact to WandB: {e}")
            # -----------------------------------------------

        except Exception as e:
            log.error(f"Error saving training examples to {filename}: {e}")

    def _find_latest_examples_file(self, folder: str) -> str | None:
        """Helper to find the latest saved training examples file by iteration number."""
        # Use the specified folder (resolved path)
        if not os.path.exists(folder):
            return None # Folder doesn't exist

        # List files matching the pattern 'train_examples_iter_XXXX.pkl'
        examples_files = [f for f in os.listdir(folder) if re.match(r'train_examples_iter_\d{4}\.pkl$', f)]
        if not examples_files:
            return None # No example files found

        # Define a function to extract the iteration number for sorting
        def get_iteration_from_filename(filename):
            match = re.search(r'_iter_(\d+)\.pkl$', filename)
            # Return -1 if no match (shouldn't happen with the regex filter, but safe)
            return int(match.group(1)) if match else -1

        # Sort files by iteration number
        examples_files.sort(key=get_iteration_from_filename)

        latest_file_name = examples_files[-1] # The last file after sorting is the latest
        log.info(f"Found latest examples file name: {latest_file_name}")
        # Return the full path to the latest file
        return os.path.join(folder, latest_file_name)


    def loadTrainExamples(self) -> None:
        """Load training examples history from the latest checkpoint file."""
        # Access load_examples_folder from cfg (already resolved path)
        folder = self.cfg.loading.load_examples_folder

        # Find the path to the latest examples file
        examplesFile = self._find_latest_examples_file(folder)

        # Check if the file exists
        if examplesFile is None or not os.path.isfile(examplesFile):
            log.warning(f'No previous trainExamples file found in "{folder}"! Starting fresh history.')
            self.trainExamplesHistory = [] # Ensure history is empty
            self.skipFirstSelfPlay = False # Must perform self-play if no examples loaded
        else:
            try:
                log.info(f"Loading trainExamples from: {examplesFile}")
                # Open and unpickle the file
                with open(examplesFile, "rb") as f:
                    loaded_history = Unpickler(f).load()

                # Verify the loaded data structure
                # Expecting a list of deques
                if isinstance(loaded_history, list) and (not loaded_history or all(isinstance(d, deque) for d in loaded_history)):
                    # Ensure loaded deques have the correct maxlen based on the current cfg
                    # Use maxlenOfQueue from cfg
                    self.trainExamplesHistory = [deque(d, maxlen=self.cfg.training.maxlenOfQueue) for d in loaded_history]
                    log.info(f'Loading done! Loaded history from {len(self.trainExamplesHistory)} previous iterations.')
                    # If examples were loaded, skip the first self-play iteration
                    self.skipFirstSelfPlay = True # Set flag to skip first self-play
                    log.info("skipFirstSelfPlay set to True as examples were loaded.")
                else:
                     # If the loaded data has an unexpected structure
                     log.error(f"Loaded examples data from {examplesFile} has unexpected structure ({type(loaded_history)}). Resetting history.")
                     self.trainExamplesHistory = [] # Reset history
                     self.skipFirstSelfPlay = False # Must perform self-play
            except Exception as e:
                 # Catch any errors during loading (e.g., file corrupted)
                 log.error(f"Error loading training examples from {examplesFile}: {e}. Starting fresh history.", exc_info=True)
                 self.trainExamplesHistory = [] # Reset history
                 self.skipFirstSelfPlay = False # Must perform self-play

    # Removed the redundant playGames method from Coach class.