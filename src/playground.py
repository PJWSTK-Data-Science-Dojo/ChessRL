import sys
import logging
import coloredlogs
from omegaconf import OmegaConf, DictConfig
from collections import deque
import torch
import numpy as np
import chess
import time
import tqdm
import os

try:
    _PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    _CONFIG_RELATIVE_PATH = 'config/config.yaml'
    _CONFIG_PATH = os.path.join(_PROJECT_ROOT, _CONFIG_RELATIVE_PATH)

    def _resolve_cfg_path(cfg_path: str):
        # Adjust this helper if paths in config.yaml are relative to a different base (e.g., project root)
        # Assuming paths like ./temp/ or ./pretrained_models/ are relative to the directory containing config.yaml (src/config)
        # Or they might be relative to the project root (parent of src/). Let's assume project root as in main.py.
        project_root_assuming_src = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) # Parent of src/
        # Check if the path is already absolute or contains '..', handle simple relative paths
        if os.path.isabs(cfg_path) or cfg_path.startswith('..'):
             # For complex or absolute paths, just return as is or handle case by case
             log.warning(f"Skipping specific resolution for complex path: {cfg_path}")
             return cfg_path
        # Assume simple relative paths like ./temp/ or ./pretrained_models/ are relative to project root
        return os.path.join(project_root_assuming_src, cfg_path)

except Exception as e:
    print(f"Error setting up project paths in playground: {e}")
    sys.exit(1)

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

# --- Load Configuration ---
try:
    cli_args = OmegaConf.from_cli(sys.argv[1:])
    config = OmegaConf.load(_CONFIG_PATH)
    cfg: DictConfig = OmegaConf.merge(config, cli_args)

    cfg.cuda = torch.cuda.is_available()

    # Resolve relevant paths in the config
    cfg.checkpoint_dir = _resolve_cfg_path(cfg.checkpoint_dir)
    cfg.loading.load_folder = _resolve_cfg_path(cfg.loading.load_folder)
    cfg.loading.load_examples_folder = _resolve_cfg_path(cfg.loading.load_examples_folder)


    log.info("Playground Configuration loaded successfully.")
    log.info(f"Effective configuration:\n{OmegaConf.to_yaml(cfg)}")

except Exception as e:
    log.error(f"Error loading configuration: {e}")
    sys.exit(1)

# Import Game, NNet, MCTS, Arena after setting logger and loading config
from luna.game.luna_game import ChessGame as Game, who, _flip_action_index
from luna.NNet import Luna_Network as nn
from luna.mcts import MCTS
from luna.game.arena import Arena


def main() -> int:
    # --- Load Configuration ---
    try:
        cli_args = OmegaConf.from_cli(sys.argv[1:])
        # Load config using the resolved absolute path
        config = OmegaConf.load(_CONFIG_PATH)
        cfg: DictConfig = OmegaConf.merge(config, cli_args)

        cfg.cuda = torch.cuda.is_available()

        # Resolve relevant paths in the config relative to project root
        # Apply path resolution to relevant paths in cfg, assuming they are relative to project root
        cfg.checkpoint_dir = _resolve_cfg_path(cfg.checkpoint_dir) # Used for Coach, not directly here
        cfg.loading.load_folder = _resolve_cfg_path(cfg.loading.load_folder) # Used for Coach, not directly here
        cfg.loading.load_examples_folder = _resolve_cfg_path(cfg.loading.load_examples_folder) # Used for Coach, not directly here

        # Resolve inference loading paths
        if hasattr(cfg, 'inference'):
            cfg.inference.load_folder = _resolve_cfg_path(cfg.inference.load_folder)
            # Assuming load_file doesn't need path resolution, it's just a filename
            # cfg.inference.load_file = cfg.inference.load_file

        log.info("Playground Configuration loaded successfully.")
        log.info(f"Effective configuration:\n{OmegaConf.to_yaml(cfg)}")

    except Exception as e:
        log.error(f"Error loading configuration in playground: {e}")
        sys.exit(1)


    log.info('Loading %s...', Game.__name__)
    g = Game(cfg=cfg)

    log.info('Loading %s...', nn.__name__)
    # Initialize NNet instance with model_cfg (extracted from full cfg)
    model_cfg_nnet = OmegaConf.create({
        'model': cfg.model,
        'cuda': cfg.cuda
    })
    nnet = nn(g, model_cfg_nnet) # Pass game and model_cfg subset

    # --- Load the inference model based on cfg.inference ---
    load_model_inference = cfg.inference.get('load_model', False) # Use .get with default
    load_folder_inference = cfg.inference.get('load_folder', None) # Use .get with default
    load_file_inference = cfg.inference.get('load_file', None)   # Use .get with default


    if load_model_inference and load_folder_inference is not None and load_file_inference is not None:
        log.info(f'Loading checkpoint "{load_folder_inference}/{load_file_inference}" for arena NNet (Inference config)...')
        try:
            # Use NNet's load_checkpoint (it uses its internal model_cfg for device)
            nnet.load_checkpoint(load_folder_inference, load_file_inference)
        except FileNotFoundError:
            log.error(f"Checkpoint not found at {load_folder_inference}/{load_file_inference} for arena NNet. Cannot run arena with loaded model as requested by config.")
            # Decide behavior: exit or run with untrained model? Exit seems safer as requested.
            return 1
        except Exception as e:
            log.error(f"Error loading checkpoint {load_folder_inference}/{load_file_inference} for arena NNet: {e}", exc_info=True)
            return 1
    elif load_model_inference:
         log.warning("cfg.inference.load_model is True, but load_folder or load_file are not specified. Running arena with untrained NNet.")
    else:
         log.info("cfg.inference.load_model is False. Running arena with untrained NNet.")


    # Create MCTS instances with the loaded NNet and inference mcts_cfg
    # mcts_cfg_inference is taken from the 'inference' section
    mcts_cfg_inference = OmegaConf.create({
        'numMCTSSims': cfg.inference.get('numMCTSSims', 100),
        'cpuct': cfg.inference.get('cpuct', 1.0),
        'dir_noise': cfg.inference.get('dir_noise', False),
        'dirichlet_alpha': cfg.inference.get('dirichlet_alpha', 0.3),
        'dirichlet_epsilon': cfg.inference.get('dirichlet_epsilon', 0.25)
         # history_len is obtained from game instance
    })

    mcts1 = MCTS(g, nnet, mcts_cfg_inference)
    mcts2 = MCTS(g, nnet, mcts_cfg_inference)


    def player1_func(board: chess.Board, history: deque) -> int:
        pi_canonical = mcts1.getActionProb(board, history, temp=0)
        action_canonical = np.argmax(pi_canonical)
        action_actual = action_canonical
        if who(board.turn) == -1:
             action_actual = _flip_action_index(action_canonical)
        return action_actual

    def player2_func(board: chess.Board, history: deque) -> int:
        pi_canonical = mcts2.getActionProb(board, history, temp=0)
        action_canonical = np.argmax(pi_canonical)
        action_actual = action_canonical
        if who(board.turn) == -1:
             action_actual = _flip_action_index(action_canonical)
        return action_actual


    log.info('Starting Arena game (pitting loaded model against itself)...')
    arena = Arena(player1_func, player2_func, g)

    start_time = time.time()
    result = arena.playGame(verbose=True)

    duration = time.time() - start_time

    print("\n--- Arena Game Result ---")
    if abs(result) < EPS: # Use defined EPS for draw
        print("Result: Draw")
    elif result > 0:
        print("Result: Player 1 (White) Wins!")
    else:
        print("Result: Player 2 (Black) Wins!")
    print(f"Game duration: {duration:.2f} seconds")

    # --- Multiple Sequential Games ---
    # Access arenaCompare from cfg.arena
    num_games_to_play = cfg.arena.arenaCompare

    if num_games_to_play > 0:
        log.info(f'Starting {num_games_to_play} sequential Arena games (pitting loaded model against itself)...')
        total_p1_wins = 0
        total_p2_wins = 0
        total_draws = 0


        for i in tqdm(range(num_games_to_play), desc="Sequential Arena Games"):
             # Alternate starting player every game for fairness
             # Game 0: P1 (loaded model) White, P2 (loaded model) Black
             # Game 1: P2 (loaded model) White, P1 (loaded model) Black
             # ... and so on.
             # In playground, P1 and P2 are the SAME model, so the outcome relative
             # to Player 1 vs Player 2 is simply 1 vs -1, alternating who starts white.
             # We just need to sum wins/losses/draws over these games.

             # Create local players for this game to avoid state contamination if MCTS instances hold state
             # The worker functions are stateless wrappers around MCTS instances which hold state per simulation.
             # For sequential play in the main process, we can reuse the main mcts1/mcts2 instances if they are clean before each game.
             # However, it's safer to create new MCTS instances per game or reset them thoroughly.
             # Let's assume mcts1/mcts2 are stateful and need cleanup or re-creation.
             # Or, even simpler, rely on the Arena class resetting the board/history state.
             # The MCTS instances cache search results (Qsa, Nsa, etc.). These caches need to be cleared *before each game*.
             # Let's clear MCTS caches before each sequential game.

             mcts1.Qsa = {}; mcts1.Nsa = {}; mcts1.Ns = {}; mcts1.Ps = {}; mcts1.Es = {}; mcts1.Vs = {}; mcts1.state_info = {}; # Clear caches
             mcts2.Qsa = {}; mcts2.Nsa = {}; mcts2.Ns = {}; mcts2.Ps = {}; mcts2.Es = {}; mcts2.Vs = {}; mcts2.state_info = {}; # Clear caches


             if i % 2 == 0: # Game i is even (0, 2, 4...) - Player 1 (Loaded Model) starts White
                  white_player_func = player1_func
                  black_player_func = player2_func
             else: # Game i is odd (1, 3, 5...) - Player 2 (Loaded Model) starts White
                  white_player_func = player2_func
                  black_player_func = player1_func

             # Create a new Arena instance for each game to ensure clean state
             local_arena = Arena(white_player_func, black_player_func, g)

             gameResult_relative_to_white = local_arena.playGame(verbose=False)

             # Result is relative to White. We want to count wins/losses/draws of the *single model* pitted against itself.
             # Since P1 and P2 are the same model, the game result (win, loss, draw) applies to the model regardless of color.
             if abs(gameResult_relative_to_white) < EPS: total_draws += 1
             elif gameResult_relative_to_white > 0: # White won
                  # White was either P1 (if i is even) or P2 (if i is odd). The model won.
                  # We can just count wins regardless of color since it's the same model.
                  total_p1_wins += 1 # Arbitrarily count as P1 win for total tally
             else: # Black won
                  # Black was either P2 (if i is even) or P1 (if i is odd). The model lost.
                  total_p2_wins += 1 # Arbitrarily count as P2 win for total tally


        print("\n--- Multiple Sequential Games Results ---")
        print(f"Total Games: {num_games_to_play}")
        print(f"Model Wins (as White): {total_p1_wins}") # Renamed for clarity in self-play arena
        print(f"Model Wins (as Black): {total_p2_wins}") # Renamed for clarity in self-play arena
        print(f"Draws: {total_draws}")
    else:
         log.info("cfg.arena.arenaCompare is 0. Skipping sequential arena games.")


    return 0

if __name__ == "__main__":
    sys.exit(main())