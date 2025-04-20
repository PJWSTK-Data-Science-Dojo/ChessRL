import sys
import logging
import coloredlogs
from omegaconf import OmegaConf, DictConfig
import gc
import torch
import wandb
import multiprocessing as mp
import os

# Add path utility import
# Assuming utils.py is in src/luna/utils.py relative to src/
try:
    # Find the project root (directory containing this script)
    _PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    # Define config path relative to project root
    _CONFIG_RELATIVE_PATH = 'config/config.yaml'
    _CONFIG_PATH = os.path.join(_PROJECT_ROOT, _CONFIG_RELATIVE_PATH)

    # Helper to resolve paths from config relative to the project root
    def _resolve_cfg_path(cfg_path: str):
        return os.path.join(_PROJECT_ROOT, cfg_path)

except Exception as e:
    print(f"Error setting up project paths: {e}")
    sys.exit(1)


# --- Set Multiprocessing Start Method ---
try:
    mp.set_start_method('spawn', force=True)
    print("Multiprocessing start method set to 'spawn'.")
except RuntimeError as e:
    print(f"Could not set start method (possibly already set): {e}")

gc.enable()

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
coloredlogs.install(level='INFO')


# --- Load Configuration ---
try:
    cli_args = OmegaConf.from_cli(sys.argv[1:])
    # Load config using the resolved absolute path
    config = OmegaConf.load(_CONFIG_PATH)
    cfg: DictConfig = OmegaConf.merge(config, cli_args)

    cfg.cuda = torch.cuda.is_available()

    # Resolve relevant paths in the config relative to project root
    cfg.checkpoint_dir = _resolve_cfg_path(cfg.checkpoint_dir)
    cfg.loading.load_folder = _resolve_cfg_path(cfg.loading.load_folder)
    cfg.loading.load_examples_folder = _resolve_cfg_path(cfg.loading.load_examples_folder)


    log.info("Configuration loaded successfully.")
    log.info(f"Effective configuration:\n{OmegaConf.to_yaml(cfg)}")

except Exception as e:
    log.error(f"Error loading configuration: {e}")
    sys.exit(1)

from luna.coach import Coach
from luna.game.luna_game import ChessGame as Game
from luna.NNet import Luna_Network as nn


def main() -> int:
    # --- WandB Initialization ---
    try:
        wandb.init(
            project=cfg.project_name,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        log.info("Weights & Biases initialized successfully.")
        wandb_enabled = True
    except Exception as e:
        log.error(f"Could not initialize Weights & Biases: {e}. Running without WandB logging.")
        wandb_enabled = False

    log.info('Loading %s...', Game.__name__)
    # Initialize Game with full cfg (it extracts game section)
    g = Game(cfg=cfg)

    log.info('Loading the Coach...')
    # Coach expects game, full cfg, wandb_enabled.
    # Coach will initialize nnet and pnet internally with appropriate config subsets.
    c = Coach(g, cfg, wandb_enabled=wandb_enabled)

    # ... (Loading examples logic) ...
    if cfg.loading.load_examples:
        log.info("Loading 'trainExamples' from file...")
        # Coach.loadTrainExamples uses its internal cfg
        c.loadTrainExamples()

    log.info('Starting the learning process')
    c.learn() # Coach.learn uses its internal cfg

    if wandb_enabled:
        wandb.finish()
    log.info("Learning process finished.")

    return 0

if __name__ == "__main__":
    sys.exit(main())