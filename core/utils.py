import os
import torch
import random
import shutil
import logging
import numpy as np
from scipy.stats import entropy

class LinearSchedule:
    """Linear interpolation between initial_p and final_p over schedule_timesteps."""
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def make_results_dir(exp_path, args):
    """Create and prepare directories for experiment results."""
    os.makedirs(exp_path, exist_ok=True)
    
    if args.opr == 'train' and os.path.exists(exp_path) and os.listdir(exp_path):
        if not args.force:
            raise FileExistsError(f'{exp_path} is not empty. Please use --force to overwrite it')
        else:
            print('Warning, path exists! Rewriting...')
            shutil.rmtree(exp_path)
            os.makedirs(exp_path)
    
    log_path = os.path.join(exp_path, 'logs')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'model'), exist_ok=True)
    return exp_path, log_path

def init_logger(base_path):
    """Initialize and configure logger."""
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
    
    for mode in ['train', 'test', 'train_test', 'root']:
        file_path = os.path.join(base_path, mode + '.log')
        logger = logging.getLogger(mode)
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        logger.setLevel(logging.DEBUG)

def select_action(visit_counts, temperature=1, deterministic=True):
    """Select action from the root visit counts."""
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
    total_count = sum(action_probs)
    action_probs = [x / total_count for x in action_probs]
    
    if deterministic:
        action_pos = np.argmax([v for v in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)
    
    count_entropy = entropy(action_probs, base=2)
    return action_pos, count_entropy

def algebraic_to_coordinates(algebraic):
    """Convert algebraic chess notation (e.g., 'e4') to coordinates (row, col)."""
    col = ord(algebraic[0]) - ord('a')
    row = 8 - int(algebraic[1])
    return row, col

def coordinates_to_algebraic(row, col):
    """Convert coordinates (row, col) to algebraic chess notation (e.g., 'e4')."""
    file_char = chr(col + ord('a'))
    rank_char = str(8 - row)
    return file_char + rank_char

def move_to_action_index(move, board_size=8):
    """
    Convert a chess move to an action index for the neural network.
    Maps (from_row, from_col, to_row, to_col) to a single action index.
    """
    from_row, from_col = move[0]
    to_row, to_col = move[1]
    # Maps to an index in range [0, 4095] for standard chess (8x8 board)
    return from_row * board_size**3 + from_col * board_size**2 + to_row * board_size + to_col

def action_index_to_move(action_idx, board_size=8):
    """
    Convert an action index back to a chess move.
    Maps a single action index to (from_row, from_col, to_row, to_col).
    """
    to_col = action_idx % board_size
    action_idx //= board_size
    to_row = action_idx % board_size
    action_idx //= board_size
    from_col = action_idx % board_size
    from_row = action_idx // board_size
    
    return ((from_row, from_col), (to_row, to_col))

def prepare_observation_lst(observation_lst):
    """
    Przygotowuje listę obserwacji szachowych jako tablicę numpy dla modelu.
    
    Parameters
    ----------
    observation_lst: list
        Lista obserwacji, gdzie każda obserwacja to tablica numpy
        o kształcie (12, 8, 8) reprezentująca stan szachownicy.
        
    Returns
    -------
    ndarray
        Pakiet obserwacji jako tablica numpy
    """
    import numpy as np
    return np.array(observation_lst, dtype=np.float32)
