"""
Handles rendering dataset for pretraining Luna model. Assumes the dataset is stored as parquet file
and has the same format as https://huggingface.co/datasets/angeluriot/chess_games dataset.

This version includes multiprocessing and threading support for significant performance improvements.
"""

import os
import time
import logging
import multiprocessing as mp
import numpy as np
import torch
import chess
import psutil
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import polars as pl
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from luna.game.luna_game import ChessGame, from_move, _flip_action_index, who
from .tools import UCI_TO_INT, INT_TO_UCI, convert_uci_strings_to_ints

# Set multiprocessing start method for Windows compatibility
if __name__ != '__main__':
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')
    except RuntimeError:
        pass  # Start method already set

log = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class GameProcessor:
    """Processes individual chess games and converts them to transitions."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.game = ChessGame(config)
        self.validate_moves = config.get('validation', {}).get('validate_during_processing', True)
        self.validate_actions = config.get('validation', {}).get('check_action_bounds', True)
        self.validate_values = config.get('validation', {}).get('check_value_bounds', True)
        # Get board history length for history indicator calculation
        self.board_history = config.get('game', {}).get('board_history', 8)
    
    def process_game(self, moves_uci_ints: List[int], winner: str) -> List[Dict[str, np.ndarray]]:
        """
        Process a single game and return all transitions.
        
        Args:
            moves_uci_ints: List of UCI move integers for the game
            winner: Game winner ('white', 'black', or 'draw')
            
        Returns:
            List of transition dictionaries with tensors
        """
        try:
            # Initialize board and history
            board, history = self.game.getInitBoard()
            transitions = []
            
            # Convert winner to numeric values from current player perspective
            if winner == 'draw':
                final_value = 0.0
            elif winner == 'white':
                white_value, black_value = 1.0, -1.0
            elif winner == 'black':
                white_value, black_value = -1.0, 1.0
            else:
                # Handle other outcomes as draws
                final_value = 0.0
                white_value, black_value = 0.0, 0.0
            
            # Play through the game
            for i, uci_int in enumerate(moves_uci_ints):
                try:
                    # Convert integer back to UCI string
                    if uci_int == -1 or uci_int not in INT_TO_UCI:
                        log.warning(f"Invalid UCI integer {uci_int} at position {i} in game. Skipping game.")
                        return []
                    
                    uci_move = INT_TO_UCI[uci_int]
                    
                    # Parse the UCI move
                    move = chess.Move.from_uci(uci_move)
                    
                    # Verify the move is legal
                    if self.validate_moves and move not in board.legal_moves:
                        log.warning(f"Illegal move {uci_move} at position {i} in game. Skipping game.")
                        return []
                    
                    # Get current player
                    current_player = who(board.turn)
                    
                    # Get canonical form for consistent representation
                    self.game.board = board
                    canonical_board = self.game.getCanonicalForm(board, current_player)
                    
                    # Get only current board state (not full history)
                    board_tensor = self.game._board_to_feature_array(canonical_board)

                    # In the mcts.py the getCannonicalHistory flips the black moves and is always called
                    # Therefore, here we need to flip the board_tensor if the current player is black
                    # The history flipping will be handled in the data laoder.
                    if current_player == -1:
                        board_tensor = self.game._flip_feature_array(board_tensor)
                    
                    # Calculate history indicator (how many previous samples are available)
                    history_indicator = min(i, self.board_history - 1)
                    
                    # Convert move to action index
                    action_index = from_move(move)
                    
                    # FIX: Convert action to canonical form to match the canonical board
                    if current_player == -1:  # If current player is black, flip the action to canonical form
                        action_index = _flip_action_index(action_index)
                    
                    # Validate action bounds
                    if self.validate_actions and (action_index < 0 or action_index >= 4096):
                        log.warning(f"Invalid action index {action_index} for move {uci_move}. Skipping game.")
                        return []
                    
                    # Determine value from current player's perspective
                    if winner == 'draw':
                        value = 0.0
                    else:
                        if current_player == 1:  # White to move
                            value = white_value
                        else:  # Black to move
                            value = black_value
                    
                    # Validate value bounds
                    if self.validate_values and (value < -1.1 or value > 1.1):
                        log.warning(f"Invalid value {value} for game. Skipping game.")
                        return []
                    
                    # Create transition dictionary with raw data for cross-process compatibility
                    # Convert to numpy for serialization, will be converted to tensors in main process
                    transition = {
                        'board': np.array(board_tensor, dtype=np.int8),
                        'action': np.array(action_index, dtype=np.int16),
                        'value': np.array(value, dtype=np.int8),
                        'history': np.array(history_indicator, dtype=np.int16),
                        'player': np.array(current_player, dtype=np.int8)
                    }
                    
                    transitions.append(transition)
                    
                    # Make the move to update board and history
                    board.push(move)
                    
                except ValueError as e:
                    log.warning(f"Failed to parse UCI move integer '{uci_int}' at position {i}: {e}. Skipping game.")
                    return []
                except Exception as e:
                    log.error(f"Error processing move {uci_int} at position {i}: {e}. Skipping game.")
                    return []
            
            return transitions
            
        except Exception as e:
            log.error(f"Error processing game: {e}")
            return []


class NumpyBatchManager:
    """
    Manages the creation and saving of training batches using pre-allocated NumPy arrays.
    This approach minimizes memory copies and avoids intermediate storage lists.
    """
    
    def __init__(self, batch_size: int, output_dir: str, config: DictConfig):
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.config = config
        self.batch_count = 0
        self.is_saving_file = False
        
        # Determine the shape of a single board state from the config
        # Only current board state (num_feature_planes, height, width) without history
        num_feature_planes = config.get('game', {}).get('num_feature_planes', 22)
        self.board_shape = (num_feature_planes, 8, 8)
        
        self.current_index = 0
        self._preallocate_batch()

    def _preallocate_batch(self):
        """Pre-allocates memory for a full batch to avoid repeated allocations."""
        log.info(f"Pre-allocating memory for new batch #{self.batch_count}...")
        self.batch_data = {
            'boards': np.zeros((self.batch_size, *self.board_shape), dtype=np.int8),
            'actions': np.zeros(self.batch_size, dtype=np.int16),
            'values': np.zeros(self.batch_size, dtype=np.int8),
            'history': np.zeros(self.batch_size, dtype=np.int16),
            'players': np.zeros(self.batch_size, dtype=np.int8)
        }
        self.current_index = 0

    def add_transition(self, transition: Dict[str, np.ndarray]):
        """
        Adds a single transition to the pre-allocated NumPy arrays.
        If the batch becomes full, it is saved to disk.
        """
        if self.is_saving_file:
            log.info("Waiting for batch to finish saving...")
            while self.is_saving_file:
                time.sleep(1.0)
            log.info("Batch saving finished. Continuing...")

        if self.current_index >= self.batch_size:
            log.error("Batch overflow detected. This should not happen.")
            return

        # Insert transition data directly into the numpy arrays
        self.batch_data['boards'][self.current_index] = transition['board']
        self.batch_data['actions'][self.current_index] = transition['action']
        self.batch_data['values'][self.current_index] = transition['value']
        self.batch_data['history'][self.current_index] = transition['history']
        self.batch_data['players'][self.current_index] = transition['player']
        self.current_index += 1

        # Save the batch if it's full
        if self.current_index == self.batch_size:
            self._save_batch()

    def _save_batch(self, is_final: bool = False):
        """
        Saves the current batch to a .pt file. If it's a final batch,
        it may be smaller than the full batch size.
        """
        if self.current_index == 0:
            return
        self.is_saving_file = True
    
        batch_id = self.batch_count
        num_transitions = self.current_index if is_final else self.batch_size
        log.info(f"Saving batch {batch_id} with {num_transitions} transitions...")

        try:
            start_time = time.time()
            
            # If it's a partial (final) batch, slice the arrays
            if is_final and self.current_index < self.batch_size:
                boards_tensor = torch.from_numpy(self.batch_data['boards'][:self.current_index])
                actions_tensor = torch.from_numpy(self.batch_data['actions'][:self.current_index])
                values_tensor = torch.from_numpy(self.batch_data['values'][:self.current_index])
                history_tensor = torch.from_numpy(self.batch_data['history'][:self.current_index])
                players_tensor = torch.from_numpy(self.batch_data['players'][:self.current_index])
            else:
                boards_tensor = torch.from_numpy(self.batch_data['boards'])
                actions_tensor = torch.from_numpy(self.batch_data['actions'])
                values_tensor = torch.from_numpy(self.batch_data['values'])
                history_tensor = torch.from_numpy(self.batch_data['history'])
                players_tensor = torch.from_numpy(self.batch_data['players'])

            batch_dict = {
                'boards': boards_tensor,
                'actions': actions_tensor,
                'values': values_tensor,
                'history': history_tensor,
                'players': players_tensor,
            }

            output_file = self.output_dir / f"batch_{batch_id:06d}.pt"
            torch.save(batch_dict, output_file)

            elapsed = time.time() - start_time
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            log.info(f"Saved batch {batch_id} in {elapsed:.2f}s ({file_size_mb:.1f}MB)")

            self.batch_count += 1

        except Exception as e:
            log.error(f"Error saving batch {batch_id}: {e}")
            raise
        finally:
            # Pre-allocate a new empty batch for subsequent transitions
            if not is_final:
                self._preallocate_batch()
            self.is_saving_file = False

    def finalize(self):
        """Saves any remaining transitions in the final, possibly partial, batch."""
        if self.current_index > 0:
            log.info(f"Finalizing... Saving last batch with {self.current_index} remaining transitions.")
            self._save_batch(is_final=True)
        log.info("All batches have been saved.")


def process_game_chunk(args: Tuple[List[Dict], DictConfig, int, mp.Queue]):
    """
    Processes a sub-chunk of games and puts the transitions list for each game onto a queue.
    
    Args:
        args: A tuple containing (sub_chunk_data, config, worker_id, queue)
    """
    sub_chunk, config, worker_id, queue = args
    games_processed, games_skipped = 0, 0
    
    try:
        processor = GameProcessor(config)
        for game_data in sub_chunk:
            try:
                transitions = processor.process_game(game_data['moves_uci_ints'], game_data['winner'])
                if transitions:
                    queue.put(transitions)
                    games_processed += 1
                else:
                    games_skipped += 1
            except Exception as e:
                log.error(f"Worker {worker_id}: Error processing a game: {e}")
                games_skipped += 1
                continue
    except Exception as e:
        log.error(f"Worker {worker_id}: A critical error occurred: {e}")
    finally:
        # Return statistics for this sub-chunk
        queue.put((games_processed, games_skipped))


class ChessDatasetRenderer:
    """Main class for rendering chess dataset to Luna-compatible training data with multiprocessing."""
    
    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize the dataset renderer."""
        self.config = config or DictConfig({})
        
        # Performance settings
        perf_config = self.config.get('performance', {})
        self.num_processes = perf_config.get('num_processes', mp.cpu_count())
        self.chunk_split_factor = perf_config.get('chunk_split_factor', 4)
        
        # Input/Output settings
        input_config = self.config.get('input', {})
        output_config = self.config.get('output', {})
        
        self.chunk_size = input_config.get('chunk_size', 1000)
        self.batch_size = output_config.get('batch_size', 10000)
        self.max_games = input_config.get('max_games', None)
        self.output_dir = output_config.get('output_dir', 'processed_data')
        
        # Statistics
        self.games_processed = 0
        self.transitions_created = 0
        self.games_skipped = 0
        
        log.info(f"Initialized renderer with {self.num_processes} processes.")

    @classmethod
    def from_config_file(cls, config_path: str):
        """Create renderer from configuration file."""
        config = OmegaConf.load(config_path)
        return cls(config)

    def split_chunk_for_multiprocessing(self, chunk_df: pl.DataFrame) -> List[List[Dict]]:
        """Split a chunk into smaller sub-chunks for parallel processing."""
        rows = chunk_df.to_dicts()
        
        # Convert UCI strings to integers for memory efficiency
        for row in rows:
            if 'moves_uci' in row:
                row['moves_uci_ints'] = convert_uci_strings_to_ints(row['moves_uci'])
                # Remove original string data to save memory
                del row['moves_uci']
        
        num_sub_chunks = self.num_processes * self.chunk_split_factor
        if not num_sub_chunks: return []
        sub_chunk_size = (len(rows) + num_sub_chunks - 1) // num_sub_chunks
        
        sub_chunks = [rows[i:i + sub_chunk_size] for i in range(0, len(rows), sub_chunk_size)]
        return [chunk for chunk in sub_chunks if chunk]

    def _process_queue(self, queue: mp.Queue, batch_manager: NumpyBatchManager, num_workers: int):
        """Consumes transitions lists and stats from the queue until all workers are done."""
        workers_done = 0
        with tqdm(total=self.batch_size, desc="  Transitions", unit="trans", leave=False) as pbar_transitions:
            while workers_done < num_workers:
                item = queue.get()
                if isinstance(item, tuple):  # This is a stats tuple (games_processed, games_skipped)
                    games_proc, games_skip = item
                    self.games_processed += games_proc
                    self.games_skipped += games_skip
                    workers_done += 1
                elif isinstance(item, list):  # This is a transitions list from a game
                    for transition in item:
                        batch_manager.add_transition(transition)
                        self.transitions_created += 1
                    pbar_transitions.update(len(item))

    def process_chunk_parallel(self, chunk_df: pl.DataFrame, batch_manager: NumpyBatchManager):
        """Process a chunk of games using a queue and a multiprocessing pool."""
        sub_chunks = self.split_chunk_for_multiprocessing(chunk_df)

        if not sub_chunks:
            return

        with mp.Manager() as manager:
            # Use mp.Queue for better performance (no manager needed)
            queue = manager.Queue()
            chunk_args = [(sub_chunk, self.config, i, queue) for i, sub_chunk in enumerate(sub_chunks)]
            
            # Start worker processes
            pool = manager.Pool(processes=self.num_processes)
            pool.map_async(process_game_chunk, chunk_args)
            
            # Consume from the queue in the main process
            self._process_queue(queue, batch_manager, len(sub_chunks))
            
            # Clean up the pool
            pool.close()
            pool.join()

    def render_dataset(self, parquet_path: str, output_dir: Optional[str] = None):
        """Main method to render the entire dataset with multiprocessing."""
        if output_dir:
            self.output_dir = output_dir
        
        log.info("="*60)
        log.info("STARTING DATASET RENDERING")
        log.info(f"Output directory: {self.output_dir}")
        
        start_time = time.time()
        batch_manager = NumpyBatchManager(self.batch_size, self.output_dir, self.config)
        
        total_games = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()
        
        with tqdm(total=total_games, desc="Total Progress", unit="games") as pbar:
            for offset in range(0, total_games, self.chunk_size):
                if self.max_games and pbar.n >= self.max_games:
                    log.info(f"Reached max_games limit of {self.max_games}.")
                    break
                
                chunk_df = pl.scan_parquet(parquet_path).slice(offset, self.chunk_size).collect()
                if chunk_df.is_empty():
                    break
                
                self.process_chunk_parallel(chunk_df, batch_manager)
                pbar.update(len(chunk_df))
        
        batch_manager.finalize()
        
        total_time = time.time() - start_time
        log.info("="*60)
        log.info("DATASET RENDERING COMPLETED!")
        log.info(f"Total time: {total_time:.2f} seconds")
        log.info(f"Total games processed: {self.games_processed:,}")
        log.info(f"Total transitions created: {self.transitions_created:,}")
        log.info(f"Total games skipped: {self.games_skipped:,}")
        log.info(f"Total batches saved: {batch_manager.batch_count}")
        log.info(f"Final memory usage: {get_memory_usage():.1f}MB")
        log.info("="*60)
    