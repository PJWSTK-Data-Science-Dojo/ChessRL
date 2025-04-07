"""
Optimized self-play implementation with better multiprocessing compatibility
"""

import numpy as np
import chess
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import copy
import time
import signal

from ChessRL.mcts import MCTS
from ChessRL.encoding import encode_board, move_to_index

global_network = None
global_config = None
global_device = None
global_mcts = None

def initialize_worker(network_state_dict, config_dict, device_str):
    """Initialize the worker process with the neural network and config"""
    global global_network
    global global_config
    global global_device
    global global_mcts
    
    from ChessRL.alpha_net import ChessNetwork
    from ChessRL.config import Config
    from ChessRL.mcts import MCTS
    
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    global_config = Config()
    for key, value in config_dict.items():
        if hasattr(global_config, key):
            setattr(global_config, key, value)
    
    # Set environment variables for better CPU performance
    os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
    os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads
    
    global_device = torch.device(device_str)
    global_network = ChessNetwork(global_config)
    global_network.to(global_device)
    global_network.load_state_dict(network_state_dict)
    global_network.eval()  # Set to evaluation mode
    
    global_mcts = MCTS(global_network, global_config)
    
    if global_device.type == 'cuda' and hasattr(global_device, 'index'):
        torch.cuda.set_device(global_device.index)
        # Enable TensorFloat32 for faster computation
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def run_self_play_game(game_idx):
    """
    Run a single self-play game in a separate process
    
    Args:
        game_idx (int): Index of the game for logging
    
    Returns:
        list: Game history with states, policies, and values
    """
    global global_network
    global global_config
    global global_device
    global global_mcts
    
    board = chess.Board()
    game_history = []
    
    move_count = 0
    
    game_start_time = time.time()
    last_time = time.time()
    move_times = []
    
    while not board.is_game_over():
        move_start = time.time()
        
        # Temperature parameter for move selection
        temp = global_config.temperature
        if move_count >= global_config.temperature_threshold:
            temp = 0.1  # Almost deterministic selection
        
        # Run MCTS
        root = global_mcts.search(board)
        
        # Select move based on visit counts and temperature
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        actions = list(root.children.keys())
        
        if temp == 0 or len(actions) == 1:
            # Choose the move with highest visit count
            best_idx = np.argmax(visit_counts)
            action = actions[best_idx]
        else:
            # Apply temperature and sample
            visit_count_distribution = visit_counts ** (1 / temp)
            sum_visits = np.sum(visit_count_distribution)
            if sum_visits > 0:
                visit_count_distribution = visit_count_distribution / sum_visits
                action_idx = np.random.choice(len(actions), p=visit_count_distribution)
                action = actions[action_idx]
            else:
                # Fallback to highest count if numerical issues
                best_idx = np.argmax(visit_counts)
                action = actions[best_idx]
        
        # Store the current state, MCTS policy, and turn
        encoded_board = encode_board(board)
        
        # Efficiently create policy vector
        policy = np.zeros(4672, dtype=np.float32)  # Size of policy vector
        for i, a in enumerate(actions):
            try:
                move_idx = move_to_index(a)
                if 0 <= move_idx < 4672:
                    policy[move_idx] = visit_counts[i]
            except Exception:
                continue
                
        # Normalize policy
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Fallback to uniform policy
            legal_moves = list(board.legal_moves)
            indices = [move_to_index(move) for move in legal_moves]
            for idx in indices:
                if 0 <= idx < 4672:
                    policy[idx] = 1.0 / len(legal_moves)
        
        game_history.append({
            'board': encoded_board.cpu().numpy() if global_device.type == 'cuda' else encoded_board.numpy(),
            'policy': policy,
            'turn': board.turn
        })
        
        # Execute the chosen move
        board.push(action)
        move_count += 1
        
        # Calculate move time
        move_time = time.time() - move_start
        move_times.append(move_time)
        
        # Clean GPU memory periodically
        if move_count % 10 == 0:
            if global_device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Log performance
            current_time = time.time()
            elapsed = current_time - last_time
            avg_move_time = elapsed / min(10, move_count)
            print(f"Game {game_idx}, Move {move_count}: {avg_move_time:.3f}s per move")
            last_time = current_time
    
    # Game result
    result = board.result()
    winner = None
    if result == "1-0":
        winner = chess.WHITE
    elif result == "0-1":
        winner = chess.BLACK
    
    # Add game result to all stored states
    for state in game_history:
        if winner is None:  # Draw
            state['value'] = 0
        else:
            # Win: 1, Loss: -1 (from perspective of the player who made the move)
            state['value'] = 1 if state['turn'] == winner else -1
    
    # Game statistics
    game_time = time.time() - game_start_time
    avg_move_time = sum(move_times) / len(move_times) if move_times else 0
    print(f"Game {game_idx} completed: {move_count} moves in {game_time:.2f}s ({avg_move_time:.3f}s per move)")
    print(f"Game {game_idx} result: {result}")
    
    # Clean GPU memory
    if global_device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return game_history

class SelfPlay:
    """
    Optimized self-play for generating training data
    """
    def __init__(self, network, config):
        """
        Initialize the self-play module
        
        Args:
            network (ChessNetwork): The neural network for position evaluation
            config (Config): Configuration parameters
        """
        self.network = network
        self.config = config
        self.device = next(network.parameters()).device
        
        # Create optimized MCTS
        self.mcts = MCTS(network, config)
        
        # Determine optimal number of processes based on hardware
        if self.device.type == 'cuda':
            # Check for GPU memory
            if torch.cuda.get_device_properties(0).total_memory > 20e9:  # More than 20GB VRAM
                # For high-end GPUs, use fewer processes to maximize GPU utilization
                self.num_processes = 2
            else:
                # For standard GPUs, use more processes
                self.num_processes = min(os.cpu_count() // 2, 4)
        else:
            # For CPU, allocate fewer processes to avoid oversubscription
            self.num_processes = max(1, os.cpu_count() // 2)

        self.num_processes = 4
            
        print(f"Using {self.num_processes} processes for self-play")
        
        # Enable CUDA optimizations if available
        if self.device.type == 'cuda':
            # Enable TensorFloat32 for faster computation
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set optimal CUDA settings
            torch.backends.cudnn.benchmark = True
    
    def execute_episode(self):
        """
        Play a full game of self-play with optimized efficiency
        
        Returns:
            list: Game history with states, policies, and values
        """
        # Pre-allocate memory and warm up GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        board = chess.Board()
        game_history = []
        move_count = 0
        
        # Set thread optimizations
        torch.set_num_threads(4)  # Limit threads to reduce overhead
        
        print("Playing self-play game...")
        game_start_time = time.time()
        
        # For tracking move times
        move_times = []
        
        while not board.is_game_over():
            move_start = time.time()
            
            # Temperature parameter for move selection
            temp = self.config.temperature
            if move_count >= self.config.temperature_threshold:
                temp = 0.1  # Almost deterministic selection
            
            # Run MCTS
            root = self.mcts.search(board)
            
            # Select move based on visit counts and temperature
            visit_counts = np.array([child.visit_count for child in root.children.values()])
            actions = list(root.children.keys())
            
            if temp == 0 or len(actions) == 1:
                best_idx = np.argmax(visit_counts)
                action = actions[best_idx]
            else:
                # Apply temperature and sample
                visit_count_distribution = visit_counts ** (1 / temp)
                visit_count_distribution = visit_count_distribution / np.sum(visit_count_distribution)
                action_idx = np.random.choice(len(actions), p=visit_count_distribution)
                action = actions[action_idx]
            
            # Store the current state, MCTS policy, and turn
            policy = np.zeros(4672, dtype=np.float32)  # Size of policy vector
            
            # Use vectorized operations for efficiency
            for i, a in enumerate(actions):
                try:
                    move_idx = move_to_index(a)
                    if 0 <= move_idx < 4672:
                        policy[move_idx] = visit_counts[i]
                except Exception:
                    continue
            
            # Normalize policy
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy = policy / policy_sum
            
            # Store state information
            with torch.no_grad():
                encoded_board = encode_board(board, self.device)
            
            game_history.append({
                'board': encoded_board,
                'policy': policy,
                'turn': board.turn
            })
            
            # Execute the chosen move
            board.push(action)
            move_count += 1
            
            # Track move time
            move_time = time.time() - move_start
            move_times.append(move_time)
            
            # Clean GPU memory periodically
            if move_count % 10 == 0:
                # Calculate and log average time for last 10 moves
                avg_time = sum(move_times[-10:]) / len(move_times[-10:])
                print(f"Move {move_count}: {avg_time:.3f}s per move")
                
                # Force GPU memory cleanup
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            if move_count % 25 == 0:
                print(board)
        
        # Game result
        result = board.result()
        winner = None
        if result == "1-0":
            winner = chess.WHITE
        elif result == "0-1":
            winner = chess.BLACK
            
        game_time = time.time() - game_start_time
        avg_move_time = sum(move_times) / len(move_times) if move_times else 0
        print(f"Game completed in {game_time:.1f}s ({move_count} moves, {avg_move_time:.3f}s per move)", flush=True)
        print(f"Game result: {result}", flush=True)
        
        # Add game result to all stored states
        for state in game_history:
            if winner is None:  # Draw
                state['value'] = 0
            else:
                # Win: 1, Loss: -1 (from perspective of the player who made the move)
                state['value'] = 1 if state['turn'] == winner else -1
        
        return game_history
    
    def execute_parallel_self_play(self, num_games):
        """
        Execute multiple self-play games in parallel with optimized resource usage
        
        Args:
            num_games (int): Number of games to play
            
        Returns:
            list: Combined game history from all games
        """
        # Check if we should use single-process optimization for high-end GPUs
        if self.device.type == 'cuda' and torch.cuda.get_device_properties(0).total_memory > 20e9:
            # For high-end GPUs, single-process is often more efficient
            print("Using optimized single-process self-play for high-end GPU")
            all_game_history = []
            for i in range(num_games):
                print(f"Playing game {i+1}/{num_games}")
                game_history = self.execute_episode()
                all_game_history.extend(game_history)
                
                # Force GPU memory cleanup
                torch.cuda.empty_cache()
                
            return all_game_history
        
        # For lower-end hardware, use multi-process approach
        # Prepare the network state dict (CPU version for sharing)
        network_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        
        # Prepare config as a dictionary
        config_dict = {
            'num_simulations': self.config.num_simulations,
            'c_puct': self.config.c_puct,
            'dirichlet_alpha': self.config.dirichlet_alpha,
            'dirichlet_noise_factor': self.config.dirichlet_noise_factor,
            'temperature': self.config.temperature,
            'temperature_threshold': self.config.temperature_threshold,
            'n_features': self.config.n_features,
            'n_residual_blocks': self.config.n_residual_blocks,
            'batch_mcts_size': self.config.batch_mcts_size
        }
        
        # Clear GPU memory before parallel processing
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        all_game_history = []
        start_time = time.time()
        
        # Use different processes based on whether we have GPU
        if self.device.type == 'cuda':
            # For GPU, use CPU workers to avoid CUDA issues in subprocesses
            device_str = 'cpu'
            
            try:
                # Set multiprocessing start method
                multiprocessing.set_start_method('spawn', force=True)
                
                # Create a process pool with CPU workers
                with ProcessPoolExecutor(max_workers=self.num_processes, 
                                      initializer=initialize_worker,
                                      initargs=(network_state_dict, config_dict, device_str)) as executor:
                    
                    # Submit games in batches to control memory usage
                    batch_size = min(8, num_games)
                    completed = 0
                    
                    for batch_start in range(0, num_games, batch_size):
                        batch_end = min(batch_start + batch_size, num_games)
                        current_batch = range(batch_start, batch_end)
                        
                        # Submit batch
                        game_futures = [executor.submit(run_self_play_game, i) for i in current_batch]
                        
                        # Process batch results
                        for future in game_futures:
                            try:
                                game_data = future.result()
                                all_game_history.extend(game_data)
                                completed += 1
                                
                                # Report progress periodically
                                elapsed = time.time() - start_time
                                games_per_hour = (completed / elapsed) * 3600
                                print(f"Completed {completed}/{num_games} games "
                                    f"({games_per_hour:.1f} games/hour)")
                            except Exception as e:
                                print(f"Error in game: {e}")
            except Exception as e:
                print(f"Error in parallel execution: {e}")
                # Fallback to single-process execution
                print("Falling back to sequential execution")
                all_game_history = []
                for i in range(num_games):
                    print(f"Running game {i+1}/{num_games}")
                    game_data = self.execute_episode()
                    all_game_history.extend(game_data)
        else:
            # For CPU-only, use simpler thread pool
            def thread_self_play(game_idx):
                """Function to run in thread pool"""
                print(f"Starting game {game_idx+1}/{num_games}")
                # Create a fresh MCTS instance for each thread
                local_mcts = MCTS(self.network, self.config)
                
                # Play game
                board = chess.Board()
                game_history = []
                move_count = 0
                
                while not board.is_game_over():
                    # Run MCTS
                    root = local_mcts.search(board)
                    
                    # Select move (same logic as before)
                    temp = self.config.temperature
                    if move_count >= self.config.temperature_threshold:
                        temp = 0.1
                        
                    visit_counts = np.array([child.visit_count for child in root.children.values()])
                    actions = list(root.children.keys())
                    
                    if temp == 0 or len(actions) == 1:
                        best_idx = np.argmax(visit_counts)
                        action = actions[best_idx]
                    else:
                        visit_count_distribution = visit_counts ** (1 / temp)
                        sum_visits = np.sum(visit_count_distribution)
                        if sum_visits > 0:
                            visit_count_distribution = visit_count_distribution / sum_visits
                            action_idx = np.random.choice(len(actions), p=visit_count_distribution)
                            action = actions[action_idx]
                        else:
                            best_idx = np.argmax(visit_counts)
                            action = actions[best_idx]
                    
                    encoded_board = encode_board(board)
                    policy = np.zeros(4672, dtype=np.float32)
                    
                    for i, a in enumerate(actions):
                        try:
                            move_idx = move_to_index(a)
                            if 0 <= move_idx < 4672:
                                policy[move_idx] = visit_counts[i]
                        except Exception:
                            continue
                            
                    policy_sum = np.sum(policy)
                    if policy_sum > 0:
                        policy = policy / policy_sum
                    
                    game_history.append({
                        'board': encoded_board.numpy(),
                        'policy': policy,
                        'turn': board.turn
                    })
                    
                    # Make the move
                    board.push(action)
                    move_count += 1
                
                # Process game result
                result = board.result()
                winner = None
                if result == "1-0":
                    winner = chess.WHITE
                elif result == "0-1":
                    winner = chess.BLACK
                
                # Add values
                for state in game_history:
                    if winner is None:
                        state['value'] = 0
                    else:
                        state['value'] = 1 if state['turn'] == winner else -1
                
                print(f"Completed game {game_idx+1}/{num_games}: {result} ({move_count} moves)")
                return game_history
            
            with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
                game_futures = [executor.submit(thread_self_play, i) for i in range(num_games)]
                
                completed = 0
                for future in game_futures:
                    try:
                        game_data = future.result()
                        all_game_history.extend(game_data)
                        completed += 1
                        
                        if completed % max(1, num_games // 10) == 0:
                            elapsed = time.time() - start_time
                            games_per_hour = (completed / elapsed) * 3600
                            print(f"Completed {completed}/{num_games} games "
                                f"({games_per_hour:.1f} games/hour)")
                    except Exception as e:
                        print(f"Error in game: {e}")
        
        # Convert NumPy arrays to torch tensors where needed
        for state in all_game_history:
            if isinstance(state['board'], np.ndarray):
                state['board'] = torch.FloatTensor(state['board']).to(self.device)
        
        return all_game_history