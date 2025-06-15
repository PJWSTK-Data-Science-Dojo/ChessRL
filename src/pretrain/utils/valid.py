"""
Simple stockfish validation utility for chess model evaluation.
"""

from collections import deque
import chess
import random
import logging
import wandb
import subprocess
import threading
import os
from typing import Optional, Tuple
from omegaconf import OmegaConf
import pytorch_lightning as pl
from luna.game.luna_game import who
from luna.luna import Luna

# Try to import stockfish library, fall back gracefully if not available
try:
    from stockfish import Stockfish
    STOCKFISH_AVAILABLE = True
except ImportError:
    STOCKFISH_AVAILABLE = False
    Stockfish = None

log = logging.getLogger(__name__)


class StockfishValidator:
    """Simple stockfish validation for chess models using stockfish library."""
    
    def __init__(self, stockfish_path: str = "stockfish", elo: int = 1400):
        """
        Initialize stockfish validator.
        
        Args:
            stockfish_path: Path to stockfish executable
            elo: ELO rating to set for stockfish
        """
        self.stockfish_path = stockfish_path
        self.elo = elo
        self.stockfish_available = self._check_stockfish_available()
        
    def _check_stockfish_available(self) -> bool:
        """Check if Stockfish is available and working."""
        if not STOCKFISH_AVAILABLE:
            log.warning("Stockfish library not installed. Install with: pip install stockfish")
            return False
            
        if not os.path.exists(self.stockfish_path):
            log.warning(f"Stockfish executable not found at {self.stockfish_path}")
            return False
            
        try:
            # Test if stockfish can be initialized
            sf = Stockfish(path=self.stockfish_path)
            sf.set_elo_rating(self.elo)
            return sf.is_fen_valid("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        except Exception as e:
            log.warning(f"Failed to initialize Stockfish: {e}")
            return False
        
    def validate_model(self, model, num_games: int = 10) -> Tuple[int, int, int, list]:
        """
        Validate model against stockfish.
        
        Args:
            model: Chess model with a method to get best move
            num_games: Number of games to play
            
        Returns:
            Tuple of (wins, draws, losses, final_boards) from model's perspective
        """
        if not self.stockfish_available:
            log.warning("Stockfish not available, skipping validation")
            return 0, 0, num_games, []  # Return all losses if Stockfish not available
            
        wins = draws = losses = 0
        final_boards = []
        
        for game_idx in range(num_games):
            result, final_board = self._play_game(model, model_plays_white=(game_idx % 2 == 0))
            final_boards.append(final_board)
            if result == 1:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1
                
        return wins, draws, losses, final_boards
    
    def _play_game(self, model, model_plays_white: bool = True) -> Tuple[int, chess.Board]:
        """Play a single game between model and stockfish. Returns (result, final_board) where result is 1 for win, 0 for draw, -1 for loss."""
        board = chess.Board()
        
        # Initialize Stockfish engine
        stockfish = Stockfish(path=self.stockfish_path)
        stockfish.set_elo_rating(self.elo)
        
        max_moves = 100  # Prevent infinite games
        move_count = 0
        
        while not board.is_game_over() and move_count < max_moves:
            if board.turn == chess.WHITE:
                if model_plays_white:
                    move = self._get_model_move(model, board)
                    if move is None or move not in board.legal_moves:
                        # Model failed to provide valid move
                        return -1, board
                else:
                    # Stockfish plays white
                    stockfish.set_fen_position(board.fen())
                    sf_move = stockfish.get_best_move()
                    if sf_move is None:
                        # Stockfish couldn't find a move (shouldn't happen)
                        return 1, board
                    move = chess.Move.from_uci(sf_move)
            else:
                if not model_plays_white:
                    move = self._get_model_move(model, board)
                    if move is None or move not in board.legal_moves:
                        # Model failed to provide valid move
                        return -1, board
                else:
                    # Stockfish plays black
                    stockfish.set_fen_position(board.fen())
                    sf_move = stockfish.get_best_move()
                    if sf_move is None:
                        # Stockfish couldn't find a move (shouldn't happen)
                        return 1, board
                    move = chess.Move.from_uci(sf_move)
            
            board.push(move)
            move_count += 1
        
        # Determine result from model's perspective
        if board.is_game_over():
            result = board.result()
            if result == "1-0":  # White wins
                return (1 if model_plays_white else -1), board
            elif result == "0-1":  # Black wins
                return (1 if not model_plays_white else -1), board
            else:  # Draw
                return 0, board
        else:
            # Game reached max moves - consider it a draw
            return 0, board
    
    def _get_model_move(self, model, board: chess.Board) -> Optional[chess.Move]:
        """Get move from model. Override this method for different model interfaces."""
        # For Luna model, use the computer_move method
        if hasattr(model, 'board') and hasattr(model, 'computer_move'):
            # Get the current player
            current_player = who(board.turn)
            
            # Create canonical board from current player's perspective
            canonical_board = model.game.getCanonicalForm(board.copy(), current_player)
            canonical_history_array = model.game._board_to_feature_array(canonical_board)
            
            # Set model's internal board state (use the original board, not canonical)
            model.board = board.copy()
            
            # Create history with single canonical frame
            model.history = deque([canonical_history_array], maxlen=model.game.history_len)
            
            move_uci = model.computer_move()
            if move_uci:
                return chess.Move.from_uci(move_uci)
        return None 


class StockfishValidationCallback(pl.Callback):
    """PyTorch Lightning callback for Stockfish validation"""
    
    def __init__(self, stockfish_path: str = "stockfish", validate_every_n_steps: int = 5000, 
                 validation_games: int = 10, stockfish_elo: int = 500):
        super().__init__()
        self.validator = StockfishValidator(stockfish_path=stockfish_path, elo=stockfish_elo)
        self.validate_every_n_steps = validate_every_n_steps
        self.validation_games = validation_games
        self.luna_instance = None
        
        # Check if validation is possible
        if not self.validator.stockfish_available:
            log.warning(f"Stockfish not found at '{stockfish_path}'. Validation will be skipped.")
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Run validation every N steps"""
        # Skip validation on step 0 and if Stockfish is not available
        if trainer.global_step % self.validate_every_n_steps == 0 and self.validator.stockfish_available:
            self._run_stockfish_validation(trainer, pl_module)
    
    def _run_stockfish_validation(self, trainer, pl_module):
        """Run stockfish validation"""
        # Create Luna instance for validation if not exists
        if self.luna_instance is None:
            # Create a minimal config for Luna instance
            luna_cfg = OmegaConf.create({
                'cuda': pl_module.cfg.cuda,
                'model': pl_module.cfg.model,
                'inference': {
                    'load_model': False,
                    'numMCTSSims': 2,  # 2 simulations effectively end up in following the prior policy (only root and one best moves expanded)
                    'cpuct': 1.0,
                    'dir_noise': False
                }
            })
            self.luna_instance = Luna(cfg=luna_cfg)
            
        # Update Luna's model with current trained weights
        self.luna_instance.luna_eval.nnet.load_state_dict(pl_module.model.state_dict())
        
        # Run validation
        log.info("Running validation...")
        wins, draws, losses, final_boards = self.validator.validate_model(self.luna_instance, self.validation_games)
        
        # Log results
        win_rate = wins / self.validation_games
        pl_module.log('stockfish_win_rate', win_rate, on_step=True)
        pl_module.log('stockfish_wins', wins, on_step=True)
        pl_module.log('stockfish_draws', draws, on_step=True)
        pl_module.log('stockfish_losses', losses, on_step=True)
        
        # # Log final board states to Wandb
        # if trainer.logger and hasattr(trainer.logger, 'experiment'):
        #     for i, board in enumerate(final_boards):
        #         # Use chess.Board's SVG representation for Wandb
        #         board_html = f"<div>{board._repr_svg_()}</div>"
        #         trainer.logger.experiment.log({f"final_boards_step_{trainer.global_step}_board{i+1}": wandb.Html(board_html)})
        
        print(f"Stockfish validation (step {trainer.global_step}): {wins}W-{draws}D-{losses}L (WR: {win_rate:.2f})")
        
