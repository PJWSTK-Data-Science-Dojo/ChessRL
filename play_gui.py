"""
GUI for playing against the trained chess model
"""

import os
import sys
import argparse
import time
import pygame
import chess
import torch
import numpy as np

from ChessRL.config import Config
from ChessRL.alpha_net import ChessNetwork
from ChessRL.mcts import MCTS
from ChessRL.encoding import encode_board, move_to_index

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_SQUARE = (118, 150, 86)  # Green
LIGHT_SQUARE = (238, 238, 210)  # Cream
HIGHLIGHT = (186, 202, 43)     # Light green
MOVE_HIGHLIGHT = (255, 255, 0, 150)  # Yellow with transparency
CHECK_HIGHLIGHT = (255, 0, 0, 150)    # Red with transparency

class ChessGUI:
    def __init__(self, config, model_path=None):
        """
        Initialize the chess GUI
        
        Args:
            config (Config): Configuration parameters
            model_path (str, optional): Path to the model to load
        """
        self.config = config
        self.board_size = config.board_size
        self.square_size = self.board_size // 8
        
        # Set up the display
        self.screen = pygame.display.set_mode((self.board_size, self.board_size))
        pygame.display.set_caption('Chess Reinforcement Learning')
        
        # Load the chess pieces
        self.pieces = {}
        self.load_pieces()
        
        # Initialize the chess board
        self.board = chess.Board()
        
        # Game state variables
        self.selected_square = None
        self.valid_moves = []
        self.game_over = False
        self.player_color = chess.WHITE
        self.ai_thinking = False
        self.last_move = None
        
        # Load the AI model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.network = ChessNetwork(config)
        self.network.to(self.device)
        
        if model_path:
            try:
                checkpoint = torch.load(os.path.join(config.model_dir, model_path), map_location=self.device)
                self.network.load_state_dict(checkpoint['network_state_dict'])
                self.network.eval()
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Playing with untrained model")
        else:
            print("No model specified. Playing with untrained model.")
        
        # MCTS will automatically detect the device from the network
        self.mcts = MCTS(self.network, config)
    
    def load_pieces(self):
        """Load chess piece images"""
        piece_chars = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
        
        # Create fallback pieces using pygame shapes if images can't be loaded
        for piece_char in piece_chars:
            # Create a surface for the piece
            piece_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            
            # Draw based on piece type
            color = WHITE if piece_char.isupper() else BLACK
            if piece_char.lower() == 'p':  # Pawn
                self.draw_pawn(piece_surface, color)
            elif piece_char.lower() == 'n':  # Knight
                self.draw_knight(piece_surface, color)
            elif piece_char.lower() == 'b':  # Bishop
                self.draw_bishop(piece_surface, color)
            elif piece_char.lower() == 'r':  # Rook
                self.draw_rook(piece_surface, color)
            elif piece_char.lower() == 'q':  # Queen
                self.draw_queen(piece_surface, color)
            elif piece_char.lower() == 'k':  # King
                self.draw_king(piece_surface, color)
            
            self.pieces[piece_char] = piece_surface
    
    def draw_pawn(self, surface, color):
        """Draw a pawn on the surface"""
        size = self.square_size
        # Draw a small circle for the head
        pygame.draw.circle(surface, color, (size//2, size//3), size//6)
        # Draw a trapezoid for the body
        points = [(size//3, size//3), (size*2//3, size//3), (size*3//4, size*3//4), (size//4, size*3//4)]
        pygame.draw.polygon(surface, color, points)
    
    def draw_knight(self, surface, color):
        """Draw a knight on the surface"""
        size = self.square_size
        # Simple knight shape
        points = [(size//4, size*3//4), (size//3, size//3), (size//2, size//4), 
                  (size*2//3, size//3), (size*3//4, size//2), (size*2//3, size*3//4)]
        pygame.draw.polygon(surface, color, points)
        # Eye
        pygame.draw.circle(surface, BLACK if color == WHITE else WHITE, 
                           (size//2, size*2//5), size//12)
    
    def draw_bishop(self, surface, color):
        """Draw a bishop on the surface"""
        size = self.square_size
        # Bishop body
        pygame.draw.polygon(surface, color, 
                           [(size//3, size*3//4), (size*2//3, size*3//4), 
                            (size//2, size//4)])
        # Base
        pygame.draw.rect(surface, color, (size//4, size*3//4, size//2, size//8))
        # Cross
        pygame.draw.rect(surface, BLACK if color == WHITE else WHITE, 
                        (size*3//8, size//3, size//4, size//12))
    
    def draw_rook(self, surface, color):
        """Draw a rook on the surface"""
        size = self.square_size
        # Rook body
        pygame.draw.rect(surface, color, (size//4, size//3, size//2, size*2//5))
        # Battlement
        for i in range(3):
            x = size//4 + i * size//6
            pygame.draw.rect(surface, color, (x, size//4, size//8, size//8))
        # Base
        pygame.draw.rect(surface, color, (size//5, size*3//4, size*3//5, size//8))
    
    def draw_queen(self, surface, color):
        """Draw a queen on the surface"""
        size = self.square_size
        # Queen body
        pygame.draw.polygon(surface, color, 
                           [(size//4, size*3//4), (size*3//4, size*3//4), 
                            (size*2//3, size//3), (size//2, size//4), 
                            (size//3, size//3)])
        # Crown points
        for i in range(3):
            x = size//3 + i * size//6
            pygame.draw.circle(surface, color, (x, size//5), size//12)
        # Base
        pygame.draw.rect(surface, color, (size//5, size*3//4, size*3//5, size//8))
    
    def draw_king(self, surface, color):
        """Draw a king on the surface"""
        size = self.square_size
        # King body
        pygame.draw.polygon(surface, color, 
                           [(size//4, size*3//4), (size*3//4, size*3//4), 
                            (size*2//3, size//3), (size//3, size//3)])
        # Cross
        pygame.draw.rect(surface, color, (size*3//8, size//6, size//4, size//3))
        pygame.draw.rect(surface, color, (size//3, size//4, size//3, size//8))
        # Base
        pygame.draw.rect(surface, color, (size//5, size*3//4, size*3//5, size//8))
    
    def draw_board(self):
        """Draw the chess board and pieces"""
        # Draw squares
        for row in range(8):
            for col in range(8):
                x = col * self.square_size
                y = (7 - row) * self.square_size if self.player_color == chess.WHITE else row * self.square_size
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, (x, y, self.square_size, self.square_size))
                
                # Highlight selected square
                if self.selected_square is not None:
                    selected_row = chess.square_rank(self.selected_square)
                    selected_col = chess.square_file(self.selected_square)
                    if row == selected_row and col == selected_col:
                        pygame.draw.rect(self.screen, HIGHLIGHT, (x, y, self.square_size, self.square_size))
                
                # Highlight valid moves
                square = chess.square(col, row)
                if square in self.valid_moves:
                    # Create a transparent surface for highlighting
                    highlight_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                    highlight_surface.fill(MOVE_HIGHLIGHT)
                    self.screen.blit(highlight_surface, (x, y))
                
                # Highlight the king if in check
                if self.board.is_check():
                    king_square = self.board.king(self.board.turn)
                    king_row = chess.square_rank(king_square)
                    king_col = chess.square_file(king_square)
                    if row == king_row and col == king_col:
                        # Create a transparent surface for highlighting
                        check_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                        check_surface.fill(CHECK_HIGHLIGHT)
                        self.screen.blit(check_surface, (x, y))
                
                # Highlight last move
                if self.last_move:
                    from_square = self.last_move.from_square
                    to_square = self.last_move.to_square
                    from_row = chess.square_rank(from_square)
                    from_col = chess.square_file(from_square)
                    to_row = chess.square_rank(to_square)
                    to_col = chess.square_file(to_square)
                    
                    if (row == from_row and col == from_col) or (row == to_row and col == to_col):
                        # Draw a border around the square
                        pygame.draw.rect(self.screen, (255, 165, 0), 
                                         (x, y, self.square_size, self.square_size), 3)
        
        # Draw pieces
        for row in range(8):
            for col in range(8):
                square = chess.square(col, row)
                piece = self.board.piece_at(square)
                if piece:
                    x = col * self.square_size
                    y = (7 - row) * self.square_size if self.player_color == chess.WHITE else row * self.square_size
                    self.screen.blit(self.pieces[piece.symbol()], (x, y))
        
        # Draw game over message
        if self.game_over:
            font = pygame.font.SysFont('Arial', 30)
            result = self.board.result()
            if result == '1-0':
                text = font.render('White wins!', True, BLACK)
            elif result == '0-1':
                text = font.render('Black wins!', True, BLACK)
            else:
                text = font.render('Draw!', True, BLACK)
            
            text_rect = text.get_rect(center=(self.board_size//2, self.board_size//2))
            # Semi-transparent background
            bg_surface = pygame.Surface((text_rect.width + 20, text_rect.height + 20), pygame.SRCALPHA)
            bg_surface.fill((255, 255, 255, 200))
            self.screen.blit(bg_surface, (text_rect.x - 10, text_rect.y - 10))
            self.screen.blit(text, text_rect)
        
        # Show "AI thinking" message
        if self.ai_thinking:
            font = pygame.font.SysFont('Arial', 20)
            text = font.render('AI is thinking...', True, BLACK)
            text_rect = text.get_rect(center=(self.board_size//2, 20))
            # Semi-transparent background
            bg_surface = pygame.Surface((text_rect.width + 20, text_rect.height + 10), pygame.SRCALPHA)
            bg_surface.fill((255, 255, 255, 200))
            self.screen.blit(bg_surface, (text_rect.x - 10, text_rect.y - 5))
            self.screen.blit(text, text_rect)
    
    def screen_to_square(self, pos):
        """Convert screen coordinates to chess square"""
        x, y = pos
        file_idx = x // self.square_size
        rank_idx = 7 - (y // self.square_size) if self.player_color == chess.WHITE else y // self.square_size
        if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
            return chess.square(file_idx, rank_idx)
        return None
    
    def get_valid_moves_from(self, square):
        """Get valid moves from the selected square"""
        valid_moves = []
        for move in self.board.legal_moves:
            if move.from_square == square:
                valid_moves.append(move.to_square)
        return valid_moves
    
    def ai_move(self):
        """Make a move with the AI"""
        self.ai_thinking = True
        pygame.display.flip()
        
        # Run MCTS to find the best move
        root = self.mcts.search(self.board, is_training=False)
        
        # Select the move with the highest visit count
        if not root.children:
            # No legal moves
            self.ai_thinking = False
            return
            
        best_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        
        # Make the move
        self.board.push(best_move)
        self.last_move = best_move
        
        # Check if the game is over
        if self.board.is_game_over():
            self.game_over = True
        
        self.ai_thinking = False
    
    def handle_click(self, pos):
        """Handle mouse click on the board"""
        if self.game_over or self.ai_thinking or self.board.turn != self.player_color:
            return
            
        square = self.screen_to_square(pos)
        if square is None:
            return
            
        piece = self.board.piece_at(square)
        
        # If a square is already selected
        if self.selected_square is not None:
            # If the clicked square is a valid move target
            if square in self.valid_moves:
                # Create the move
                move = chess.Move(self.selected_square, square)
                
                # Check for promotion
                if (self.board.piece_at(self.selected_square).piece_type == chess.PAWN and 
                   ((chess.square_rank(square) == 7 and self.board.turn == chess.WHITE) or 
                    (chess.square_rank(square) == 0 and self.board.turn == chess.BLACK))):
                    move = chess.Move(self.selected_square, square, chess.QUEEN)
                
                # Make the move
                self.board.push(move)
                self.last_move = move
                
                # Check if the game is over
                if self.board.is_game_over():
                    self.game_over = True
                # Otherwise, it's the AI's turn
                else:
                    # Schedule AI move
                    pygame.time.set_timer(pygame.USEREVENT, 100)
            
            # Clear selection
            self.selected_square = None
            self.valid_moves = []
            
        # If no square is selected and the clicked square has a piece of the player's color
        elif piece is not None and (piece.color == self.player_color):
            self.selected_square = square
            self.valid_moves = self.get_valid_moves_from(square)
    
    def run(self):
        """Run the chess GUI"""
        clock = pygame.time.Clock()
        running = True
        
        # If AI is black, it goes first
        if self.player_color == chess.BLACK:
            pygame.time.set_timer(pygame.USEREVENT, 100)
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # Reset the game
                        self.board = chess.Board()
                        self.selected_square = None
                        self.valid_moves = []
                        self.game_over = False
                        self.last_move = None
                        # If AI is black, it goes first
                        if self.player_color == chess.BLACK:
                            pygame.time.set_timer(pygame.USEREVENT, 100)
                    elif event.key == pygame.K_s:
                        # Switch sides
                        self.player_color = not self.player_color
                        # Reset the game
                        self.board = chess.Board()
                        self.selected_square = None
                        self.valid_moves = []
                        self.game_over = False
                        self.last_move = None
                        # If AI is white, it goes first
                        if self.player_color == chess.BLACK:
                            pygame.time.set_timer(pygame.USEREVENT, 100)
                elif event.type == pygame.USEREVENT:
                    # Cancel the timer
                    pygame.time.set_timer(pygame.USEREVENT, 0)
                    if not self.game_over and self.board.turn != self.player_color:
                        self.ai_move()
            
            # Draw the board
            self.draw_board()
            
            # Update the display
            pygame.display.flip()
            
            # Limit to 60 FPS
            clock.tick(60)
        
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='Play against the trained chess model')
    parser.add_argument('--model', type=str, default=None, help='Model file to use (in the models directory)')
    args = parser.parse_args()
    
    config = Config()
    
    # For GUI play, reduce MCTS simulations for faster response
    config.num_simulations = 100
    
    gui = ChessGUI(config, args.model)
    gui.run()

if __name__ == "__main__":
    main()
