import numpy as np
import chess
from core.game import Game


class ChessEnv:
    """Chess environment that follows gym interface"""
    def __init__(self):
        self.board = chess.Board()
        self.action_space = 4672  # Maksymalna teoretyczna liczba ruchów w szachach
        self.reset()
        
    def reset(self):
        self.board = chess.Board()
        return self._get_observation()
        
    def step(self, action):
        # Konwertujemy indeks akcji na ruch szachowy
        move = self._action_to_move(action)
        
        # Wykonujemy ruch
        self.board.push(move)
        
        # Sprawdzamy, czy gra się zakończyła
        done = self.board.is_game_over()
        
        # Obliczamy nagrodę
        reward = self._get_reward(done)
        
        return self._get_observation(), reward, done, {}
    
    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return [seed]
        
    def close(self):
        pass
        
    def _get_observation(self):
        """Konwertuje planszę szachową do tensora 12x8x8"""
        observation = np.zeros((12, 8, 8), dtype=np.float32)
        
        piece_to_index = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                color_idx = 0 if piece.color == chess.WHITE else 6
                piece_idx = piece_to_index[piece.piece_type]
                channel_idx = color_idx + piece_idx
                
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                observation[channel_idx][rank][file] = 1.0
                
        return observation
        
    def _get_reward(self, done):
        """Oblicza nagrodę na podstawie stanu gry"""
        if not done:
            return 0.0
            
        if self.board.is_checkmate():
            # Nagroda +1 za wygraną, -1 za przegraną
            return 1.0 if self.board.turn == chess.BLACK else -1.0
        
        # Remis
        return 0.0
        
    def _action_to_move(self, action_idx):
        """Konwertuje indeks akcji na ruch szachowy"""
        legal_moves = list(self.board.legal_moves)
        if action_idx < len(legal_moves):
            return legal_moves[action_idx]
        # Jeśli indeks jest poza zakresem, zwracamy pierwszy legalny ruch
        return legal_moves[0] if legal_moves else None


class ChessWrapper(Game):

    def __init__(self, env, discount: float, cvt_string=False):
        """Chess Wrapper"""
        super().__init__(env, env.action_space, discount)
        self.cvt_string = cvt_string

    def legal_actions(self):
        """Zwraca legalne akcje"""
        legal_moves = list(self.env.board.legal_moves)
        return [i for i in range(len(legal_moves))]

    def step(self, action):
        """Wykonuje jeden krok"""
        observation, reward, done, info = self.env.step(action)
        
        if self.cvt_string:
            observation = self.env.board.fen()
            
        return observation, reward, done, info

    def reset(self, **kwargs):
        """Resetuje środowisko"""
        observation = self.env.reset(**kwargs)
        
        if self.cvt_string:
            observation = self.env.board.fen()
            
        return observation
        
    def close(self):
        """Zamyka środowisko"""
        self.env.close()
        
    def get_max_episode_steps(self):
        """Maksymalna liczba kroków w epizodzie"""
        return 512
