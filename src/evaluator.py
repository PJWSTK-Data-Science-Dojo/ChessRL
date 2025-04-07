"""
Moduł oceny modeli dla szachowego AlphaZero
"""

import os
import torch
import numpy as np
import copy
import time
import pickle
import chess

from ChessRL.chess_board import board
from ChessRL.encoding import encode_board, decode_action
from ChessRL.mcts import MCTS

class arena:
    """
    Arena do porównania dwóch sieci neuronowych w rozgrywkach szachowych
    """
    def __init__(self, current_chessnet, best_chessnet, mcts_simulations=800, device=None):
        """
        Inicjalizacja areny
        
        Args:
            current_chessnet: Aktualnie trenowana sieć
            best_chessnet: Poprzednia najlepsza sieć
            mcts_simulations: Liczba symulacji MCTS na ruch
            device: Urządzenie do obliczeń (CPU/GPU)
        """
        self.current = current_chessnet
        self.best = best_chessnet
        self.mcts_simulations = mcts_simulations
        
        # Ustaw device - wykryj automatycznie, jeśli nie podano
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Inicjalizuj MCTS dla obu sieci
        self.current_mcts = MCTS(self.current, num_simulations=self.mcts_simulations)
        self.best_mcts = MCTS(self.best, num_simulations=self.mcts_simulations)
    
    def play_round(self):
        """
        Rozegraj jedną partię między sieciami
        
        Returns:
            tuple: (zwycięzca, dane gry)
        """
        # Losowo wybierz, która sieć gra białymi
        if np.random.uniform(0, 1) <= 0.5:
            white_net = self.current_mcts
            black_net = self.best_mcts
            white_name, black_name = "current", "best"
        else:
            white_net = self.best_mcts
            black_net = self.current_mcts
            white_name, black_name = "best", "current"
            
        print(f"Nowa gra: {white_name} (białe) vs {black_name} (czarne)")
        
        # Inicjalizuj szachownicę i dane do zbierania
        current_board = board()
        checkmate = False
        states = []
        dataset = []
        value = 0
        moves_without_progress = 0
        
        # Rozgrywka
        start_time = time.time()
        while not checkmate and current_board.move_count <= 100 and moves_without_progress < 20:
            # Sprawdź remis przez powtórzenie
            draw_counter = 0
            for s in states:
                if np.array_equal(current_board.current_board, s):
                    draw_counter += 1
            if draw_counter >= 3:  # remis przez powtórzenie
                print("Remis przez powtórzenie")
                break
                
            # Zapisz bieżący stan
            states.append(copy.deepcopy(current_board.current_board))
            board_state = copy.deepcopy(encode_board(current_board))
            dataset.append(board_state)
            
            # Wybierz sieć dla bieżącego gracza
            if current_board.player == 0:  # białe
                mcts_engine = white_net
            else:  # czarne
                mcts_engine = black_net
            
            # Wykonaj MCTS i wybierz najlepszy ruch
            best_move, root = mcts_engine.search(current_board)
            
            # Wykonaj ruch
            current_board = self.do_decode_n_move_pieces(current_board, best_move)
            
            # Wyświetl informacje o ruchu
            move_time = time.time() - move_start
            print(f"Ruch #{current_board.move_count}: Czas: {move_time:.2f}s")
            
            # Sprawdź czy partia się skończyła
            if current_board.check_status() and len(current_board.in_check_possible_moves()) == 0:
                if current_board.player == 0:  # czarne wygrywają
                    value = -1
                    winner = black_name
                    print(f"{black_name} (czarne) wygrywają przez mata")
                elif current_board.player == 1:  # białe wygrywają
                    value = 1
                    winner = white_name
                    print(f"{white_name} (białe) wygrywają przez mata")
                checkmate = True
            
            # Sprawdź brak postępu
            # Uproszczona implementacja - liczba ruchów bez bicia lub ruchu pionem
            has_progress = False
            last_move = current_board.last_move if hasattr(current_board, 'last_move') else None
            if last_move and (current_board.current_board[last_move[0]] in ['p', 'P'] or 
                             (current_board.current_board[last_move[0]] != ' ' and 
                              current_board.current_board[last_move[1]] != ' ')):
                has_progress = True
                moves_without_progress = 0
            else:
                moves_without_progress += 1
        
        # Sprawdź pat lub przekroczenie limitu ruchów
        if not checkmate:
            if current_board.move_count > 100 or moves_without_progress >= 20:
                print("Remis przez limit ruchów")
            else:
                print("Remis (pat)")
            value = 0
            winner = None
        
        # Dodaj wynik do wszystkich stanów
        dataset.append(value)
        
        game_time = time.time() - start_time
        print(f"Gra zakończona w {game_time:.2f}s, {current_board.move_count} ruchów")
        
        if winner:
            return winner, dataset
        else:
            return None, dataset
    
    def do_decode_n_move_pieces(self, board, move):
        """
        Zdekoduj i wykonaj ruch na szachownicy
        
        Args:
            board: Stan szachownicy
            move: Zakodowany ruch do wykonania
            
        Returns:
            board: Nowy stan szachownicy po wykonaniu ruchu
        """
        i_pos, f_pos, prom = decode_action(board, move)
        for i, f, p in zip(i_pos, f_pos, prom):
            board.move_piece(i, f, p)
            a, b = i
            c, d = f
            # Jeśli król porusza się o 2 pola, przesuń również wieżę (roszada)
            if board.current_board[c, d] in ["K", "k"] and abs(d - b) == 2:
                if a == 7 and d - b > 0:  # roszada krótka białych
                    board.player = 0
                    board.move_piece((7, 7), (7, 5), None)
                elif a == 7 and d - b < 0:  # roszada długa białych
                    board.player = 0
                    board.move_piece((7, 0), (7, 3), None)
                elif a == 0 and d - b > 0:  # roszada krótka czarnych
                    board.player = 1
                    board.move_piece((0, 7), (0, 5), None)
                elif a == 0 and d - b < 0:  # roszada długa czarnych
                    board.player = 1
                    board.move_piece((0, 0), (0, 3), None)
        return board
    
    def evaluate(self, num_games, cpu_id=0, save_games=True):
        """
        Ocena sieci poprzez rozegranie wielu partii
        
        Args:
            num_games (int): Liczba gier do rozegrania
            cpu_id (int): ID procesora (dla logowania)
            save_games (bool): Czy zapisywać dane z gier
            
        Returns:
            float: Wskaźnik wygranych bieżącej sieci
        """
        current_wins = 0
        best_wins = 0
        draws = 0
        
        for i in range(num_games):
            print(f"\nGra {i+1}/{num_games}")
            winner, dataset = self.play_round()
            
            if save_games:
                # Zapisz dane gry
                self.save_as_pickle(f"evaluate_game_cpu{cpu_id}_{i}.pkl", dataset)
            
            if winner == "current":
                current_wins += 1
                print(f"Bieżąca sieć wygrywa grę {i+1}")
            elif winner == "best":
                best_wins += 1
                print(f"Poprzednia najlepsza sieć wygrywa grę {i+1}")
            else:
                draws += 1
                print(f"Remis w grze {i+1}")
        
        # Wyświetl podsumowanie
        win_rate = current_wins / num_games
        print(f"\nWyniki ewaluacji po {num_games} grach:")
        print(f"Bieżąca sieć: {current_wins} wygranych ({win_rate:.1%})")
        print(f"Poprzednia najlepsza sieć: {best_wins} wygranych ({best_wins/num_games:.1%})")
        print(f"Remisy: {draws} ({draws/num_games:.1%})")
        
        return win_rate
        
    def save_as_pickle(self, filename, data, directory="./evaluator_data/"):
        """
        Zapisz dane do pliku pickle
        
        Args:
            filename (str): Nazwa pliku
            data: Dane do zapisania
            directory (str): Katalog docelowy
        """
        os.makedirs(directory, exist_ok=True)
        completeName = os.path.join(directory, filename)
        with open(completeName, 'wb') as output:
            pickle.dump(data, output)