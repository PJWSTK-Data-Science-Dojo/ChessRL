"""
Zoptymalizowana implementacja Monte Carlo Tree Search (MCTS) z przetwarzaniem wsadowym na GPU
i efektywnym zarządzaniem pamięcią
"""

import numpy as np
import torch
import torch.nn.functional as F
import math
import copy
import time
import chess
from collections import deque

from ChessRL.encoding import encode_board, move_to_index

class MCTSNode:
    """
    Węzeł w drzewie MCTS
    """
    __slots__ = ('game', 'move', 'parent', 'is_expanded', 'children', 'child_priors', 
                 'child_total_value', 'child_number_visits', 'action_idxes')
    
    def __init__(self, game, move=None, parent=None):
        self.game = game  # stan s
        self.move = move  # indeks akcji
        self.is_expanded = False
        self.parent = parent  
        self.children = {}
        self.child_priors = np.zeros([4672], dtype=np.float32)
        self.child_total_value = np.zeros([4672], dtype=np.float32)
        self.child_number_visits = np.zeros([4672], dtype=np.float32)
        self.action_idxes = []
        
    @property
    def number_visits(self):
        if self.parent is None:
            return 0
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        if self.parent is not None:
            self.parent.child_number_visits[self.move] = value
    
    @property
    def total_value(self):
        if self.parent is None:
            return 0
        return self.parent.child_total_value[self.move]
    
    @total_value.setter
    def total_value(self, value):
        if self.parent is not None:
            self.parent.child_total_value[self.move] = value
    
    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)
    
    def child_U(self):
        return math.sqrt(self.number_visits + 1) * (
            abs(self.child_priors) / (1 + self.child_number_visits))
    
    def best_child(self):
        if self.action_idxes:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove
    
    def select_leaf(self):
        current = self
        while current.is_expanded and not current.game.check_status():
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current
    
    def add_dirichlet_noise(self, action_idxs, child_priors):
        valid_child_priors = child_priors[action_idxs]
        valid_child_priors = 0.75 * valid_child_priors + 0.25 * np.random.dirichlet(
            np.zeros([len(valid_child_priors)], dtype=np.float32) + 0.3)
        child_priors[action_idxs] = valid_child_priors
        return child_priors
    
    def expand(self, child_priors):
        """
        Rozszerz węzeł o możliwe ruchy
        
        Args:
            child_priors: Rozkład prawdopodobieństwa ruchów z sieci neuronowej
        """
        self.is_expanded = True
        action_idxs = []
        c_p = child_priors.copy()
        
        # Uzyskaj możliwe akcje
        for action in self.game.actions():
            if action:
                initial_pos, final_pos, underpromote = action
                action_idxs.append(move_to_index(self.game, initial_pos, final_pos, underpromote))
        
        if not action_idxs:
            self.is_expanded = False
            return
            
        self.action_idxes = action_idxs
        
        # Zamaskuj nielegalne ruchy
        for i in range(len(child_priors)):
            if i not in action_idxs:
                c_p[i] = 0.0
        
        # Dodaj szum Dirichleta w korzeniu drzewa
        if self.parent is None or self.parent.parent is None:
            c_p = self.add_dirichlet_noise(action_idxs, c_p)
            
        self.child_priors = c_p
    
    def decode_n_move_pieces(self, board, move):
        """
        Zdekoduj i wykonaj ruch na szachownicy
        
        Args:
            board: Stan szachownicy
            move: Zakodowany ruch do wykonania
            
        Returns:
            board: Nowy stan szachownicy po wykonaniu ruchu
        """
        from ChessRL.encoding import decode_action
        i_pos, f_pos, prom = decode_action(board, move)
        for i, f, p in zip(i_pos, f_pos, prom):
            board.player = self.game.player
            board.move_piece(i, f, p)
            a, b = i
            c, d = f
            # Jeśli król porusza się o 2 pola, przesuń również wieżę (roszada)
            if board.current_board[c, d] in ["K", "k"] and abs(d - b) == 2:
                if a == 7 and d - b > 0:  # roszada krótka białych
                    board.player = self.game.player
                    board.move_piece((7, 7), (7, 5), None)
                elif a == 7 and d - b < 0:  # roszada długa białych
                    board.player = self.game.player
                    board.move_piece((7, 0), (7, 3), None)
                elif a == 0 and d - b > 0:  # roszada krótka czarnych
                    board.player = self.game.player
                    board.move_piece((0, 7), (0, 5), None)
                elif a == 0 and d - b < 0:  # roszada długa czarnych
                    board.player = self.game.player
                    board.move_piece((0, 0), (0, 3), None)
        return board
    
    def maybe_add_child(self, move):
        """
        Dodaj dziecko do węzła, jeśli nie istnieje
        
        Args:
            move: Indeks ruchu do wykonania
            
        Returns:
            Węzeł dziecka
        """
        if move not in self.children:
            copy_board = copy.deepcopy(self.game)
            copy_board = self.decode_n_move_pieces(copy_board, move)
            self.children[move] = MCTSNode(
                copy_board, move=move, parent=self)
        return self.children[move]
    
    def backup(self, value_estimate):
        """
        Propaguj wartość w górę drzewa
        
        Args:
            value_estimate: Oszacowana wartość stanu
        """
        current = self
        while current.parent is not None:
            current.number_visits += 1
            # Odwróć znak wartości dla przeciwnego gracza
            if current.game.player == 1:
                current.total_value += value_estimate
            elif current.game.player == 0:
                current.total_value += -value_estimate
            current = current.parent


class DummyNode:
    """
    Korzeń drzewa MCTS z uproszczoną funkcjonalnością
    """
    def __init__(self):
        self.parent = None
        self.child_total_value = np.zeros([4672], dtype=np.float32)
        self.child_number_visits = np.zeros([4672], dtype=np.float32)


class MCTS:
    """
    Zoptymalizowany algorytm Monte Carlo Tree Search
    """
    def __init__(self, net, num_simulations=800, batch_size=8, c_puct=1.0):
        """
        Inicjalizacja MCTS
        
        Args:
            net: Sieć neuronowa do oceny pozycji
            num_simulations: Liczba symulacji do wykonania
            batch_size: Rozmiar wsadu do ewaluacji sieci
            c_puct: Parametr eksploracji w formule PUCT
        """
        self.net = net
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.device = next(net.parameters()).device
        
        # Statystyki wydajności
        self.eval_times = []
        self.search_times = []
    
    def batch_evaluate(self, leaf_nodes):
        """
        Ewaluuj wiele węzłów jednocześnie w jednym wsadzie
        
        Args:
            leaf_nodes: Lista węzłów do oceny
            
        Returns:
            List of (policy, value) pairs for each node
        """
        # Koduj stany szachownicy
        encoded_states = []
        for node in leaf_nodes:
            encoded_state = encode_board(node.game)
            encoded_states.append(encoded_state.transpose(2, 0, 1))
            
        # Konwertuj na tensor
        encoded_batch = torch.FloatTensor(np.array(encoded_states)).to(self.device)
        
        # Ewaluuj wsad przy użyciu sieci
        with torch.no_grad():
            policy_batch, value_batch = self.net(encoded_batch)
        
        # Przetwórz wyniki
        policies = F.softmax(policy_batch, dim=1).cpu().numpy()
        values = value_batch.cpu().numpy()
        
        return list(zip(policies, values))
    
    def search(self, game_state):
        """
        Wykonaj przeszukiwanie MCTS, aby znaleźć najlepszy ruch
        
        Args:
            game_state: Stan gry do przeszukania
            
        Returns:
            Najlepszy ruch i korzeń drzewa
        """
        root = MCTSNode(game_state, parent=DummyNode())
        
        # Sprawdź, czy gra nie jest już zakończona
        if game_state.check_status() and game_state.in_check_possible_moves() == []:
            # Gra zakończona - nie ma potrzeby przeszukiwać
            return np.random.choice(len(root.child_number_visits)), root
        
        start_time = time.time()
        
        # Wykonaj symulacje
        for _ in range(self.num_simulations):
            # Zbieraj węzły liści do ewaluacji wsadowej
            leaf_batch = []
            paths = []
            
            # Zbierz węzły liści aż do rozmiaru wsadu lub końca symulacji
            while len(leaf_batch) < self.batch_size:
                leaf = root.select_leaf()
                
                # Jeśli gra już się zakończyła w tym węźle
                if leaf.game.check_status() and leaf.game.in_check_possible_moves() == []:
                    if leaf.game.player == 0:  # czarne wygrywają
                        leaf.backup(-1)
                    elif leaf.game.player == 1:  # białe wygrywają
                        leaf.backup(1)
                    continue
                
                # Dodaj węzeł liścia do wsadu
                leaf_batch.append(leaf)
                paths.append(leaf)
                
                # Jeśli osiągnęliśmy rozmiar wsadu lub zakończyliśmy symulacje, przerwij
                if len(leaf_batch) >= self.batch_size or len(paths) >= self.num_simulations:
                    break
            
            # Jeśli nie zebraliśmy żadnych węzłów, przejdź do następnej iteracji
            if not leaf_batch:
                continue
            
            # Ewaluuj wsad węzłów liści
            eval_start = time.time()
            evaluations = self.batch_evaluate(leaf_batch)
            self.eval_times.append(time.time() - eval_start)
            
            # Rozszerz i propaguj dla każdego węzła w wsadzie
            for leaf, evaluation in zip(leaf_batch, evaluations):
                policy, value = evaluation
                value = value[0][0]  # Uzyskaj skalarną wartość
                
                # Rozszerz węzeł liścia
                leaf.expand(policy)
                
                # Propaguj wartość w górę drzewa
                leaf.backup(value)
        
        # Zapisz całkowity czas przeszukiwania
        self.search_times.append(time.time() - start_time)
        
        # Zwróć ruch z największą liczbą wizyt
        return np.argmax(root.child_number_visits), root


def get_policy(root):
    """
    Uzyskaj rozkład polityki z korzenia drzewa MCTS
    
    Args:
        root: Korzeń drzewa MCTS
        
    Returns:
        Znormalizowana polityka
    """
    policy = np.zeros([4672], dtype=np.float32)
    visit_sum = np.sum(root.child_number_visits)
    if visit_sum > 0:
        for idx in np.where(root.child_number_visits != 0)[0]:
            policy[idx] = root.child_number_visits[idx] / visit_sum
    return policy


def do_decode_n_move_pieces(board, move):
    """
    Zdekoduj i wykonaj ruch na szachownicy
    
    Args:
        board: Stan szachownicy
        move: Zakodowany ruch do wykonania
        
    Returns:
        Nowy stan szachownicy po wykonaniu ruchu
    """
    from ChessRL.encoding import decode_action
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


def MCTS_self_play(net, num_games, output_path="./datasets/iter2/", num_simulations=800):
    """
    Generuj gry przy użyciu MCTS i sieci neuronowej
    
    Args:
        net: Sieć neuronowa do oceny pozycji
        num_games: Liczba gier do wygenerowania
        output_path: Ścieżka do zapisania wygenerowanych danych
        num_simulations: Liczba symulacji MCTS na ruch
    """
    # Tworzenie obiektu szachownicy
    from ChessRL.chess_board import board as c_board  # Zakładam, że masz tego modułu
    
    # Inicjalizuj MCTS
    mcts = MCTS(net, num_simulations=num_simulations)
    
    for game_idx in range(num_games):
        current_board = c_board()
        checkmate = False
        dataset = []
        states = []
        value = 0
        game_start_time = time.time()
        
        print(f"Rozpoczynanie gry {game_idx+1}/{num_games}")
        
        # Kontynuuj grę do szach-mata lub 100 ruchów
        while not checkmate and current_board.move_count <= 100:
            # Sprawdź powtórzenia (remis)
            draw_counter = 0
            for s in states:
                if np.array_equal(current_board.current_board, s):
                    draw_counter += 1
            if draw_counter == 3:
                break
                
            # Zapisz obecny stan
            states.append(copy.deepcopy(current_board.current_board))
            board_state = copy.deepcopy(encode_board(current_board))
            
            # Wykonaj przeszukiwanie MCTS
            move_start_time = time.time()
            best_move, root = mcts.search(current_board)
            
            # Pobierz politykę i wykonaj najlepszy ruch
            policy = get_policy(root)
            dataset.append([board_state, policy])
            
            # Wykonaj ruch
            current_board = do_decode_n_move_pieces(current_board, best_move)
            
            move_time = time.time() - move_start_time
            print(f"Gra {game_idx+1}, ruch {current_board.move_count}, czas: {move_time:.2f}s")
            
            # Sprawdź czy gra się zakończyła
            if current_board.check_status() and current_board.in_check_possible_moves() == []:
                if current_board.player == 0:  # czarne wygrywają
                    value = -1
                elif current_board.player == 1:  # białe wygrywają
                    value = 1
                checkmate = True
        
        # Przypisz wartość wyniku każdemu stanowi
        dataset_with_value = []
        for idx, data in enumerate(dataset):
            s, p = data
            if idx == 0:
                dataset_with_value.append([s, p, 0])
            else:
                dataset_with_value.append([s, p, value])
        
        # Zapisz dane
        import pickle
        import os
        import datetime
        
        today = datetime.datetime.today().strftime("%Y-%m-%d")
        filename = f"dataset_game{game_idx}_{today}.pkl"
        
        # Upewnij się, że katalog istnieje
        os.makedirs(output_path, exist_ok=True)
        
        with open(os.path.join(output_path, filename), 'wb') as f:
            pickle.dump(dataset_with_value, f)
        
        game_time = time.time() - game_start_time
        print(f"Gra {game_idx+1} zakończona w {game_time:.2f}s, wynik: {value}")