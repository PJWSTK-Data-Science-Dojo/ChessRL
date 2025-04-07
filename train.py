"""
Skrypt szkoleniowy dla modelu szachowego AlphaZero, dostosowany do aktualnej struktury projektu.
"""

import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
import time
import datetime
import pickle
import chess

from ChessRL.config import Config
from ChessRL.network import ChessNet
from ChessRL.trainer import Trainer
from ChessRL.self_play import SelfPlay

def setup_directories():
    """Utwórz wymagane katalogi, jeśli nie istnieją"""
    dirs = ['./model_data', './datasets/iter0', './datasets/iter1', './datasets/iter2']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_datasets(paths):
    """
    Wczytaj dane treningowe z wielu katalogów
    
    Args:
        paths: Lista ścieżek do katalogów z danymi
        
    Returns:
        Połączone dane treningowe
    """
    datasets = []
    
    for path in paths:
        if not os.path.exists(path):
            print(f"Ścieżka {path} nie istnieje, pomijanie.")
            continue
            
        print(f"Wczytywanie danych z {path}...")
        for file in os.listdir(path):
            if not file.endswith('.pkl'):
                continue
                
            filepath = os.path.join(path, file)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                datasets.extend(data)
                print(f"  Wczytano {len(data)} próbek z {file}")
            except Exception as e:
                print(f"  Błąd wczytywania {file}: {e}")
    
    print(f"Wczytano łącznie {len(datasets)} próbek.")
    return datasets

def run_iteration(iteration, args, config):
    """
    Uruchom jedną iterację potoku treningowego
    
    Args:
        iteration: Numer iteracji
        args: Argumenty z linii poleceń
        config: Obiekt konfiguracyjny
        
    Returns:
        Ścieżka do wytrenowanego modelu
    """
    print(f"\n{'='*50}")
    print(f"Rozpoczynanie iteracji {iteration}")
    print(f"{'='*50}\n")
    
    prev_model_path = args.load if iteration == 1 else f"./model_data/best_model_iter_{iteration-1}.pt"
    current_model_path = f"./model_data/model_iter_{iteration}.pt"
    best_model_path = f"./model_data/best_model_iter_{iteration}.pt"
    data_path = f"./datasets/iter{iteration-1}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używanie urządzenia: {device}")
    
    try:
        network = ChessNet()
        print("ChessNet zainicjalizowano bez parametrów")
    except TypeError as e:
        print(f"Błąd inicjalizacji ChessNet bez parametrów: {e}")
        
        try:
            in_channels = 22  # Standardowa wartość dla reprezentacji szachownicy (12 figur, 10 flag)
            network = ChessNet(in_channels=in_channels, n_features=config.n_features, n_residual_blocks=config.n_residual_blocks)
            print(f"ChessNet zainicjalizowano z parametrami: in_channels={in_channels}, n_features={config.n_features}, n_residual_blocks={config.n_residual_blocks}")
        except Exception as e2:
            # Jeśli to też nie zadziała, probujemy ostatniego podejścia
            print(f"Błąd inicjalizacji ChessNet z parametrami: {e2}")
            try:
                # Przekazujemy tylko liczbę kanałów wejściowych
                network = ChessNet(22)
                print("ChessNet zainicjalizowano z 22 kanałami wejściowymi")
            except Exception as e3:
                print(f"Błąd inicjalizacji ChessNet z 22 kanałami: {e3}")
                raise ValueError("Nie udało się zainicjalizować sieci ChessNet. Sprawdź implementację konstruktora.")
    
    network.to(device)
    
    if os.path.exists(prev_model_path):
        print(f"Wczytywanie modelu z {prev_model_path}")
        try:
            checkpoint = torch.load(prev_model_path, map_location=device)
            if 'state_dict' in checkpoint:
                network.load_state_dict(checkpoint['state_dict'])
            else:
                network.load_state_dict(checkpoint)
            print("Model wczytany pomyślnie")
        except Exception as e:
            print(f"Błąd wczytywania modelu: {e}")
            print("Używanie losowo zainicjowanego modelu.")
    else:
        print(f"Plik modelu {prev_model_path} nie istnieje. Używanie losowo zainicjowanego modelu.")
    
    network.eval()
    
    print("\nGenerowanie danych przez samodzielną grę...")
    os.makedirs(data_path, exist_ok=True)
    
    start_time = time.time()
    
    self_play = SelfPlay(network, config)
    
    if args.workers <= 1:
        game_examples = []
        print(f"Generowanie {args.games_per_iteration} gier w trybie jednoprocesowym...")
        for i in range(args.games_per_iteration):
            print(f"Gra {i+1}/{args.games_per_iteration}")
            game_start_time = time.time()
            game_data = self_play.execute_episode()
            game_examples.extend(game_data)
            game_time = time.time() - game_start_time
            print(f"Gra {i+1} ukończona w {game_time:.2f}s")
    else:
        print(f"Generowanie {args.games_per_iteration} gier z użyciem {args.workers} procesów...")
        game_examples = self_play.execute_parallel_self_play(args.games_per_iteration)
    
    # Zapisz wygenerowane dane
    if game_examples:
        filename = f"dataset_iter_{iteration}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        filepath = os.path.join(data_path, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(game_examples, f)
        print(f"Zapisano {len(game_examples)} przykładów do {filepath}")
    
    gen_time = time.time() - start_time
    print(f"Generowanie danych zakończone w {gen_time:.2f}s")
    
    print("\nRozpoczęcie treningu sieci...")
    
    # Przygotuj listę ścieżek do danych treningowych
    training_paths = []
    for i in range(max(0, iteration - args.training_window), iteration):
        training_paths.append(f"./datasets/iter{i}")
    
    train_data = load_datasets(training_paths)
    
    if not train_data:
        print("Brak danych treningowych. Pomijanie treningu.")
        # Kopiuj poprzedni model jako aktualny
        if os.path.exists(prev_model_path):
            import shutil
            shutil.copy(prev_model_path, current_model_path)
        return current_model_path
    
    trainer = Trainer(network, config)
    
    # Wczytaj poprzedni bufor powtórek, jeśli istnieje i jest włączona opcja --resume
    if args.resume and hasattr(config, 'replay_buffer_file') and os.path.exists(config.replay_buffer_file):
        print(f"Wczytywanie bufora powtórek z {config.replay_buffer_file}")
        trainer.load_replay_buffer(config.replay_buffer_file)
    
    # Trenuj sieć
    start_time = time.time()
    print(f"Trening na {len(train_data)} przykładach, {args.epochs} epok")
    
    # Dodaj dane do bufora powtórek
    trainer.replay_buffer.add(train_data)
    
    # Pętla treningowa
    for epoch in range(args.epochs):
        epoch_start = time.time()
        trainer.train(train_data)
        epoch_time = time.time() - epoch_start
        print(f"Epoka {epoch+1}/{args.epochs} zakończona w {epoch_time:.2f}s")
    
    train_time = time.time() - start_time
    print(f"Trening zakończony w {train_time:.2f}s")
    
    # Zapisz wytrenowany model
    trainer.save_model(f"model_iter_{iteration}.pt")
    
    # Zapisz bufor powtórek
    if hasattr(trainer, 'save_replay_buffer') and hasattr(config, 'replay_buffer_file'):
        trainer.save_replay_buffer(args.replay_buffer_file)
    
    # Również zapisz jako najlepszy model tej iteracji
    trainer.save_model(f"best_model_iter_{iteration}.pt")
    
    return best_model_path


def main():
    """Główny punkt wejścia"""
    parser = argparse.ArgumentParser(description='Potok treningowy AlphaZero dla szachów')
    
    # Argumenty ogólne
    parser.add_argument('--iterations', type=int, default=10, help='Liczba iteracji')
    parser.add_argument('--workers', type=int, default=4, help='Liczba procesów roboczych')
    parser.add_argument('--load', type=str, default=None, help='Ścieżka do wczytania początkowego modelu')
    
    # Argumenty samodzielnej gry
    parser.add_argument('--games-per-iteration', type=int, default=50, help='Liczba gier na iterację')
    
    # Argumenty treningu
    parser.add_argument('--epochs', type=int, default=10, help='Liczba epok na iterację')
    parser.add_argument('--batch-size', type=int, default=None, help='Rozmiar wsadu (nadpisuje config)')
    parser.add_argument('--training-window', type=int, default=3, 
                       help='Liczba ostatnich iteracji danych używanych do treningu')
    parser.add_argument('--resume', action='store_true', help='Wznów trening z zapisanym buforem powtórek')
    parser.add_argument('--replay-buffer-file', type=str, default="replay_buffer.pkl", 
                      help='Nazwa pliku bufora powtórek (domyślnie: replay_buffer.pkl)')
    
    args = parser.parse_args()
    
    setup_directories()
    
    config = Config()
    
    if not hasattr(config, 'replay_buffer_file'):
        config.replay_buffer_file = os.path.join("./model_data/", args.replay_buffer_file)
    
    if args.batch_size:
        config.batch_size = args.batch_size
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Metoda 'spawn' już ustawiona")
    
    for i in range(1, args.iterations + 1):
        try:
            best_model_path = run_iteration(i, args, config)
            print(f"Iteracja {i} zakończona. Najlepszy model: {best_model_path}")
        except Exception as e:
            print(f"Błąd w iteracji {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print("Potok treningowy zakończony.")


if __name__ == "__main__":
    main()