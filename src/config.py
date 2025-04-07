"""
Optymalne parametry konfiguracyjne dla modelu szachowego uczenia ze wzmocnieniem
"""

import os
import multiprocessing
import torch

class Config:
    """
    Klasa konfiguracyjna dla systemu szachowego uczenia ze wzmocnieniem 
    z optymalnymi ustawieniami wydajnościowymi
    """
    
    def __init__(self):
        """
        Inicjalizacja konfiguracji z domyślnymi wartościami dostosowanymi do dostępnego sprzętu.
        Wszystkie parametry są automatycznie skalowane na podstawie wykrytych możliwości GPU/CPU.
        """
        # ---------- ŚCIEŻKI SYSTEMOWE ----------
        # Katalog, w którym przechowywane są modele i bufor powtórek
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        # ---------- WYKRYWANIE MOŻLIWOŚCI SPRZĘTOWYCH ----------
        # Sprawdzenie dostępności CUDA do obliczeń na GPU
        self.has_cuda = torch.cuda.is_available()
        # Flaga określająca czy mamy wysokiej klasy GPU
        self.has_high_end_gpu = False
        
        if self.has_cuda:
            try:
                props = torch.cuda.get_device_properties(0)
                # Obliczanie dostępnej pamięci GPU w GB
                gpu_mem = getattr(props, 'total_memory', 0) / (1024 * 1024 * 1024)
                # Sprawdzenie czy karta to wysokiej klasy GPU (np. RTX 3080+ lub 4000)
                # - major >= 8 oznacza architekturę Ampere/Ada lub nowszą
                # - pamięć > 10GB to typowa wartość dla kart high-end
                if props.major >= 8 and gpu_mem > 10:
                    self.has_high_end_gpu = True
                    print(f"Wykryto wysokiej klasy GPU: {props.name} z {gpu_mem:.1f}GB VRAM")
            except Exception as e:
                print(f"Błąd podczas wykrywania możliwości GPU: {e}")
        
        # ---------- ARCHITEKTURA SIECI NEURONOWEJ ----------
        # Liczba bloków rezydualnych - kluczowy parametr wpływający na głębokość sieci
        # i jej zdolność do wykrywania złożonych wzorców w pozycjach szachowych
        if self.has_high_end_gpu:
            self.n_residual_blocks = 8  # Więcej bloków = większa złożoność modelu
        else:
            self.n_residual_blocks = 4  # Mniejsza sieć dla słabszego sprzętu
        
        # Liczba filtrów w warstwach konwolucyjnych - określa szerokość sieci
        # i jej zdolność do przetwarzania wielu cech jednocześnie
        if self.has_high_end_gpu:
            self.n_features = 192  # Więcej filtrów = lepsza reprezentacja pozycji
        else:
            self.n_features = 128  # Mniej filtrów dla słabszego sprzętu
        
        # ---------- PARAMETRY TRENINGU ----------
        # Rozmiar partii danych (batch size) - większe partie lepiej wykorzystują GPU, 
        # ale wymagają więcej pamięci
        if self.has_high_end_gpu:
            self.batch_size = 1024  # Duże partie dla kart high-end
        elif self.has_cuda:
            self.batch_size = 512   # Średnie partie dla standardowych GPU
        else:
            self.batch_size = 256   # Małe partie dla CPU
            
        # Liczba epok treningowych na iterację
        self.epochs = 10  # Ile razy przetwarzamy dane treningowe w każdej iteracji
        
        # Współczynnik uczenia - wpływa na wielkość aktualizacji wag podczas treningu
        self.learning_rate = 0.001  # 0.001 to typowa wartość początkowa dla optymalizatora Adam
        
        # Regularyzacja L2 - pomaga zapobiegać przeuczeniu modelu
        self.weight_decay = 1e-4  # Typowa wartość regularyzacji to 10^-4
        
        # ---------- PARAMETRY MCTS (MONTE CARLO TREE SEARCH) ----------
        # Liczba symulacji MCTS dla każdego ruchu - więcej = lepsza jakość gry,
        # ale wolniejsze wykonanie ruchu
        if self.has_high_end_gpu:
            self.num_simulations = 32  # Zwiększona liczba dla mocnych GPU
        elif self.has_cuda:
            self.num_simulations = 16  # Standardowa wartość dla większości GPU
        else:
            self.num_simulations = 8   # Niższa wartość dla CPU
            
        # Parametr eksploracji w formule UCB - balansuje eksplorację vs eksploatację
        self.c_puct = 1.0  # Współczynnik w formule Upper Confidence Bound
        
        # Parametry szumu Dirichleta dodawanego w korzeniu drzewa dla zwiększenia eksploracji
        self.dirichlet_alpha = 0.3         # Parametr kształtu rozkładu Dirichleta
        self.dirichlet_noise_factor = 0.25  # Waga szumu (0 = brak szumu, 1 = tylko szum)
        
        # ---------- PARAMETRY SELF-PLAY ----------
        # Liczba gier self-play na iterację treningu
        self.num_self_play_games = 10  # Liczba gier generowanych w każdej iteracji
        
        # Temperatura początkowa dla wyboru ruchów - wyższe wartości = większa losowość
        self.temperature = 1.0  # Temperatura = 1.0 oznacza wybór proporcjonalny do prawdopodobieństw
        
        # Próg temperatury - liczba ruchów, po których temperatura jest zmniejszana
        self.temperature_threshold = 15  # Po tylu ruchach sieć gra bardziej zachłannie
        
        # ---------- PARAMETRY RÓWNOLEGŁOŚCI ----------
        # Liczba procesów roboczych używanych do równoległego self-play
        if self.has_high_end_gpu:
            # Dla high-end GPU używamy mniej procesów, by skupić się na wykorzystaniu GPU
            self.num_workers = 3
        elif self.has_cuda:
            # Dla standardowych GPU balansujemy obciążenie CPU/GPU
            self.num_workers = min(2, multiprocessing.cpu_count() // 2)
        else:
            # Dla maszyn tylko z CPU
            self.num_workers = max(1, multiprocessing.cpu_count() // 2)
        
        # Rozmiar partii dla operacji MCTS na GPU
        if self.has_high_end_gpu:
            self.batch_mcts_size = 64  # Większe partie dla mocnych GPU
        elif self.has_cuda:
            self.batch_mcts_size = 32  # Standardowe partie dla GPU
        else:
            self.batch_mcts_size = 8   # Małe partie dla CPU
        
        # ---------- ZARZĄDZANIE PAMIĘCIĄ ----------
        # Rozmiar bufora powtórek - im większy, tym więcej różnorodnych przykładów treningowych
        self.replay_buffer_size = 300_000 if self.has_high_end_gpu else 200_000
        
        # ---------- OPTYMALIZACJE WYDAJNOŚCIOWE ----------
        # Przypięcie pamięci tensora do pamięci fizycznej - przyspiesza transfer CPU->GPU
        self.pin_memory = self.has_cuda
        
        # Liczba wątków do ładowania danych
        self.num_data_workers = min(2, multiprocessing.cpu_count() // 2)
        
        # Użycie treningu z mieszaną precyzją (FP16/FP32) dla szybszych obliczeń
        self.use_mixed_precision = self.has_cuda
        
        # ---------- PARAMETRY GUI ----------
        # Rozmiar planszy w pikselach
        self.board_size = 600
        # Motyw graficzny bierek
        self.piece_theme = 'default'
        
        # ---------- ŚCIEŻKI PLIKÓW ----------
        # Ścieżka do pliku z buforem powtórek
        self.replay_buffer_file = os.path.join(self.model_dir, "replay_buffer.pkl")