"""
Zoptymalizowana implementacja sieci neuronowej AlphaZero dla szachów
z wydajniejszym treningiem i lepszym wykorzystaniem GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt

class ChessBoardDataset(Dataset):
    """
    Dataset do przechowywania i przetwarzania danych treningowych
    """
    def __init__(self, dataset, device='cpu', transform_to_tensor=True):
        """
        Inicjalizacja datasetu
        
        Args:
            dataset: Tablica danych (stan, polityka, wartość)
            device: Urządzenie do przechowywania tensorów
            transform_to_tensor: Czy konwertować dane do tensorów
        """
        self.data = dataset
        self.device = device
        self.transform_to_tensor = transform_to_tensor
        
        # Wstępnie przekształć wszystkie dane, jeśli flaga jest ustawiona
        if self.transform_to_tensor:
            self.X = []
            self.y_p = []
            self.y_v = []
            
            for item in self.data:
                # Konwertuj dane na tensory i przenieś na odpowiednie urządzenie
                x = torch.FloatTensor(item[0].transpose(2, 0, 1)).to(self.device)
                y_p = torch.FloatTensor(item[1]).to(self.device)
                y_v = torch.FloatTensor([item[2]]).to(self.device)
                
                self.X.append(x)
                self.y_p.append(y_p)
                self.y_v.append(y_v)
    
    def __len__(self):
        """Zwróć rozmiar datasetu"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Pobierz element z datasetu
        
        Args:
            idx: Indeks elementu
            
        Returns:
            Tuple (stan, polityka, wartość)
        """
        if self.transform_to_tensor:
            return self.X[idx], self.y_p[idx], self.y_v[idx]
        else:
            # Przekształć dane na bieżąco
            x = torch.FloatTensor(self.data[idx][0].transpose(2, 0, 1))
            y_p = torch.FloatTensor(self.data[idx][1])
            y_v = torch.FloatTensor([self.data[idx][2]])
            
            # Przenieś na odpowiednie urządzenie
            if self.device != 'cpu':
                x = x.to(self.device)
                y_p = y_p.to(self.device)
                y_v = y_v.to(self.device)
                
            return x, y_p, y_v


class ConvBlock(nn.Module):
    """
    Blok konwolucyjny z normalizacją wsadową
    """
    def __init__(self, in_channels, out_channels=256):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResBlock(nn.Module):
    """
    Blok rezydualny dla architektury AlphaZero
    """
    def __init__(self, inplanes=256, planes=256):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out, inplace=True)
        return out


class OutBlock(nn.Module):
    """
    Blok wyjściowy z głową polityki i wartości
    """
    def __init__(self, in_channels=256):
        super(OutBlock, self).__init__()
        # Głowa wartości
        self.conv_v = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.bn_v = nn.BatchNorm2d(1)
        self.fc1_v = nn.Linear(8*8, 64)
        self.fc2_v = nn.Linear(64, 1)
        
        # Głowa polityki
        self.conv_p = nn.Conv2d(in_channels, 128, kernel_size=1, bias=False)
        self.bn_p = nn.BatchNorm2d(128)
        self.fc_p = nn.Linear(8*8*128, 4672)
    
    def forward(self, x):
        # Głowa wartości
        v = F.relu(self.bn_v(self.conv_v(x)), inplace=True)
        v = v.view(-1, 8*8)
        v = F.relu(self.fc1_v(v), inplace=True)
        v = torch.tanh(self.fc2_v(v))
        
        # Głowa polityki
        p = F.relu(self.bn_p(self.conv_p(x)), inplace=True)
        p = p.view(-1, 8*8*128)
        p = self.fc_p(p)
        p = F.log_softmax(p, dim=1).exp()
        
        return p, v


class ChessNet(nn.Module):
    """
    Sieć neuronowa AlphaZero dla szachów
    """
    def __init__(self, in_channels=22, num_res_blocks=19):
        """
        Inicjalizacja sieci
        
        Args:
            in_channels: Liczba kanałów wejściowych
            num_res_blocks: Liczba bloków rezydualnych
        """
        super(ChessNet, self).__init__()
        self.conv_block = ConvBlock(in_channels)
        self.res_blocks = nn.ModuleList([ResBlock() for _ in range(num_res_blocks)])
        self.out_block = OutBlock()
        
        # Inicjalizacja wag
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicjalizacja wag z użyciem inicjalizacji Kaiming"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Przekazanie danych przez sieć
        
        Args:
            x: Wejściowy tensor [batch_size, kanały, wysokość, szerokość]
            
        Returns:
            Tuple (polityka, wartość)
        """
        x = self.conv_block(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.out_block(x)


class AlphaLoss(nn.Module):
    """
    Funkcja straty dla treningu AlphaZero
    """
    def __init__(self):
        super(AlphaLoss, self).__init__()
    
    def forward(self, y_value, value, y_policy, policy):
        """
        Oblicz stratę
        
        Args:
            y_value: Przewidywana wartość
            value: Rzeczywista wartość
            y_policy: Przewidywana polityka
            policy: Rzeczywista polityka
            
        Returns:
            Całkowita wartość straty
        """
        # Strata wartości (błąd średniokwadratowy)
        value_error = (value - y_value) ** 2
        
        # Strata polityki (entropia krzyżowa)
        policy_error = torch.sum((-policy * (1e-8 + y_policy.float()).log()), 1)
        
        # Całkowita strata
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error


def train_network(net, dataset, epochs=20, batch_size=32, lr=0.001, device='cuda', save_dir='./model_data'):
    """
    Trenuj sieć na podanym zbiorze danych
    
    Args:
        net: Sieć do trenowania
        dataset: Dane treningowe [(stan, polityka, wartość),...]
        epochs: Liczba epok
        batch_size: Rozmiar wsadu
        lr: Współczynnik uczenia
        device: Urządzenie do obliczeń (cpu/cuda)
        save_dir: Katalog do zapisania modelu
    """
    # Przygotuj dataset
    train_dataset = ChessBoardDataset(dataset, device=device)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if device == 'cpu' else 0,
        pin_memory=device != 'cpu'
    )
    
    # Ustaw sieć w tryb treningowy
    net.train()
    
    # Inicjalizuj funkcję straty i optymalizator
    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True
    )
    
    # Ścieżka do zapisu modelu
    os.makedirs(save_dir, exist_ok=True)
    
    # Statystyki treningowe
    losses = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        batch_count = 0
        
        for batch_idx, (states, policies, values) in enumerate(train_loader):
            # Zerowanie gradientów
            optimizer.zero_grad()
            
            # Forward pass
            policy_pred, value_pred = net(states)
            
            # Oblicz stratę
            loss = criterion(value_pred, values, policy_pred, policies)
            
            # Backward pass
            loss.backward()
            
            # Przycinanie gradientów (zapobiega wybuchającym gradientom)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            # Aktualizacja wag
            optimizer.step()
            
            # Aktualizacja statystyk
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_count += 1
            
            # Wyświetl postęp co 10 wsadów
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f'Epoka [{epoch+1}/{epochs}], Wsad [{batch_idx+1}/{len(train_loader)}], '
                      f'Strata: {batch_loss:.4f}, Czas: {elapsed:.2f}s')
        
        # Średnia strata epoki
        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)
        
        # Aktualizacja planera
        scheduler.step(avg_loss)
        
        # Zapisz model, jeśli jest najlepszy
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {'epoch': epoch + 1, 'state_dict': net.state_dict(), 'loss': best_loss},
                os.path.join(save_dir, f'best_model_epoch_{epoch+1}.pth.tar')
            )
        
        # Wyświetl podsumowanie epoki
        epoch_time = time.time() - start_time
        print(f'Epoka [{epoch+1}/{epochs}] zakończona, Średnia strata: {avg_loss:.4f}, Czas: {epoch_time:.2f}s')
    
    # Zapisz ostatni model
    torch.save(
        {'epoch': epochs, 'state_dict': net.state_dict(), 'loss': avg_loss},
        os.path.join(save_dir, 'final_model.pth.tar')
    )
    
    # Wykres straty
    plt.figure(figsize=(10, 5))
    plt.plot(losses, 'b-')
    plt.title('Strata treningowa')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'training_loss_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
    
    return net


def load_and_prepare_data(data_paths):
    """
    Wczytaj dane z wielu plików
    
    Args:
        data_paths: Lista ścieżek do katalogów z danymi
    
    Returns:
        Połączone dane treningowe
    """
    import pickle
    
    all_data = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"Ścieżka {path} nie istnieje, pomijanie.")
            continue
            
        print(f"Wczytywanie danych z {path}...")
        for filename in os.listdir(path):
            if not filename.endswith('.pkl'):
                continue
                
            filepath = os.path.join(path, filename)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                all_data.extend(data)
                print(f"  Wczytano {len(data)} próbek z {filename}")
            except Exception as e:
                print(f"  Błąd wczytywania {filename}: {e}")
    
    print(f"Wczytano łącznie {len(all_data)} próbek.")
    return all_data


def run_training_pipeline(input_paths, output_dir, epochs=20, batch_size=32):
    """
    Uruchom pełny proces treningu
    
    Args:
        input_paths: Lista ścieżek do katalogów z danymi
        output_dir: Katalog wyjściowy dla modelu
        epochs: Liczba epok treningu
        batch_size: Rozmiar wsadu
    """
    # Wczytaj dane
    dataset = load_and_prepare_data(input_paths)
    
    if not dataset:
        print("Brak danych do treningu.")
        return
    
    # Inicjalizuj sieć
    net = ChessNet()
    
    # Sprawdź dostępność CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Używanie urządzenia: {device}")
    
    if device.type == 'cuda':
        # Ustaw optymalizacje CUDA
        torch.backends.cudnn.benchmark = True
        net = net.to(device)
    
    # Trenuj sieć
    train_network(
        net,
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        save_dir=output_dir
    )
    
    print("Trening zakończony.")

if __name__ == "__main__":
    # Przykładowe użycie
    run_training_pipeline(
        input_paths=['./datasets/iter0/', './datasets/iter1/', './datasets/iter2/'],
        output_dir='./model_data',
        epochs=30,
        batch_size=64
    )