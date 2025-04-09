import os
import ray
import torch
import numpy as np
from core.train import train
from torch.utils.tensorboard import SummaryWriter
from config.chess import game_config

def main():
    # Inicjalizacja Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_gpus=1)
    
    print("Initializing chess training with EfficientZero...")
    
    # Konfiguracja
    config = game_config
    config.set_game()
    config.seed = 0
    
    # Ścieżki
    exp_path = os.path.join('experiments', 'chess')
    os.makedirs(exp_path, exist_ok=True)
    model_path = os.path.join(exp_path, 'model')
    os.makedirs(model_path, exist_ok=True)
    
    # Ustawienie ścieżek
    config.exp_path = exp_path
    config.model_path = os.path.join(model_path, 'model.p')
    config.model_dir = model_path
    
    # Parametry szkolenia
    config.training_steps = 100000
    config.batch_size = 128
    config.num_simulations = 50
    config.td_steps = 5
    config.num_actors = 4
    
    # Inicjalizacja TensorBoard
    summary_writer = SummaryWriter(os.path.join(exp_path, 'summary'))
    
    # Rozpocznij szkolenie
    trained_model, _ = train(config, summary_writer)
    
    # Zapisz ostateczny model
    torch.save(trained_model.state_dict(), os.path.join(model_path, 'final_model.p'))
    print(f"Training completed. Model saved to {os.path.join(model_path, 'final_model.p')}")

if __name__ == '__main__':
    main()
