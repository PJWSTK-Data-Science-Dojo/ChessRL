import os
import ray
import torch
from core.train import train
from torch.utils.tensorboard import SummaryWriter
from config.chess import game_config, DefaultSetupConfig


def main():
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

    # Make results dir for the `_test` function
    os.makedirs("./results", exist_ok=True)
    
    # Parametry szkolenia
    config.training_steps = 100000
    config.batch_size = 128
    config.num_simulations = 50
    config.td_steps = 5
    config.num_actors = 2
    config.amp_type = None
    config.log_interval = 1
    config.object_store_memory = 2 * 1024 * 1024 * 1024

    # Some of the fields are hidden in setup config, not config.
    setup_config = DefaultSetupConfig()
    config.set_config(setup_config)

    # Inicjalizacja Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_gpus=1, include_dashboard=True, object_store_memory=config.object_store_memory)
    
    # Inicjalizacja TensorBoard
    summary_writer = SummaryWriter(os.path.join(exp_path, 'summary'))
    
    # Rozpocznij szkolenie
    trained_model, _ = train(config, summary_writer)
    
    # Zapisz ostateczny model
    torch.save(trained_model.state_dict(), os.path.join(model_path, 'final_model.p'))
    print(f"Training completed. Model saved to {os.path.join(model_path, 'final_model.p')}")


if __name__ == '__main__':
    main()
