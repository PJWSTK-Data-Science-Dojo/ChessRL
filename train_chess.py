import os
import ray
import torch
from core.train import train
from torch.utils.tensorboard import SummaryWriter
from config.chess import game_config

def main():
    # Inicjalizacja Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_gpus=1, include_dashboard=True)

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
    config.num_actors = 1
    config.cpu_actor = 1
    config.gpu_actor = 1
    config.device = "cpu"
    config.amp_type = None
    config.use_augmentation = False
    config.log_interval = 1
    config.p_mcts_num = 5
    config.env = "chess"
    config.case = "atari"
    config.opr = "train"
    config.result_dir = os.path.join(os.getcwd(), 'results')
    config.no_cuda = True
    config.debug = True
    config.render = False
    config.save_video = False
    config.force = False
    config.cpu_actor = 4
    config.gpu_actor = 4
    config.p_mcts_num = 4
    config.seed = 0
    config.num_gpus = 1
    config.num_cpus = 4
    config.revisit_policy_search_rate = 0.99
    config.use_root_value = False
    config.use_priority = False
    config.use_max_priority = False
    config.test_episodes = 10
    config.use_augmentation = True
    config.augmentation = ['shift', 'intensity']
    config.info = 'none'
    config.load_model = False
    config.model_path = './results/test_model.p'
    config.object_store_memory = 10 * 1024 * 1024 * 1024  # 10 GB
    
    # Inicjalizacja TensorBoard
    summary_writer = SummaryWriter(os.path.join(exp_path, 'summary'))
    
    # Rozpocznij szkolenie
    trained_model, _ = train(config, summary_writer)
    
    # Zapisz ostateczny model
    torch.save(trained_model.state_dict(), os.path.join(model_path, 'final_model.p'))
    print(f"Training completed. Model saved to {os.path.join(model_path, 'final_model.p')}")


if __name__ == '__main__':
    main()
