import torch
import os

from core.config import BaseConfig
from core.dataset import Transforms
from .env_wrapper import ChessWrapper
from .model import EfficientZeroNet
from typing import List
from dataclasses import dataclass, field


@dataclass
class DefaultSetupConfig:
    # Required arguments
    env: str = "EZEnvironment"
    opr: str = "train"  # choices=['train', 'test']
    amp_type: str = "none"  # choices=['torch_amp', 'none']
    case: str = "atari"  # choices=['atari']
    device: str = "cpu"
    result_dir: str = os.path.join(os.getcwd(), 'results')
    no_cuda: bool = True
    debug: bool = True
    render: bool = False
    save_video: bool = False
    force: bool = False
    cpu_actor: int = 4
    gpu_actor: int = 4
    p_mcts_num: int = 4
    seed: int = 0
    num_gpus: int = 4
    num_cpus: int = 4
    revisit_policy_search_rate: float = 0.99
    use_root_value: bool = False
    use_priority: bool = False
    use_max_priority: bool = False
    test_episodes: int = 10
    use_augmentation: bool = False
    augmentation: List[str] = field(default_factory=lambda: ["shift", "intensity"])
    info: str = "none"
    load_model: bool = False
    model_path: str = "./results/test_model.p"


class ChessConfig(BaseConfig):

    def __init__(self):
        super(ChessConfig, self).__init__(
            training_steps=100000,
            last_steps=20000,
            test_interval=10000,
            log_interval=1000,
            vis_interval=1000,
            test_episodes=32,
            checkpoint_interval=100,
            target_model_interval=200,
            save_ckpt_interval=10000,
            max_moves=512,  # Typowa maksymalna długość gry w szachy
            test_max_moves=512,
            history_length=400,
            discount=0.997,
            dirichlet_alpha=0.3,
            value_delta_max=0.01,
            num_simulations=50,
            batch_size=256,
            td_steps=5,
            num_actors=1,
            # Parametry zmienione dla szachów
            episode_life=False,  # W szachach nie ma koncepcji "życia"
            init_zero=True,
            clip_reward=False,  # W szachach nagrody są dyskretne
            cvt_string=False,
            image_based=False,  # Szachy nie są reprezentowane jako obrazy
            # Pozostałe parametry
            lr_warm_up=0.01,
            lr_init=0.2,
            lr_decay_rate=0.1,
            lr_decay_steps=100000,
            auto_td_steps_ratio=0.3,
            start_transitions=1,
            total_transitions=100 * 1000,
            transition_num=25,
            frame_skip=1,  # W szachach nie ma koncepcji "frame skip"
            stacked_observations=1,  # Ilość poprzednich stanów planszy
            reward_loss_coeff=1,
            value_loss_coeff=0.25,
            policy_loss_coeff=1,
            consistency_coeff=2,
            lstm_hidden_size=512,
            lstm_horizon_len=5,
            proj_hid=1024,
            proj_out=1024,
            pred_hid=512,
            pred_out=1024,
        )

        self.env_name = "ChessEnv"
        self.image_channel = 12  # 12 kanałów dla różnych typów figur
        self.obs_shape = (self.image_channel, 8, 8)  # Plansza szachowa 8x8
        self.discount **= self.frame_skip
        self.max_moves //= self.frame_skip
        self.test_max_moves //= self.frame_skip
        self.start_transitions = self.start_transitions * 1000 // self.frame_skip
        self.start_transitions = max(1, self.start_transitions)

        self.bn_mt = 0.1
        self.blocks = 1  # Liczba bloków w ResNet
        self.channels = 64  # Liczba kanałów w ResNet
        self.reduced_channels_reward = 16
        self.reduced_channels_value = 16
        self.reduced_channels_policy = 16
        self.resnet_fc_reward_layers = [32]
        self.resnet_fc_value_layers = [32]
        self.resnet_fc_policy_layers = [32]
        self.downsample = False  # W szachach nie potrzebujemy downsamplingu

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

    def set_game(self, save_video=False, save_path=None, video_callable=None):
        game = self.new_game()
        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        return EfficientZeroNet(
            self.obs_shape,
            self.action_space_size,
            self.blocks,
            self.channels,
            self.reduced_channels_reward,
            self.reduced_channels_value,
            self.reduced_channels_policy,
            self.resnet_fc_reward_layers,
            self.resnet_fc_value_layers,
            self.resnet_fc_policy_layers,
            self.reward_support.size,
            self.value_support.size,
            self.downsample,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            self.lstm_hidden_size,
            bn_mt=self.bn_mt,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            init_zero=self.init_zero,
            state_norm=self.state_norm
        )

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False, final_test=False):
        from .env_wrapper import ChessEnv
        env = ChessEnv()
        
        if seed is not None:
            env.seed(seed)
            
        return ChessWrapper(env, discount=self.discount, cvt_string=self.cvt_string)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def set_transforms(self):
        if self.use_augmentation:
            self.transforms = Transforms(self.augmentation, image_shape=(self.obs_shape[1], self.obs_shape[2]))

    def transform(self, images):
        return images


game_config = ChessConfig()
