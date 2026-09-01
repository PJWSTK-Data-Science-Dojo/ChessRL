"""Construction and optional compilation of Luna network runtime state."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch
import torch.optim as optim
from loguru import logger

from luna.config import EzV2LearnerConfig, validate_learner_config
from luna.ezv2_networks import action_index_to_planes
from luna.game.chess_game import ChessGame
from luna.model_factory import build_model
from luna.network_runtime import configure_dynamic_cudagraphs, get_device
from luna.network_types import NetworkRuntime


def initialize_network(
    network: NetworkRuntime,
    game: ChessGame,
    learner: EzV2LearnerConfig | None,
) -> None:
    network._learner = learner or EzV2LearnerConfig()
    network._game = game
    validate_learner_config(network._learner)
    network.device = get_device(network._learner.device, network._learner.cuda_device)
    network.board_x, network.board_y, network.board_z = game.get_board_size()
    network.action_size = game.get_action_size()
    _configure_backend(network)
    network.nnet = build_model(game, network._learner).to(network.device)
    network._action_plane_lookup = _action_plane_lookup(network)
    network.optimizer = network._new_optimizer()
    network.scaler = network._new_grad_scaler()
    _initialize_progress(network)
    _initialize_forward_paths(network)
    _compile_forward_paths(network)
    network._prefetch_executor = _prefetch_executor(network._learner)


def new_optimizer(network: NetworkRuntime) -> optim.AdamW:
    return optim.AdamW(
        network.nnet.parameters(),
        lr=network._learner.lr,
        weight_decay=network._learner.weight_decay,
        fused=network.device.type == "cuda",
    )


def new_grad_scaler(network: NetworkRuntime) -> torch.GradScaler:
    backend = "cuda" if network.device.type == "cuda" else "cpu"
    enabled = network._learner.mixed_precision and network.device.type == "cuda" and network._amp_dtype == torch.float16
    return torch.GradScaler(backend, enabled=enabled)


def _configure_backend(network: NetworkRuntime) -> None:
    if network.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    requested = network._learner.amp_dtype.lower()
    network._amp_dtype = torch.bfloat16 if requested == "bfloat16" else torch.float16
    if network.device.type == "cuda" and network._amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        logger.warning("CUDA bfloat16 is unavailable; falling back to float16 autocast.")
        network._amp_dtype = torch.float16


def _action_plane_lookup(network: NetworkRuntime) -> torch.Tensor | None:
    if network.device.type != "cuda":
        return None
    actions = torch.arange(network.action_size, device=network.device)
    return action_index_to_planes(actions, network.device)


def _initialize_progress(network: NetworkRuntime) -> None:
    network._global_step = 0
    network._trainer_iteration = 0
    network._lr_schedule_total_steps = 0
    network._lr_schedule_mismatch_warned = False
    network._loaded_checkpoint_path = None
    network._training_phase_provenance = None
    network._low_diversity_reports = 0
    network._mcts_inference_compiled = False
    network._training_compiled = False


def _initialize_forward_paths(network: NetworkRuntime) -> None:
    network._mcts_initial_inference = network.nnet.initial_inference_with_latent
    network._mcts_recurrent_inference = network.nnet.recurrent_inference
    network._training_initial_inference = network.nnet.initial_inference_for_training
    network._training_representation = network.nnet.representation
    network._training_dynamics = network.nnet.dynamics
    network._training_prediction = network.nnet.prediction


def _compile_forward_paths(network: NetworkRuntime) -> None:
    if network.device.type != "cuda":
        return
    capability, _minor = torch.cuda.get_device_capability(network.device)
    compatible = capability >= 7
    if network._learner.compile_inference:
        _compile_inference(network, compatible)
    if network._learner.compile_training and compatible:
        _compile_training(network)


def _compile_inference(network: NetworkRuntime, compatible: bool) -> None:
    if not compatible:
        logger.warning("torch.compile disabled: device capability < 7.0 (Volta+). Run without --compile-inference.")
        return
    configure_dynamic_cudagraphs()
    logger.info("Compiling MCTS inference paths with torch.compile (reduce-overhead; dynamic CUDA Graphs skipped)")
    network._mcts_initial_inference = torch.compile(network._mcts_initial_inference, mode="reduce-overhead")
    network._mcts_recurrent_inference = torch.compile(network._mcts_recurrent_inference, mode="reduce-overhead")
    network._mcts_inference_compiled = True


def _compile_training(network: NetworkRuntime) -> None:
    logger.info("Compiling training forward paths with torch.compile (default)")
    network._training_initial_inference = torch.compile(network._training_initial_inference, mode="default")
    network._training_representation = torch.compile(network._training_representation, mode="default")
    network._training_dynamics = torch.compile(network._training_dynamics, mode="default")
    network._training_prediction = torch.compile(network._training_prediction, mode="default")
    network._training_compiled = True


def _prefetch_executor(learner: EzV2LearnerConfig) -> ThreadPoolExecutor | None:
    if learner.dataloader_workers <= 0:
        return None
    return ThreadPoolExecutor(
        max_workers=learner.dataloader_workers,
        thread_name_prefix="replay-fetch",
    )
