"""Neural Network Wrapper -- EfficientZeroV2 learner with unroll training."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from loguru import logger
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from luna.config import EzV2LearnerConfig, MCTSParams, validate_learner_config
from luna.ezv2_networks import (
    EZV2Networks,
    SimSiamProjector,
    _scale_latent,
    _support_to_scalar,
    action_index_to_planes,
    scalar_to_support,
)
from luna.game.chess_game import ChessGame
from luna.mcts import BatchedMCTS
from luna.replay_buffer import PrioritizedReplayBuffer
from luna.targets import build_unroll_targets, collate_batch
from luna.utils import AverageMeter


class RecurrentBatchResult(NamedTuple):
    """Batched recurrent forward: either full policy rows or top-K sparse policies per row."""

    policy_full: np.ndarray | None
    topk_indices: np.ndarray | None
    topk_probs: np.ndarray | None
    values: np.ndarray
    rewards: np.ndarray
    next_latent: torch.Tensor


_InitialInference = Callable[
    [torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
_RecurrentInference = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]
_RepresentationInference = Callable[[torch.Tensor], torch.Tensor]
_DynamicsInference = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
_PredictionInference = Callable[
    [torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor],
]
_PreparedBatch = tuple[dict[str, np.ndarray], np.ndarray, list[int]]
_RUNTIME_LEARNER_FIELDS = frozenset({"device", "cuda_device", "compile_inference", "compile_training"})
_MAX_CONSECUTIVE_AMP_SKIPS = 16
_GRAD_SCALER_FIELDS = frozenset({"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"})


def _clone_state_to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_to_cpu(item) for item in value)
    return deepcopy(value)


def _first_non_finite_path(value: object, path: str) -> str | None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            return path
        return None
    if isinstance(value, Mapping):
        for key, item in value.items():
            invalid = _first_non_finite_path(item, f"{path}.{key}")
            if invalid is not None:
                return invalid
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            invalid = _first_non_finite_path(item, f"{path}[{index}]")
            if invalid is not None:
                return invalid
    elif isinstance(value, float) and not math.isfinite(value):
        return path
    return None


def _validate_finite_state(value: object, label: str) -> None:
    invalid = _first_non_finite_path(value, label)
    if invalid is not None:
        raise ValueError(f"Checkpoint contains a non-finite value at {invalid}")


def _scaler_number(state: Mapping[str, object], name: str) -> float:
    value = state[name]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"Checkpoint scaler field {name} must be finite and numeric")
    return float(value)


def _scaler_float32(state: Mapping[str, object], name: str) -> float:
    value = _scaler_number(state, name)
    if abs(value) > torch.finfo(torch.float32).max:
        raise ValueError(f"Checkpoint scaler field {name} must be representable as float32")
    return float(np.float32(value))


def _scaler_integer(state: Mapping[str, object], name: str) -> int:
    value = state[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Checkpoint scaler field {name} must be an integer")
    return value


def _validate_grad_scaler_state(state: Mapping[str, object]) -> None:
    if not state:
        return
    missing = sorted(name for name in _GRAD_SCALER_FIELDS if name not in state)
    unexpected = sorted(str(name) for name in state if name not in _GRAD_SCALER_FIELDS)
    if missing or unexpected:
        raise ValueError(f"Checkpoint scaler fields are invalid (missing={missing}, unexpected={unexpected})")
    scale = _scaler_float32(state, "scale")
    if scale < torch.finfo(torch.float32).tiny:
        raise ValueError("Checkpoint scaler scale must be a positive normal float32")
    if _scaler_float32(state, "growth_factor") <= 1:
        raise ValueError("Checkpoint scaler growth_factor must be greater than 1")
    backoff_factor = _scaler_float32(state, "backoff_factor")
    if not 0 < backoff_factor < 1:
        raise ValueError("Checkpoint scaler backoff_factor must be between 0 and 1")
    if float(np.float32(scale * backoff_factor)) >= scale:
        raise ValueError("Checkpoint scaler backoff_factor must reduce the float32 scale")
    growth_interval = _scaler_integer(state, "growth_interval")
    if growth_interval <= 0:
        raise ValueError("Checkpoint scaler growth_interval must be positive")
    if growth_interval > torch.iinfo(torch.int32).max:
        raise ValueError("Checkpoint scaler growth_interval must fit int32")
    growth_tracker = _scaler_integer(state, "_growth_tracker")
    if growth_tracker < 0:
        raise ValueError("Checkpoint scaler _growth_tracker must be non-negative")
    if growth_tracker >= growth_interval:
        raise ValueError("Checkpoint scaler _growth_tracker must be less than growth_interval")


def _has_non_finite_gradients(parameters: Iterable[torch.nn.Parameter]) -> bool:
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            return True
    return False


def _pinned_h2d_float32(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Host numpy (C-contiguous) → GPU float32 with pinned staging when useful."""
    if device.type != "cuda" or not arr.flags.c_contiguous:
        return torch.as_tensor(arr, dtype=torch.float32, device=device)
    t = torch.from_numpy(arr)
    pin = torch.empty(arr.shape, dtype=torch.float32, pin_memory=True)
    pin.copy_(t)
    return pin.to(device, non_blocking=True)


def _scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Return ``tensor`` unchanged while multiplying its backward gradient by ``scale``.

    MuZero applies this at every recurrent dynamics edge so gradients from deeper
    unroll steps do not grow disproportionately with the unroll horizon.
    """
    return tensor * scale + tensor.detach() * (1.0 - scale)


def _get_device(device_type: str = "cuda", cuda_device_index: int | None = None) -> torch.device:
    """Resolve an available compute device or raise with setup guidance."""
    device_type = device_type.lower()

    if device_type == "cpu":
        logger.info("Using CPU backend")
        return torch.device("cpu")

    if device_type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS backend requested but unavailable. Verify the host and PyTorch build, or use --learner.device cpu."
            )
        if not torch.backends.mps.is_built():
            raise RuntimeError(
                "This PyTorch installation has no MPS support. Use a compatible build or --learner.device cpu."
            )
        logger.info("Using MPS backend")
        return torch.device("mps")

    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA backend requested but not available. "
                "Verify the driver and PyTorch build, or use --learner.device cpu."
            )

        def _is_cuda_device_compatible(idx: int) -> bool:
            try:
                with torch.cuda.device(idx):
                    probe = torch.zeros(1, device=f"cuda:{idx}")
                    _ = probe + 1
                return True
            except RuntimeError:
                return False

        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("CUDA available but no devices found.")

        indices_to_try = [cuda_device_index] if cuda_device_index is not None else list(range(device_count))
        for idx in indices_to_try:
            if idx is None or idx < 0 or idx >= device_count:
                continue
            if _is_cuda_device_compatible(idx):
                logger.info("Using CUDA device {}", idx)
                return torch.device(f"cuda:{idx}")

        available_indices = ", ".join(str(index) for index in range(device_count))
        if cuda_device_index is not None:
            raise RuntimeError(
                f"CUDA device {cuda_device_index} unavailable or incompatible. "
                f"Detected device indices: {available_indices}. "
                "Try another --learner.cuda-device index or use --learner.device cpu."
            )
        raise RuntimeError(
            f"No compatible CUDA device found among indices {available_indices}. "
            "Use a compatible PyTorch build or --learner.device cpu."
        )

    raise ValueError(f"Unknown device type '{device_type}'. Valid options are 'cuda', 'mps', and 'cpu'.")


class LunaNetwork:
    """EfficientZeroV2 learner with persistent optimizer, mixed-precision, and unroll training."""

    _learner: EzV2LearnerConfig

    def __init__(self, game: ChessGame, learner: EzV2LearnerConfig | None = None) -> None:
        self._learner = learner or EzV2LearnerConfig()
        self._game = game
        validate_learner_config(self._learner)
        self.device = _get_device(self._learner.device, self._learner.cuda_device)
        self.board_x, self.board_y, self.board_z = game.get_board_size()
        self.action_size = game.get_action_size()

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        requested_amp_dtype = self._learner.amp_dtype.lower()
        self._amp_dtype = torch.bfloat16 if requested_amp_dtype == "bfloat16" else torch.float16
        if self.device.type == "cuda" and self._amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            logger.warning("CUDA bfloat16 is unavailable; falling back to float16 autocast.")
            self._amp_dtype = torch.float16

        self.nnet = EZV2Networks(game, self._learner).to(self.device)
        self._action_plane_lookup = (
            action_index_to_planes(torch.arange(self.action_size, device=self.device), self.device)
            if self.device.type == "cuda"
            else None
        )
        self.optimizer = optim.AdamW(
            self.nnet.parameters(),
            lr=self._learner.lr,
            weight_decay=self._learner.weight_decay,
            fused=self.device.type == "cuda",
        )

        scaler_backend = "cuda" if self.device.type == "cuda" else "cpu"
        scaler_enabled = (
            self._learner.mixed_precision and self.device.type == "cuda" and self._amp_dtype == torch.float16
        )
        self.scaler = torch.GradScaler(scaler_backend, enabled=scaler_enabled)

        self._global_step = 0
        self._trainer_iteration = 0
        self._lr_schedule_total_steps = 0
        self._loaded_checkpoint_path: Path | None = None
        self._mcts_inference_compiled = False
        self._training_compiled = False

        self._mcts_initial_inference: _InitialInference = self.nnet.initial_inference_with_latent
        self._mcts_recurrent_inference: _RecurrentInference = self.nnet.recurrent_inference
        self._training_initial_inference: _InitialInference = self.nnet.initial_inference_for_training
        self._training_representation: _RepresentationInference = self.nnet.representation
        self._training_dynamics: _DynamicsInference = self.nnet.dynamics
        self._training_prediction: _PredictionInference = self.nnet.prediction

        if self.device.type == "cuda":
            cap_major, _ = torch.cuda.get_device_capability(self.device)
            can_compile = cap_major >= 7

            if self._learner.compile_inference:
                if not can_compile:
                    logger.warning(
                        "torch.compile disabled: device capability < 7.0 (Volta+). Run without --compile-inference.",
                    )
                else:
                    logger.info("Compiling MCTS inference paths with torch.compile (reduce-overhead)")
                    self._mcts_initial_inference = torch.compile(
                        self._mcts_initial_inference,
                        mode="reduce-overhead",
                    )
                    self._mcts_recurrent_inference = torch.compile(
                        self._mcts_recurrent_inference,
                        mode="reduce-overhead",
                    )
                    self._mcts_inference_compiled = True

            if self._learner.compile_training and can_compile:
                logger.info("Compiling training forward paths with torch.compile (default)")
                self._training_initial_inference = torch.compile(self._training_initial_inference, mode="default")
                self._training_representation = torch.compile(self._training_representation, mode="default")
                self._training_dynamics = torch.compile(self._training_dynamics, mode="default")
                self._training_prediction = torch.compile(self._training_prediction, mode="default")
                self._training_compiled = True

        self._prefetch_executor: ThreadPoolExecutor | None = None
        if self._learner.dataloader_workers > 0:
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=self._learner.dataloader_workers,
                thread_name_prefix="replay-fetch",
            )

    def warmup_mcts_inference(self, game: ChessGame) -> None:
        """Run one initial + one recurrent forward to pay torch.compile warmup cost before self-play."""
        if not self._mcts_inference_compiled:
            return
        board = game.get_init_board()
        canonical = game.get_canonical_form(board, 1)
        obs = game.to_array(canonical)
        valid = game.get_valid_moves(canonical, 1)
        pi, _v, latent = self.predict_with_latent(obs, valid)
        action = int(np.argmax(pi))
        _ = self.recurrent_predict(latent, action)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _lr_schedule(self, step_in_run: int, total_steps: int) -> float:
        """Linear warmup followed by cosine annealing over the full training run."""
        L = self._learner
        warmup_steps = min(max(L.lr_warmup_steps, 0), max(total_steps - 1, 0))
        if warmup_steps > 0 and step_in_run <= warmup_steps:
            return L.lr * step_in_run / warmup_steps
        progress = (step_in_run - warmup_steps) / max(total_steps - warmup_steps, 1)
        return L.lr_min + 0.5 * (L.lr - L.lr_min) * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    def _resolve_lr_schedule_total(self, requested_total: int, current_call_steps: int) -> int:
        candidate = requested_total if requested_total > 0 else self._global_step + current_call_steps
        if self._lr_schedule_total_steps == 0:
            self._lr_schedule_total_steps = candidate
        elif requested_total > 0 and requested_total != self._lr_schedule_total_steps:
            logger.warning(
                "Ignoring changed LR horizon {} and preserving checkpoint horizon {} steps.",
                requested_total,
                self._lr_schedule_total_steps,
            )
        return self._lr_schedule_total_steps

    def _async_batch_prefetch(self, upcoming_steps: int = 0) -> bool:
        """Allow background sampling only while no upcoming batch can run reanalysis.

        Plain replay collation is safe to overlap with training. Reanalysis uses the
        live network, so the call that reaches its activation step and all later calls
        remain on the training thread.
        """
        if self._prefetch_executor is None:
            return False
        L = self._learner
        if L.reanalyze_mcts_sims <= 0 or L.reanalyze_prob <= 0:
            return True
        return self._global_step + max(0, upcoming_steps) < L.reanalyze_start_step

    def _prepare_batch(
        self,
        replay: PrioritizedReplayBuffer,
        bs: int,
        unroll: int,
        td: int,
        discount: float,
        training_step: int,
        mcts_for_reanalyze: MCTSParams | None,
    ) -> _PreparedBatch:
        """Sample and collate replay, replacing selected stale targets with fresh search."""
        L = self._learner
        game = self._game
        batch, is_weights, tree_indices = replay.sample(bs, unroll)
        mcts_base = mcts_for_reanalyze or MCTSParams()

        root_overrides: list[dict[int, float] | None] = [None] * len(batch)
        policy_overrides: list[dict[int, np.ndarray] | None] = [None] * len(batch)
        requests: list[tuple[int, int]] = []
        boards = []
        reanalysis_enabled = (
            game is not None
            and L.reanalyze_mcts_sims > 0
            and L.reanalyze_prob > 0
            and training_step >= L.reanalyze_start_step
        )
        if reanalysis_enabled:
            for sample_idx, (traj, pos_idx) in enumerate(batch):
                if np.random.random() >= L.reanalyze_prob:
                    continue
                root_overrides[sample_idx] = {}
                if L.reanalyze_policy:
                    policy_overrides[sample_idx] = {}
                board, player = game.replay_board_player(traj.actions, pos_idx)
                for offset in range(unroll + 1):
                    position = pos_idx + offset
                    if position >= traj.game_length:
                        break
                    canonical = game.get_canonical_form(board, player)
                    boards.append(canonical.copy(stack=True))
                    requests.append((sample_idx, position))
                    if position + 1 < traj.game_length:
                        player = game.push_action(board, player, int(traj.actions[position]))

        if boards:
            mcts_r = replace(
                mcts_base,
                num_mcts_sims=L.reanalyze_mcts_sims,
                dir_noise=False,
            )
            was_training = self.nnet.training
            try:
                results = BatchedMCTS(game, self, mcts_r).search_batch(
                    boards,
                    temp=1.0,
                    add_exploration_noise=False,
                )
            finally:
                self.nnet.train(was_training)
            for (sample_idx, position), (pi, root_value, _obs, _valid) in zip(requests, results):
                root_override = root_overrides[sample_idx]
                if root_override is None:
                    raise RuntimeError("Reanalysis result has no matching value target")
                root_override[position] = root_value
                policy_override = policy_overrides[sample_idx]
                if policy_override is not None:
                    policy_override[position] = pi.astype(np.float32, copy=False)

        batch_targets: list[dict[str, Any]] = []
        for sample_idx, (traj, pos_idx) in enumerate(batch):
            batch_targets.append(
                build_unroll_targets(
                    traj,
                    pos_idx,
                    unroll,
                    td,
                    discount,
                    root_value_override=root_overrides[sample_idx],
                    policy_override=policy_overrides[sample_idx],
                )
            )

        collated = collate_batch(batch_targets)
        return collated, is_weights, tree_indices

    def _validate_training_inputs(
        self,
        replay: PrioritizedReplayBuffer,
        steps: int,
        bs: int,
        unroll: int,
        td: int,
    ) -> None:
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if bs <= 0:
            raise ValueError(f"batch_size must be positive, got {bs}")
        if unroll <= 0:
            raise ValueError(f"unroll_steps must be positive, got {unroll}")
        if td < 0:
            raise ValueError(f"td_steps cannot be negative, got {td}")
        if replay.size == 0:
            raise ValueError("Cannot train on empty replay buffer")

    def train_ezv2(
        self,
        replay: PrioritizedReplayBuffer,
        steps: int,
        total_train_steps: int = 0,
        *,
        discount: float | None = None,
        mcts_for_reanalyze: MCTSParams | None = None,
        torch_profile_steps: int = 0,
        torch_profile_dir: str | None = None,
        torch_profile_iter: int = 0,
        torch_profile_export_chrome: bool = True,
        torch_profile_tensorboard_dir: str | None = None,
        torch_profile_with_stack: bool = False,
    ) -> dict[str, float]:
        """Run optimizer steps from prioritized replay and return mean loss components."""
        self._validate_training_inputs(
            replay,
            steps,
            self._learner.batch_size,
            self._learner.unroll_steps,
            self._learner.td_steps,
        )

        self.nnet.train()

        trace_path: str | None = None
        prof: profile | None = None
        want_tb = bool(torch_profile_tensorboard_dir)
        want_chrome = bool(torch_profile_export_chrome and torch_profile_dir)
        if torch_profile_steps > 0 and (want_chrome or want_tb):
            if want_chrome:
                if torch_profile_dir is None:
                    raise ValueError("A profile directory is required for Chrome trace export")
                os.makedirs(torch_profile_dir, exist_ok=True)
                trace_path = os.path.join(
                    torch_profile_dir,
                    f"train_trace_iter{torch_profile_iter}.json",
                )
            tb_cb: Callable[[profile], None] | None = None
            if want_tb:
                if torch_profile_tensorboard_dir is None:
                    raise ValueError("A TensorBoard directory is required for profiler export")
                tb_dir = torch_profile_tensorboard_dir
                os.makedirs(tb_dir, exist_ok=True)
                tb_cb = tensorboard_trace_handler(tb_dir)

            def _on_trace_ready(p: profile) -> None:
                # tensorboard_trace_handler also calls export_chrome_trace; Kineto allows only one save per cycle.
                if tb_cb is not None:
                    tb_cb(p)
                    logger.info("TensorBoard / Kineto trace written under {}", torch_profile_tensorboard_dir)
                elif want_chrome and trace_path is not None:
                    p.export_chrome_trace(trace_path)
                    logger.info("PyTorch Chrome trace saved to {}", trace_path)

            activities = [ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)
            prof = profile(
                activities=activities,
                schedule=schedule(wait=0, warmup=0, active=torch_profile_steps, repeat=1),
                on_trace_ready=_on_trace_ready,
                record_shapes=True,
                profile_memory=True,
                with_stack=torch_profile_with_stack,
            )
            prof.start()

        total_loss_m = AverageMeter()
        pi_loss_m = AverageMeter()
        v_loss_m = AverageMeter()
        r_loss_m = AverageMeter()
        consist_loss_m = AverageMeter()
        step_time_m = AverageMeter()

        L = self._learner
        unroll = L.unroll_steps
        td = L.td_steps
        bs = L.batch_size
        micro_bs = bs // L.grad_accum_steps
        support = L.support_size
        lr_total = self._resolve_lr_schedule_total(total_train_steps, steps)
        grad_accum = L.grad_accum_steps
        train_discount = discount if discount is not None else L.discount
        async_pf = self._async_batch_prefetch(steps)

        self.optimizer.zero_grad(set_to_none=True)
        prefetch_future: Future[_PreparedBatch] | None = None
        prefetch_training_step: int | None = None
        if async_pf:
            prefetch_executor = self._prefetch_executor
            if prefetch_executor is None:
                raise RuntimeError("Asynchronous replay prefetch has no executor")
            prefetch_training_step = self._global_step + 1
            prefetch_future = prefetch_executor.submit(
                self._prepare_batch,
                replay,
                bs,
                unroll,
                td,
                train_discount,
                prefetch_training_step,
                mcts_for_reanalyze,
            )

        completed_steps = 0
        consecutive_amp_skips = 0
        retry_batch: _PreparedBatch | None = None
        try:
            while completed_steps < steps:
                step = completed_steps + 1
                training_step = self._global_step + 1
                new_lr = self._lr_schedule(training_step, lr_total)
                previous_lrs = [group["lr"] for group in self.optimizer.param_groups]
                for pg in self.optimizer.param_groups:
                    pg["lr"] = new_lr

                t0 = time.time()

                accum_weighted = torch.zeros(1, device=self.device)
                accum_pi_acc = torch.zeros((), device=self.device, dtype=torch.float32)
                accum_v_acc = torch.zeros((), device=self.device, dtype=torch.float32)
                accum_r_acc = torch.zeros((), device=self.device, dtype=torch.float32)
                accum_c_acc = torch.zeros((), device=self.device, dtype=torch.float32)
                all_priority_errors: list[np.ndarray] = []
                all_tree_indices: list[list[int]] = []

                if retry_batch is not None:
                    collated, is_weights, tree_indices = retry_batch
                elif async_pf and prefetch_future is not None:
                    if prefetch_training_step != training_step:
                        raise RuntimeError("Asynchronous replay prefetch is out of sequence")
                    collated, is_weights, tree_indices = prefetch_future.result()
                    prefetch_future = None
                    prefetch_training_step = None
                else:
                    collated, is_weights, tree_indices = self._prepare_batch(
                        replay,
                        bs,
                        unroll,
                        td,
                        train_discount,
                        training_step,
                        mcts_for_reanalyze,
                    )

                if async_pf and prefetch_future is None and step < steps:
                    prefetch_executor = self._prefetch_executor
                    if prefetch_executor is None:
                        raise RuntimeError("Asynchronous replay prefetch has no executor")
                    prefetch_training_step = training_step + 1
                    prefetch_future = prefetch_executor.submit(
                        self._prepare_batch,
                        replay,
                        bs,
                        unroll,
                        td,
                        train_discount,
                        prefetch_training_step,
                        mcts_for_reanalyze,
                    )

                for accum_idx in range(grad_accum):
                    start = accum_idx * micro_bs
                    stop = start + micro_bs
                    microbatch = {name: value[start:stop] for name, value in collated.items()}
                    micro_weights = is_weights[start:stop]
                    micro_tree_indices = tree_indices[start:stop]

                    obs = torch.as_tensor(microbatch["observations"], dtype=torch.float32, device=self.device)
                    valid = torch.as_tensor(microbatch["valid_masks"], dtype=torch.float32, device=self.device)
                    t_values = torch.as_tensor(microbatch["target_values"], dtype=torch.float32, device=self.device)
                    t_rewards = torch.as_tensor(microbatch["target_rewards"], dtype=torch.float32, device=self.device)
                    t_policies = torch.as_tensor(microbatch["target_policies"], dtype=torch.float32, device=self.device)
                    obs_unroll = torch.as_tensor(
                        microbatch["observations_unroll"], dtype=torch.float32, device=self.device
                    )
                    actions = torch.as_tensor(microbatch["actions"], dtype=torch.long, device=self.device)
                    is_w = torch.as_tensor(micro_weights, dtype=torch.float32, device=self.device)
                    u_mask = torch.as_tensor(microbatch["unroll_mask"], dtype=torch.float32, device=self.device)
                    c_mask = torch.as_tensor(microbatch["consistency_mask"], dtype=torch.float32, device=self.device)
                    v_mask = torch.as_tensor(microbatch["value_mask"], dtype=torch.float32, device=self.device)
                    valid_unroll = torch.as_tensor(
                        microbatch["valid_masks_unroll"], dtype=torch.float32, device=self.device
                    )

                    with torch.autocast(
                        "cuda",
                        enabled=L.mixed_precision and self.device.type == "cuda",
                        dtype=self._amp_dtype,
                    ):
                        latent, log_pi_0, value_logits_0 = self._training_initial_inference(obs, valid)
                        value_pred_0 = _support_to_scalar(value_logits_0, support)

                        loss_pi = -(t_policies[:, 0] * log_pi_0).sum(dim=1)

                        v_target_0 = scalar_to_support(t_values[:, 0], support)
                        loss_v_pred = _soft_ce_with_support(value_logits_0, v_target_0)

                        loss_r_total = torch.zeros(micro_bs, device=self.device)
                        loss_pi_total = loss_pi * v_mask[:, 0]
                        loss_v_total = loss_v_pred * v_mask[:, 0]
                        loss_consist_total = torch.zeros(micro_bs, device=self.device)

                        with torch.no_grad():
                            flat_obs = obs_unroll[:, 1:].reshape(-1, *obs_unroll.shape[2:])
                            flat_planes = self.nnet._obs_to_planes(flat_obs)
                            all_target_latents = _scale_latent(self._training_representation(flat_planes))
                            all_target_latents = all_target_latents.view(
                                micro_bs, unroll, *all_target_latents.shape[1:]
                            )

                        current_latent = latent
                        for k in range(unroll):
                            mask_k = u_mask[:, k]
                            valid_k = valid_unroll[:, k + 1]

                            act_planes = self._encode_action_planes(actions[:, k])
                            dynamics_input = _scale_gradient(
                                current_latent,
                                L.recurrent_gradient_scale,
                            )
                            next_latent_raw, r_logits = self._training_dynamics(
                                dynamics_input,
                                act_planes,
                            )
                            next_latent = _scale_latent(next_latent_raw)
                            policy_logits_k, value_logits_k = self._training_prediction(next_latent, valid_k)
                            log_pi_k = F.log_softmax(policy_logits_k, dim=1)

                            r_target = scalar_to_support(t_rewards[:, k], support)
                            loss_r = _soft_ce_with_support(r_logits, r_target) * mask_k

                            loss_pi_k = -(t_policies[:, k + 1] * log_pi_k).sum(dim=1) * v_mask[:, k + 1]

                            v_target_k = scalar_to_support(t_values[:, k + 1], support)
                            loss_v_k = _soft_ce_with_support(value_logits_k, v_target_k) * v_mask[:, k + 1]

                            target_latent = all_target_latents[:, k]
                            loss_consist = _simsiam_loss(self.nnet.simsiam, next_latent, target_latent) * c_mask[:, k]

                            loss_r_total = loss_r_total + loss_r
                            loss_pi_total = loss_pi_total + loss_pi_k
                            loss_v_total = loss_v_total + loss_v_k
                            loss_consist_total = loss_consist_total + loss_consist

                            current_latent = next_latent

                        num_valid = v_mask.sum(dim=1).clamp(min=1.0)
                        total = (
                            L.policy_loss_weight * loss_pi_total
                            + L.value_loss_weight * loss_v_total
                            + L.reward_loss_weight * loss_r_total
                            + L.consistency_loss_weight * loss_consist_total
                        ) / num_valid

                        weighted = (total * is_w).mean() / grad_accum

                    torch.autograd.backward(self.scaler.scale(weighted))

                    accum_weighted = accum_weighted + weighted.detach()
                    accum_pi_acc = accum_pi_acc + loss_pi_total.mean().detach().float()
                    accum_v_acc = accum_v_acc + loss_v_total.mean().detach().float()
                    accum_r_acc = accum_r_acc + loss_r_total.mean().detach().float()
                    accum_c_acc = accum_c_acc + loss_consist_total.mean().detach().float()
                    all_priority_errors.append((value_pred_0.float() - t_values[:, 0]).abs().detach().cpu().numpy())
                    all_tree_indices.append(micro_tree_indices)

                # Check for NaN/Inf in accumulated losses (training divergence detection)
                if not torch.isfinite(accum_weighted).all():
                    logger.error(
                        "Non-finite loss detected at step {}/{}! "
                        "total={:.4f} pi={:.4f} v={:.4f} r={:.4f} consist={:.4f}",
                        step,
                        steps,
                        accum_weighted.item(),
                        accum_pi_acc.item(),
                        accum_v_acc.item(),
                        accum_r_acc.item(),
                        accum_c_acc.item(),
                    )
                    raise RuntimeError(
                        f"Training diverged at step {step}/{steps}: loss is NaN or Inf. "
                        "Try lowering learning rate, increasing gradient clipping, or checking data preprocessing."
                    )

                self.scaler.unscale_(self.optimizer)
                scaler_enabled = self.scaler.is_enabled()
                previous_scale = self.scaler.get_scale() if scaler_enabled else 1.0
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.nnet.parameters(),
                    L.grad_clip_norm,
                    error_if_nonfinite=not scaler_enabled,
                )
                norm_is_finite = bool(torch.isfinite(grad_norm))
                gradient_overflow = (
                    scaler_enabled and not norm_is_finite and _has_non_finite_gradients(self.nnet.parameters())
                )
                if scaler_enabled and not norm_is_finite and not gradient_overflow:
                    self.scaler.update(new_scale=previous_scale)
                    self.optimizer.zero_grad(set_to_none=True)
                    for group, previous_lr in zip(self.optimizer.param_groups, previous_lrs):
                        group["lr"] = previous_lr
                    raise RuntimeError(
                        "Gradient norm overflowed despite finite gradient elements; optimizer update was not applied."
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                current_scale = self.scaler.get_scale() if scaler_enabled else previous_scale

                if gradient_overflow:
                    consecutive_amp_skips += 1
                    retry_batch = (collated, is_weights, tree_indices)
                    if consecutive_amp_skips >= _MAX_CONSECUTIVE_AMP_SKIPS:
                        raise RuntimeError(
                            "Mixed-precision training stopped after "
                            f"{consecutive_amp_skips} consecutive non-finite gradient updates."
                        )
                    logger.warning(
                        "Retrying optimizer step {}/{} after mixed-precision overflow "
                        "(loss scale {:.1f} -> {:.1f}, consecutive skips {}).",
                        step,
                        steps,
                        previous_scale,
                        current_scale,
                        consecutive_amp_skips,
                    )
                    continue

                retry_batch = None
                consecutive_amp_skips = 0
                self._global_step = training_step
                completed_steps += 1

                for priority_error, tri in zip(all_priority_errors, all_tree_indices):
                    replay.update_priorities(tri, priority_error)

                scale_m = float(grad_accum)
                total_loss_m.update(float(accum_weighted.item()), bs)
                pi_loss_m.update(float((accum_pi_acc / scale_m).item()), bs)
                v_loss_m.update(float((accum_v_acc / scale_m).item()), bs)
                r_loss_m.update(float((accum_r_acc / scale_m).item()), bs)
                consist_loss_m.update(float((accum_c_acc / scale_m).item()), bs)
                step_time_m.update(time.time() - t0)

                if step % 50 == 0 or step == steps:
                    logger.info(
                        "(step {}/{}) {:.3f}s lr={:.1e} | loss={:.4f} pi={:.4f} v={:.4f} r={:.4f} c={:.4f}",
                        step,
                        steps,
                        step_time_m.avg,
                        new_lr,
                        total_loss_m.avg,
                        pi_loss_m.avg,
                        v_loss_m.avg,
                        r_loss_m.avg,
                        consist_loss_m.avg,
                    )

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "train/loss_total": total_loss_m.avg,
                                "train/loss_policy": pi_loss_m.avg,
                                "train/loss_value": v_loss_m.avg,
                                "train/loss_reward": r_loss_m.avg,
                                "train/loss_consistency": consist_loss_m.avg,
                                "train/lr": new_lr,
                                "train/grad_norm": float(grad_norm),
                                "train/step_time": step_time_m.avg,
                                "global_step": self._global_step,
                            }
                        )

                if prof is not None:
                    prof.step()
        finally:
            if prof is not None:
                prof.stop()

        return {
            "total": total_loss_m.avg,
            "policy": pi_loss_m.avg,
            "value": v_loss_m.avg,
            "reward": r_loss_m.avg,
            "consistency": consist_loss_m.avg,
        }

    def _persist_compiled_mcts_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Copy latent off CUDAGraph output buffers when MCTS inference is torch.compile'd.

        ``mode='reduce-overhead'`` captures CUDA graphs; retaining outputs across subsequent
        compiled invocations triggers "tensor output ... overwritten" unless we snapshot.
        """
        if not self._mcts_inference_compiled:
            return latent
        return latent.clone()

    def _encode_action_planes(self, actions: torch.Tensor) -> torch.Tensor:
        if self._action_plane_lookup is None:
            return action_index_to_planes(actions, self.device)
        return torch.index_select(self._action_plane_lookup, 0, actions)

    def batched_initial_inference(
        self,
        obs_batch: np.ndarray,
        valid_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """Batched initial inference: (B, 8, 8, C), (B, A) -> (policies, values, latents)."""
        board_t = _pinned_h2d_float32(obs_batch.astype(np.float32, copy=False), self.device)
        valid_t = _pinned_h2d_float32(valid_batch.astype(np.float32, copy=False), self.device)

        self.nnet.eval()
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda",
                enabled=self._learner.mixed_precision and self.device.type == "cuda",
                dtype=self._amp_dtype,
            ),
        ):
            latent, log_pi, v = self._mcts_initial_inference(board_t, valid_t)

        policies = torch.exp(log_pi).float().cpu().numpy()
        values = v.float().cpu().numpy()
        return policies, values, self._persist_compiled_mcts_latent(latent)

    def batched_recurrent_inference(
        self,
        latents: torch.Tensor,
        actions: list[int],
        *,
        valid_masks: list[np.ndarray | None] | None = None,
        policy_topk: int | None = None,
    ) -> RecurrentBatchResult:
        """Expand MCTS leaves, optionally copying only renormalized top-K policies to the host."""
        action_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        act_planes = self._encode_action_planes(action_t)

        # Convert valid_masks to tensor if provided
        valid_mask_tensor = None
        if valid_masks is not None and len(valid_masks) > 0:
            # Stack valid masks, using all-ones for None entries
            batch_size = len(valid_masks)
            action_size = self.action_size
            valid_mask_np = np.ones((batch_size, action_size), dtype=np.float32)
            for i, mask in enumerate(valid_masks):
                if mask is not None:
                    valid_mask_np[i] = mask
            valid_mask_tensor = torch.as_tensor(valid_mask_np, dtype=torch.float32, device=self.device)

        self.nnet.eval()
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda",
                enabled=self._learner.mixed_precision and self.device.type == "cuda",
                dtype=self._amp_dtype,
            ),
        ):
            next_latent, reward, log_pi, v = self._mcts_recurrent_inference(latents, act_planes, valid_mask_tensor)

        a_dim = int(log_pi.shape[1])
        k_limit = policy_topk if policy_topk is not None else a_dim
        if valid_masks and all(mask is not None for mask in valid_masks):
            k_limit = max(int(np.count_nonzero(mask)) for mask in valid_masks if mask is not None)
        elif valid_masks:
            max_legal = max((int(np.count_nonzero(mask)) for mask in valid_masks if mask is not None), default=0)
            k_limit = max(k_limit, max_legal)
        if k_limit <= 0:
            k_use = a_dim
        else:
            k_use = min(k_limit, a_dim)

        values = v.float().cpu().numpy()
        rewards = reward.float().cpu().numpy()
        next_latent = self._persist_compiled_mcts_latent(next_latent)

        if k_use >= a_dim:
            policies = torch.exp(log_pi).float().cpu().numpy()
            return RecurrentBatchResult(policies, None, None, values, rewards, next_latent)

        top_log, top_i = torch.topk(log_pi, k=k_use, dim=1)
        probs_t = torch.softmax(top_log.float(), dim=1)
        idx_np = top_i.cpu().numpy().astype(np.int32)
        pr_np = probs_t.cpu().numpy().astype(np.float32)
        return RecurrentBatchResult(None, idx_np, pr_np, values, rewards, next_latent)

    def predict_with_latent(self, board: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float, torch.Tensor]:
        board_t = torch.as_tensor(board, dtype=torch.float32, device=self.device)
        valid_t = torch.as_tensor(valid, dtype=torch.float32, device=self.device)
        board_t = board_t.view(1, self.board_x, self.board_y, self.board_z)
        if valid_t.dim() == 1:
            valid_t = valid_t.unsqueeze(0)

        self.nnet.eval()
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda",
                enabled=self._learner.mixed_precision and self.device.type == "cuda",
                dtype=self._amp_dtype,
            ),
        ):
            latent, log_pi, v = self._mcts_initial_inference(board_t, valid_t)

        latent = self._persist_compiled_mcts_latent(latent)
        return torch.exp(log_pi).float().cpu().numpy()[0], float(v.item()), latent

    def recurrent_predict(
        self,
        latent: torch.Tensor,
        action: int,
        valid_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float, torch.Tensor]:
        """Advance a latent state while masking illegal continuation moves when supplied."""
        action_t = torch.tensor([action], device=self.device)
        act_planes = self._encode_action_planes(action_t)

        # Convert valid_mask to tensor if provided
        valid_mask_tensor = None
        if valid_mask is not None:
            valid_mask_tensor = torch.as_tensor(valid_mask, dtype=torch.float32, device=self.device).unsqueeze(
                0
            )  # Add batch dimension

        self.nnet.eval()
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda",
                enabled=self._learner.mixed_precision and self.device.type == "cuda",
                dtype=self._amp_dtype,
            ),
        ):
            next_latent, reward, log_pi, v = self._mcts_recurrent_inference(latent, act_planes, valid_mask_tensor)
        next_latent = self._persist_compiled_mcts_latent(next_latent)
        return (
            torch.exp(log_pi).float().cpu().numpy()[0],
            float(v.item()),
            float(reward.item()),
            next_latent,
        )

    def save_checkpoint(
        self,
        folder: str = "checkpoint",
        filename: str = "checkpoint.pth.tar",
        *,
        extra_state: dict[str, object] | None = None,
    ) -> None:
        filepath = os.path.join(folder, filename)
        output_dir = os.path.dirname(filepath) or "."
        os.makedirs(output_dir, exist_ok=True)
        model_state = self.nnet.state_dict()
        optimizer_state = self.optimizer.state_dict()
        scaler_state = self.scaler.state_dict()
        _validate_grad_scaler_state(scaler_state)
        payload: dict[str, object] = {
            "format_version": 2,
            "state_dict": model_state,
            "optimizer": optimizer_state,
            "scaler": scaler_state,
            "global_step": self._global_step,
            "trainer_iteration": self._trainer_iteration,
            "lr_schedule_total_steps": self._lr_schedule_total_steps,
            "learner_config": asdict(self._learner),
            "model_spec": {
                "action_size": self.action_size,
                "observation_shape": [self.board_x, self.board_y, self.board_z],
            },
        }
        if extra_state:
            reserved = payload.keys() & extra_state.keys()
            if reserved:
                raise ValueError(f"extra_state cannot replace reserved checkpoint fields: {sorted(reserved)}")
            payload.update(extra_state)
        _validate_finite_state(payload, "checkpoint")
        temporary_path = f"{filepath}.tmp-{os.getpid()}"
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, filepath)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary_path)

    def load_checkpoint(
        self,
        folder: str = "checkpoint",
        filename: str = "checkpoint.pth.tar",
        *,
        load_optimizer: bool = True,
    ) -> None:
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        checkpoint = self._read_checkpoint(filepath)
        self._restore_checkpoint(checkpoint, filepath, load_optimizer=load_optimizer)

    @staticmethod
    def _read_checkpoint(filepath: str | os.PathLike[str]) -> dict[str, Any]:
        """Read one supported checkpoint onto CPU without executing pickled code."""
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint payload is not a mapping: {filepath}")
        if checkpoint.get("format_version") != 2:
            raise ValueError(f"Unsupported checkpoint format in {filepath}; only format version 2 is accepted.")
        required = {
            "state_dict",
            "optimizer",
            "scaler",
            "global_step",
            "trainer_iteration",
            "lr_schedule_total_steps",
            "learner_config",
            "model_spec",
        }
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError(f"Checkpoint is missing required fields {missing}: {filepath}")
        return checkpoint

    @classmethod
    def checkpoint_trainer_iteration(cls, filepath: str | os.PathLike[str]) -> int:
        """Validate a format-v2 checkpoint and return its completed trainer iteration."""
        checkpoint = cls._read_checkpoint(filepath)
        return cls._checkpoint_counter(checkpoint, "trainer_iteration", filepath)

    @staticmethod
    def _checkpoint_counter(checkpoint: Mapping[str, Any], name: str, filepath: str | os.PathLike[str]) -> int:
        value: object = checkpoint[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Checkpoint field '{name}' must be a non-negative integer: {filepath}")
        return value

    @staticmethod
    def _checkpoint_learner_config(checkpoint: Mapping[str, Any], filepath: str | os.PathLike[str]) -> dict[str, Any]:
        stored = checkpoint["learner_config"]
        if not isinstance(stored, dict) or not all(isinstance(key, str) for key in stored):
            raise ValueError(f"Checkpoint learner_config must be a string-keyed mapping: {filepath}")
        expected_fields = {field.name for field in fields(EzV2LearnerConfig)}
        stored_fields = set(stored)
        if stored_fields != expected_fields:
            missing = sorted(expected_fields - stored_fields)
            unexpected = sorted(stored_fields - expected_fields)
            raise ValueError(
                f"Checkpoint learner_config does not match format version 2: {filepath} "
                f"(missing={missing}, unexpected={unexpected})."
            )
        return cast(dict[str, Any], stored)

    def _validate_learner_config(self, checkpoint: Mapping[str, Any], filepath: str | os.PathLike[str]) -> None:
        stored = self._checkpoint_learner_config(checkpoint, filepath)
        current = asdict(self._learner)
        mismatched = sorted(
            name for name in stored if name not in _RUNTIME_LEARNER_FIELDS and stored[name] != current[name]
        )
        if mismatched:
            raise ValueError(f"Checkpoint learner configuration differs in fields {mismatched}: {filepath}")

    def _restore_training_state(self, checkpoint: Mapping[str, Any], filepath: str | os.PathLike[str]) -> None:
        optimizer_state = checkpoint["optimizer"]
        scaler_state = checkpoint["scaler"]
        try:
            self.optimizer.load_state_dict(optimizer_state)
            self.scaler.load_state_dict(scaler_state)
        except (KeyError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Checkpoint training state is incompatible: {filepath}") from exc

    def _restore_checkpoint(
        self,
        checkpoint: dict[str, Any],
        filepath: str | os.PathLike[str],
        *,
        load_optimizer: bool,
    ) -> None:
        """Validate and restore an already-read format-v2 checkpoint."""
        self._validate_learner_config(checkpoint, filepath)
        global_step = self._checkpoint_counter(checkpoint, "global_step", filepath)
        trainer_iteration = self._checkpoint_counter(checkpoint, "trainer_iteration", filepath)
        lr_schedule_total_steps = self._checkpoint_counter(checkpoint, "lr_schedule_total_steps", filepath)
        if not isinstance(checkpoint["optimizer"], dict) or not isinstance(checkpoint["scaler"], dict):
            raise ValueError(f"Checkpoint optimizer and scaler states must be mappings: {filepath}")
        model_spec = checkpoint.get("model_spec")
        expected_shape = [self.board_x, self.board_y, self.board_z]
        if not isinstance(model_spec, dict):
            raise ValueError(f"Checkpoint is missing model_spec metadata: {filepath}")
        if model_spec.get("action_size") != self.action_size or model_spec.get("observation_shape") != expected_shape:
            raise ValueError(
                f"Checkpoint model specification does not match this game: {filepath} "
                f"(expected action_size={self.action_size}, observation_shape={expected_shape})."
            )

        raw_state_dict = checkpoint.get("state_dict")
        if not isinstance(raw_state_dict, dict) or not all(
            isinstance(name, str) and isinstance(tensor, torch.Tensor) for name, tensor in raw_state_dict.items()
        ):
            raise ValueError(f"Checkpoint state_dict must map string names to tensors: {filepath}")
        state_dict = self._normalize_compiled_state_dict(cast(dict[str, torch.Tensor], raw_state_dict))
        _validate_finite_state(state_dict, "checkpoint.state_dict")
        _validate_finite_state(checkpoint["optimizer"], "checkpoint.optimizer")
        _validate_finite_state(checkpoint["scaler"], "checkpoint.scaler")
        _validate_grad_scaler_state(checkpoint["scaler"])
        previous_model = {name: tensor.detach().cpu().clone() for name, tensor in self.nnet.state_dict().items()}
        previous_optimizer = _clone_state_to_cpu(self.optimizer.state_dict()) if load_optimizer else None
        previous_scaler = deepcopy(self.scaler.state_dict()) if load_optimizer else None
        previous_global_step = self._global_step
        previous_trainer_iteration = self._trainer_iteration
        previous_lr_schedule_total_steps = self._lr_schedule_total_steps
        previous_loaded_checkpoint_path = self._loaded_checkpoint_path
        try:
            try:
                self.nnet.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Checkpoint architecture does not match the configured network: {filepath}. "
                    "Construct it with LunaNetwork.from_checkpoint() or use matching learner settings."
                ) from exc
            if load_optimizer:
                self._restore_training_state(checkpoint, filepath)
            self._global_step = global_step
            self._trainer_iteration = trainer_iteration
            self._lr_schedule_total_steps = lr_schedule_total_steps
            self._loaded_checkpoint_path = Path(filepath).expanduser().resolve()
        except (KeyError, RuntimeError, TypeError, ValueError) as restore_error:
            self.nnet.load_state_dict(previous_model, strict=True)
            if load_optimizer:
                if not isinstance(previous_optimizer, dict) or not isinstance(previous_scaler, dict):
                    raise RuntimeError("Checkpoint rollback state is invalid") from restore_error
                self.optimizer.load_state_dict(previous_optimizer)
                self.scaler.load_state_dict(previous_scaler)
            self._global_step = previous_global_step
            self._trainer_iteration = previous_trainer_iteration
            self._lr_schedule_total_steps = previous_lr_schedule_total_steps
            self._loaded_checkpoint_path = previous_loaded_checkpoint_path
            raise

    @staticmethod
    def _normalize_compiled_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remove ``torch.compile`` wrapper segments from persisted module keys."""
        normalized: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            normalized_key = key.replace("._orig_mod.", ".").removeprefix("_orig_mod.")
            if normalized_key in normalized:
                raise ValueError(f"Checkpoint state_dict contains duplicate normalized key {normalized_key!r}")
            normalized[normalized_key] = value
        return normalized

    @classmethod
    def from_checkpoint(
        cls,
        game: ChessGame,
        checkpoint_path: str | os.PathLike[str],
        *,
        device: str = "cuda",
        cuda_device: int | None = None,
        compile_inference: bool = False,
        load_optimizer: bool = False,
    ) -> LunaNetwork:
        """Build the matching architecture from versioned checkpoint metadata and load it."""
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"No model in path {path}")
        checkpoint = cls._read_checkpoint(path)
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint has no valid state_dict: {path}")

        config_values = dict(cls._checkpoint_learner_config(checkpoint, path))
        config_values.update(
            device=device,
            cuda_device=cuda_device,
            compile_inference=compile_inference,
            compile_training=False,
        )
        network = cls(game, EzV2LearnerConfig(**config_values))
        network._restore_checkpoint(checkpoint, path, load_optimizer=load_optimizer)
        return network

    def log_model_summary(self) -> None:
        total = sum(p.numel() for p in self.nnet.parameters())
        logger.info(
            "Model: {} parameters | observation={} | actions={} | channels={} | representation_blocks={} | dynamics_blocks={}",
            f"{total:,}",
            (self.board_x, self.board_y, self.board_z),
            self.action_size,
            self._learner.num_channels,
            self._learner.repr_blocks,
            self._learner.dyn_blocks,
        )


def _soft_ce_with_support(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with soft categorical support targets."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(target_probs * log_probs).sum(dim=1)


def _simsiam_loss(
    simsiam: SimSiamProjector,
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
) -> torch.Tensor:
    """SimSiam-style negative cosine similarity loss (per sample)."""
    z_pred = simsiam.project(predicted_latent)
    p_pred = simsiam.predict(z_pred)

    with torch.no_grad():
        z_target = simsiam.project(target_latent)

    p_pred = F.normalize(p_pred, dim=1)
    z_target = F.normalize(z_target, dim=1)
    return 1.0 - (p_pred * z_target).sum(dim=1)
