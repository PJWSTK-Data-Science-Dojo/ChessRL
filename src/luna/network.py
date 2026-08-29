"""Neural Network Wrapper -- EfficientZeroV2 learner with unroll training."""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from torch.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

from .config import EzV2LearnerConfig, MCTSParams
from .ezv2_networks import (
    EZV2Networks,
    _scale_latent,
    action_index_to_planes,
    action_int_to_planes,
    scalar_to_support,
)
from .game.chess_game import ChessGame
from .mcts import BatchedMCTS
from .replay_buffer import PrioritizedReplayBuffer
from .targets import build_unroll_targets, collate_batch
from .utils import AverageMeter


class RecurrentBatchResult(NamedTuple):
    """Batched recurrent forward: either full policy rows or top-K sparse policies per row."""

    policy_full: np.ndarray | None
    topk_indices: np.ndarray | None
    topk_probs: np.ndarray | None
    values: np.ndarray
    rewards: np.ndarray
    next_latent: torch.Tensor


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
    """Select compute device: CUDA GPU, Apple Silicon MPS, or CPU.

    Args:
        device_type: One of "cuda", "mps", or "cpu"
        cuda_device_index: Specific CUDA device index (only used if device_type="cuda")

    Returns:
        torch.device configured for the requested backend

    Raises:
        RuntimeError: If requested device is unavailable or incompatible
    """
    device_type = device_type.lower()

    # CPU fallback - always available
    if device_type == "cpu":
        logger.info("Using CPU backend (slow, recommended only for testing/inference)")
        return torch.device("cpu")

    # Apple Metal Performance Shaders (M1/M2/M3 Macs)
    if device_type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS backend requested but not available. "
                "MPS requires macOS 12.3+ and PyTorch 1.12+ on Apple Silicon. "
                "Fall back to CPU with --learner.device cpu"
            )
        if not torch.backends.mps.is_built():
            raise RuntimeError(
                "MPS backend not built into this PyTorch installation. "
                "Install a PyTorch build with MPS support or use --learner.device cpu"
            )
        logger.info("Using MPS backend (Apple Silicon GPU)")
        return torch.device("mps")

    # CUDA GPU (NVIDIA)
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA backend requested but not available. "
                "Ensure NVIDIA driver is installed and PyTorch was built with CUDA support. "
                "For CPU-only testing, use --learner.device cpu"
            )

        def _is_cuda_device_compatible(idx: int) -> bool:
            """Test if CUDA device can run basic operations."""
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

        # Try requested device index, or scan all devices
        indices_to_try = [cuda_device_index] if cuda_device_index is not None else list(range(device_count))
        for idx in indices_to_try:
            if idx is None or idx < 0 or idx >= device_count:
                continue
            if _is_cuda_device_compatible(idx):
                device_name = torch.cuda.get_device_name(idx)
                logger.info(f"Using CUDA device {idx}: {device_name}")
                return torch.device(f"cuda:{idx}")

        # No compatible device found
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        devices_str = ", ".join(f"{i}:{name}" for i, name in enumerate(device_names))
        if cuda_device_index is not None:
            raise RuntimeError(
                f"CUDA device {cuda_device_index} unavailable or incompatible. "
                f"Available devices: {devices_str}. "
                "Try a different --learner.cuda-device index or use CPU/MPS."
            )
        raise RuntimeError(
            f"No compatible CUDA devices found. Available: {devices_str}. "
            "Install a PyTorch build matching your GPU architecture or use --learner.device cpu"
        )

    # Unknown device type
    raise ValueError(
        f"Unknown device type '{device_type}'. Valid options: 'cuda' (NVIDIA GPU), 'mps' (Apple Silicon), 'cpu'"
    )


class LunaNetwork:
    """EfficientZeroV2 learner with persistent optimizer, mixed-precision, and unroll training."""

    _learner: EzV2LearnerConfig

    def __init__(self, game: ChessGame, learner: EzV2LearnerConfig | None = None) -> None:
        self._learner = learner or EzV2LearnerConfig()
        self._game = game
        self.device = _get_device(self._learner.device, self._learner.cuda_device)
        self.board_x, self.board_y, self.board_z = game.get_board_size()
        self.action_size = game.get_action_size()

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.nnet = EZV2Networks(game, self._learner).to(self.device)

        requested_amp_dtype = self._learner.amp_dtype.lower()
        if requested_amp_dtype not in {"bfloat16", "float16"}:
            raise ValueError("amp_dtype must be 'bfloat16' or 'float16'")
        if not 0.0 <= self._learner.recurrent_gradient_scale <= 1.0:
            raise ValueError("recurrent_gradient_scale must be between 0 and 1")
        self._amp_dtype = torch.bfloat16 if requested_amp_dtype == "bfloat16" else torch.float16
        if self.device.type == "cuda" and self._amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            logger.warning("CUDA bfloat16 is unavailable; falling back to float16 autocast.")
            self._amp_dtype = torch.float16

        self.optimizer = optim.AdamW(
            self.nnet.parameters(),
            lr=self._learner.lr,
            weight_decay=self._learner.weight_decay,
            fused=self.device.type == "cuda",
        )

        # GradScaler only supports CUDA, disable for MPS/CPU
        scaler_backend = "cuda" if self.device.type == "cuda" else "cpu"
        scaler_enabled = (
            self._learner.mixed_precision and self.device.type == "cuda" and self._amp_dtype == torch.float16
        )
        self.scaler = GradScaler(scaler_backend, enabled=scaler_enabled)

        self._global_step = 0
        self._trainer_iteration = 0
        self._mcts_inference_compiled = False
        self._training_compiled = False

        # Training must call eager forward (gradients, batch shapes); MCTS uses compiled wrappers below.
        self._train_initial_inference_with_latent = self.nnet.initial_inference_with_latent

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
                    self.nnet.initial_inference_with_latent = torch.compile(  # type: ignore[method-assign]
                        self._train_initial_inference_with_latent,
                        mode="reduce-overhead",
                    )
                    self.nnet.recurrent_inference = torch.compile(  # type: ignore[method-assign]
                        self.nnet.recurrent_inference,
                        mode="reduce-overhead",
                    )
                    self._mcts_inference_compiled = True

            if self._learner.compile_training and can_compile:
                logger.info("Compiling training forward paths with torch.compile (default)")
                self.nnet.representation = torch.compile(self.nnet.representation, mode="default")  # type: ignore[assignment]
                self.nnet.dynamics = torch.compile(self.nnet.dynamics, mode="default")  # type: ignore[assignment]
                self.nnet.prediction = torch.compile(self.nnet.prediction, mode="default")  # type: ignore[assignment]
                self._training_compiled = True

        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=max(1, self._learner.dataloader_workers),
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

    def _async_batch_prefetch(self, upcoming_steps: int = 0) -> bool:
        """Allow background sampling only while no upcoming batch can run reanalysis.

        Plain replay collation is safe to overlap with training. Reanalysis uses the
        live network, so the call that reaches its activation step and all later calls
        remain on the training thread.
        """
        L = self._learner
        if L.reanalyze_mcts_sims <= 0 or L.reanalyze_prob <= 0:
            return True
        return self._global_step + max(0, upcoming_steps) < L.mixed_value_td_until_step

    def _prepare_batch(
        self,
        replay: PrioritizedReplayBuffer,
        bs: int,
        unroll: int,
        td: int,
        discount: float,
        training_step: int,
        mcts_for_reanalyze: MCTSParams | None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, list[int]]:
        """Prepare training batch from replay samples with optional reanalysis.

        Converts numpy samples to torch tensors, optionally runs fresh MCTS to override
        stale targets, and builds K-step unroll targets for representation/dynamics/prediction
        training.

        Args:
            replay: Prioritized replay buffer to sample from.
            bs: Batch size.
            unroll: Number of recurrent unroll steps (K).
            td: Bootstrap horizon for value targets (n).
            discount: Discount factor for n-step returns.
            training_step: Current global training step for warmup logic.
            mcts_for_reanalyze: MCTS parameters for search-based value (None = disabled).

        Returns:
            Tuple of (collated_batch_dict, importance_weights, tree_indices) ready for GPU training.
        """
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
            and training_step >= L.mixed_value_td_until_step
        )
        if reanalysis_enabled:
            for sample_idx, (traj, pos_idx) in enumerate(batch):
                if np.random.random() >= L.reanalyze_prob:
                    continue
                root_overrides[sample_idx] = {}
                if L.reanalyze_policy:
                    policy_overrides[sample_idx] = {}
                for offset in range(unroll + 1):
                    position = pos_idx + offset
                    if position >= traj.game_length:
                        break
                    board, player = game.replay_board_player(traj.actions, position)
                    boards.append(game.get_canonical_form(board, player))
                    requests.append((sample_idx, position))

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
                assert root_override is not None
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
        """Validate training parameters before starting.

        Raises:
            ValueError: If any parameter is invalid or replay buffer is empty.
        """
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if bs <= 0:
            raise ValueError(f"batch_size must be positive, got {bs}")
        if unroll < 0:
            raise ValueError(f"unroll_steps cannot be negative, got {unroll}")
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
        """Run ``steps`` gradient updates from ``replay``.

        Supports gradient accumulation (``grad_accum_steps``) to simulate
        larger effective batch sizes and prefetches the next batch on a
        background thread to overlap data prep with GPU compute.

        When ``torch_profile_steps`` > 0, optional exports:

        * **Chrome trace** (``export_chrome_trace``) — ``chrome://tracing`` / Edge
          (only if TensorBoard export is off; otherwise the TB handler writes the trace).
        * **TensorBoard Kineto** (``tensorboard_trace_handler``) — run
          ``tensorboard --logdir <dir>`` and open the **PyTorch Profiler** / trace
          dashboards (memory, step breakdown, CPU vs CUDA overlap).

        ``with_stack=True`` adds Python stacks to ops (heavier, best for short runs).
        """
        # Validate inputs before training
        self._validate_training_inputs(
            replay,
            steps,
            self._learner.batch_size,
            self._learner.unroll_steps,
            self._learner.td_steps,
        )

        self.nnet.train()

        trace_path: str | None = None
        prof: Any = None
        want_tb = bool(torch_profile_tensorboard_dir)
        want_chrome = bool(torch_profile_export_chrome and torch_profile_dir)
        if torch_profile_steps > 0 and (want_chrome or want_tb):
            if want_chrome:
                assert torch_profile_dir is not None
                os.makedirs(torch_profile_dir, exist_ok=True)
                trace_path = os.path.join(
                    torch_profile_dir,
                    f"train_trace_iter{torch_profile_iter}.json",
                )
            tb_cb: Any | None = None
            if want_tb:
                assert torch_profile_tensorboard_dir is not None
                tb_dir = torch_profile_tensorboard_dir
                os.makedirs(tb_dir, exist_ok=True)
                tb_cb = tensorboard_trace_handler(tb_dir)

            def _on_trace_ready(p: Any) -> None:
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
            prof.__enter__()

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
        support = L.support_size
        lr_total = total_train_steps if total_train_steps > 0 else steps
        grad_accum = max(1, L.grad_accum_steps)
        train_discount = discount if discount is not None else L.discount
        async_pf = self._async_batch_prefetch(steps)

        self.optimizer.zero_grad(set_to_none=True)
        prefetch_future = None
        if async_pf:
            prefetch_future = self._prefetch_executor.submit(
                self._prepare_batch,
                replay,
                bs,
                unroll,
                td,
                train_discount,
                self._global_step + 1,
                mcts_for_reanalyze,
            )

        try:
            for step in range(1, steps + 1):
                self._global_step += 1
                new_lr = self._lr_schedule(self._global_step, lr_total)
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

                for accum_idx in range(grad_accum):
                    if async_pf:
                        assert prefetch_future is not None
                        collated, is_weights, tree_indices = prefetch_future.result()
                        if step < steps or accum_idx < grad_accum - 1:
                            next_ts = self._global_step + (1 if accum_idx == grad_accum - 1 else 0)
                            prefetch_future = self._prefetch_executor.submit(
                                self._prepare_batch,
                                replay,
                                bs,
                                unroll,
                                td,
                                train_discount,
                                next_ts,
                                mcts_for_reanalyze,
                            )
                    else:
                        collated, is_weights, tree_indices = self._prepare_batch(
                            replay,
                            bs,
                            unroll,
                            td,
                            train_discount,
                            self._global_step,
                            mcts_for_reanalyze,
                        )

                    obs = torch.as_tensor(collated["observations"], dtype=torch.float32, device=self.device)
                    valid = torch.as_tensor(collated["valid_masks"], dtype=torch.float32, device=self.device)
                    t_values = torch.as_tensor(collated["target_values"], dtype=torch.float32, device=self.device)
                    t_rewards = torch.as_tensor(collated["target_rewards"], dtype=torch.float32, device=self.device)
                    t_policies = torch.as_tensor(collated["target_policies"], dtype=torch.float32, device=self.device)
                    obs_unroll = torch.as_tensor(
                        collated["observations_unroll"], dtype=torch.float32, device=self.device
                    )
                    actions = torch.as_tensor(collated["actions"], dtype=torch.long, device=self.device)
                    is_w = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)
                    u_mask = torch.as_tensor(collated["unroll_mask"], dtype=torch.float32, device=self.device)
                    c_mask = torch.as_tensor(collated["consistency_mask"], dtype=torch.float32, device=self.device)
                    v_mask = torch.as_tensor(collated["value_mask"], dtype=torch.float32, device=self.device)
                    valid_unroll = torch.as_tensor(
                        collated["valid_masks_unroll"], dtype=torch.float32, device=self.device
                    )

                    with autocast(
                        "cuda",
                        enabled=L.mixed_precision and self.device.type == "cuda",
                        dtype=self._amp_dtype,
                    ):
                        latent, log_pi_0, value_pred_0 = self._train_initial_inference_with_latent(obs, valid)

                        loss_pi = -(t_policies[:, 0] * log_pi_0).sum(dim=1)

                        v_target_0 = scalar_to_support(t_values[:, 0], support)
                        loss_v_pred = _soft_ce_with_support(self._raw_value_logits(latent), v_target_0)

                        loss_r_total = torch.zeros(bs, device=self.device)
                        loss_pi_total = loss_pi * v_mask[:, 0]
                        loss_v_total = loss_v_pred * v_mask[:, 0]
                        loss_consist_total = torch.zeros(bs, device=self.device)

                        with torch.no_grad():
                            flat_obs = obs_unroll[:, 1:].reshape(-1, *obs_unroll.shape[2:])
                            flat_planes = self.nnet._obs_to_planes(flat_obs)
                            all_target_latents = _scale_latent(self.nnet.representation(flat_planes))
                            all_target_latents = all_target_latents.view(bs, unroll, *all_target_latents.shape[1:])

                        current_latent = latent
                        for k in range(unroll):
                            mask_k = u_mask[:, k]
                            valid_k = valid_unroll[:, k + 1]

                            act_planes = action_index_to_planes(actions[:, k], self.device)
                            dynamics_input = _scale_gradient(
                                current_latent,
                                L.recurrent_gradient_scale,
                            )
                            next_latent_raw, r_logits = self.nnet.dynamics(
                                dynamics_input,
                                act_planes,
                            )
                            next_latent = _scale_latent(next_latent_raw)
                            policy_logits_k, _ = self.nnet.prediction(next_latent, valid_k)
                            log_pi_k = F.log_softmax(policy_logits_k, dim=1)

                            r_target = scalar_to_support(t_rewards[:, k], support)
                            loss_r = _soft_ce_with_support(r_logits, r_target) * mask_k

                            loss_pi_k = -(t_policies[:, k + 1] * log_pi_k).sum(dim=1) * v_mask[:, k + 1]

                            v_target_k = scalar_to_support(t_values[:, k + 1], support)
                            loss_v_k = (
                                _soft_ce_with_support(self._raw_value_logits(next_latent), v_target_k)
                                * v_mask[:, k + 1]
                            )

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

                    self.scaler.scale(weighted).backward()

                    accum_weighted = accum_weighted + weighted.detach()
                    accum_pi_acc = accum_pi_acc + loss_pi_total.mean().detach().float()
                    accum_v_acc = accum_v_acc + loss_v_total.mean().detach().float()
                    accum_r_acc = accum_r_acc + loss_r_total.mean().detach().float()
                    accum_c_acc = accum_c_acc + loss_consist_total.mean().detach().float()
                    all_priority_errors.append((value_pred_0.float() - t_values[:, 0]).abs().detach().cpu().numpy())
                    all_tree_indices.append(tree_indices)

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
                # Capture gradient norm before clipping for monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.nnet.parameters(),
                    L.grad_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                for priority_error, tri in zip(all_priority_errors, all_tree_indices):
                    replay.update_priorities(tri, priority_error)

                scale_m = float(grad_accum)
                total_loss_m.update(float(accum_weighted.item()) * grad_accum, bs * grad_accum)
                pi_loss_m.update(float((accum_pi_acc / scale_m).item()), bs * grad_accum)
                v_loss_m.update(float((accum_v_acc / scale_m).item()), bs * grad_accum)
                r_loss_m.update(float((accum_r_acc / scale_m).item()), bs * grad_accum)
                consist_loss_m.update(float((accum_c_acc / scale_m).item()), bs * grad_accum)
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

                    # Log training metrics to WandB
                    if wandb is not None and wandb.run is not None:
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
                prof.__exit__(None, None, None)

        return {
            "total": total_loss_m.avg,
            "policy": pi_loss_m.avg,
            "value": v_loss_m.avg,
            "reward": r_loss_m.avg,
            "consistency": consist_loss_m.avg,
        }

    def _raw_value_logits(self, latent: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.nnet.prediction.value_head(latent))

    def _persist_compiled_mcts_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Copy latent off CUDAGraph output buffers when MCTS inference is torch.compile'd.

        ``mode='reduce-overhead'`` captures CUDA graphs; retaining outputs across subsequent
        compiled invocations triggers "tensor output ... overwritten" unless we snapshot.
        """
        if not self._mcts_inference_compiled:
            return latent
        return latent.clone()

    # ------------------------------------------------------------------
    # Batched inference for parallel self-play
    # ------------------------------------------------------------------

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
            autocast(
                "cuda",
                enabled=self._learner.mixed_precision and self.device.type == "cuda",
                dtype=self._amp_dtype,
            ),
        ):
            latent, log_pi, v = self.nnet.initial_inference_with_latent(board_t, valid_t)

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
        """Batched recurrent inference for parallel MCTS leaf expansion.

        Args:
            latents: Batch of hidden states (B, C, H, W)
            actions: List of action indices
            valid_masks: Optional list of legal move masks per position
            policy_topk: If set, return only top-K policy entries

        Returns:
            RecurrentBatchResult with policies, values, rewards, and next latents

        When ``policy_topk`` is set and smaller than the action dimension, only the top-K
        log-probability indices and renormalized probabilities are copied to the host
        (see :class:`RecurrentBatchResult`).
        """
        action_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        act_planes = action_index_to_planes(action_t, self.device)

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
            autocast(
                "cuda",
                enabled=self._learner.mixed_precision and self.device.type == "cuda",
                dtype=self._amp_dtype,
            ),
        ):
            next_latent, reward, log_pi, v = self.nnet.recurrent_inference(latents, act_planes, valid_mask_tensor)

        a_dim = int(log_pi.shape[1])
        k_limit = policy_topk if policy_topk is not None else a_dim
        if valid_masks:
            max_legal = max(
                (int(np.count_nonzero(mask)) for mask in valid_masks if mask is not None),
                default=0,
            )
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
            autocast(
                "cuda",
                enabled=self._learner.mixed_precision and self.device.type == "cuda",
                dtype=self._amp_dtype,
            ),
        ):
            latent, log_pi, v = self.nnet.initial_inference_with_latent(board_t, valid_t)

        latent = self._persist_compiled_mcts_latent(latent)
        return torch.exp(log_pi).float().cpu().numpy()[0], float(v.item()), latent

    def recurrent_predict(
        self,
        latent: torch.Tensor,
        action: int,
        valid_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float, torch.Tensor]:
        """Recurrent inference in latent space with optional legal move masking.

        Args:
            latent: Hidden state representation
            action: Action index to take from this state
            valid_mask: Optional legal move mask (1.0 for legal, 0.0 for illegal)

        Returns:
            Tuple of (policy_probs, value, reward, next_latent)
        """
        act_planes = action_int_to_planes(action, self.device)

        # Convert valid_mask to tensor if provided
        valid_mask_tensor = None
        if valid_mask is not None:
            valid_mask_tensor = torch.as_tensor(valid_mask, dtype=torch.float32, device=self.device).unsqueeze(
                0
            )  # Add batch dimension

        self.nnet.eval()
        with (
            torch.inference_mode(),
            autocast(
                "cuda",
                enabled=self._learner.mixed_precision and self.device.type == "cuda",
                dtype=self._amp_dtype,
            ),
        ):
            next_latent, reward, log_pi, v = self.nnet.recurrent_inference(latent, act_planes, valid_mask_tensor)
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
        payload: dict[str, object] = {
            "format_version": 2,
            "state_dict": self.nnet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self._global_step,
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
        return checkpoint

    def _restore_checkpoint(
        self,
        checkpoint: dict[str, Any],
        filepath: str | os.PathLike[str],
        *,
        load_optimizer: bool,
    ) -> None:
        """Validate and restore an already-read format-v2 checkpoint."""
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
        if not isinstance(raw_state_dict, dict):
            raise ValueError(f"Checkpoint has no valid state_dict: {filepath}")
        state_dict = self._normalize_compiled_state_dict(raw_state_dict)
        try:
            self.nnet.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint architecture does not match the configured network: {filepath}. "
                "Construct it with LunaNetwork.from_checkpoint() or use matching learner settings."
            ) from exc
        optimizer_restored = False
        if load_optimizer and "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                optimizer_restored = True
            except (ValueError, KeyError, RuntimeError):
                logger.warning("Could not restore optimizer state, starting fresh.")
        if optimizer_restored and "scaler" in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint["scaler"])
            except (ValueError, KeyError, RuntimeError):
                logger.warning("Could not restore gradient-scaler state, starting fresh.")
        if "global_step" in checkpoint:
            self._global_step = int(checkpoint["global_step"])
        self._trainer_iteration = int(checkpoint.get("trainer_iteration", 0))

    @staticmethod
    def _normalize_compiled_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remove ``torch.compile`` wrapper segments from persisted module keys."""
        return {key.replace("._orig_mod.", ".").removeprefix("_orig_mod."): value for key, value in state_dict.items()}

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

        allowed = {field.name for field in fields(EzV2LearnerConfig)}
        stored = checkpoint.get("learner_config")
        if not isinstance(stored, dict):
            raise ValueError(f"Checkpoint is missing learner_config metadata: {path}")
        config_values = {key: value for key, value in stored.items() if key in allowed}
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
    simsiam: torch.nn.Module,
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
    return cast(torch.Tensor, 1.0 - (p_pred * z_target).sum(dim=1))
