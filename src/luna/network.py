"""Public EfficientZeroV2 network facade."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.optim as optim
import wandb as wandb
from loguru import logger

from luna.config import EzV2LearnerConfig, MCTSParams
from luna.ezv2_networks import EZV2Networks
from luna.game.chess_game import ChessGame
from luna.mcts import BatchedMCTS
from luna.network_batches import async_batch_prefetch, prepare_batch, validate_training_inputs
from luna.network_checkpoint_api import NetworkCheckpointMixin
from luna.network_checkpoint_state import (
    clone_state_to_cpu,
    first_non_finite_path,
    validate_finite_state,
    validate_grad_scaler_state,
)
from luna.network_inference import (
    batched_initial_inference,
    batched_recurrent_inference,
    encode_action_planes,
    persist_compiled_latent,
    predict_with_latent,
    recurrent_predict,
)
from luna.network_losses import (
    latent_health_metrics,
    piece_class_targets,
    piece_reconstruction_accuracy_metrics,
    piece_reconstruction_loss,
    raw_latent_health_metrics,
    simsiam_loss,
    soft_ce_with_support,
)
from luna.network_runtime import (
    configure_dynamic_cudagraphs,
    get_device,
    has_non_finite_gradients,
    pinned_h2d_float32,
    scale_gradient,
)
from luna.network_setup import initialize_network, new_grad_scaler, new_optimizer
from luna.network_training import TrainingRequest, train_ezv2
from luna.network_training_profiler import TrainingProfilerConfig
from luna.network_training_types import TrainingFunctions
from luna.network_types import (
    DynamicsInference,
    InitialInference,
    NetworkRuntime,
    PredictionInference,
    PreparedBatch,
    ReanalysisSearch,
    RecurrentBatchResult,
    RecurrentInference,
    RepresentationCollapseError,
    RepresentationInference,
    TrainingPhaseProvenance,
    ValidatedCheckpoint,
)
from luna.replay_buffer import PrioritizedReplayBuffer

_configure_dynamic_cudagraphs = configure_dynamic_cudagraphs
_clone_state_to_cpu = clone_state_to_cpu
_first_non_finite_path = first_non_finite_path
_validate_finite_state = validate_finite_state
_validate_grad_scaler_state = validate_grad_scaler_state
_has_non_finite_gradients = has_non_finite_gradients
_pinned_h2d_float32 = pinned_h2d_float32
_scale_gradient = scale_gradient
_get_device = get_device
_soft_ce_with_support = soft_ce_with_support
_piece_class_targets = piece_class_targets
_piece_reconstruction_loss = piece_reconstruction_loss
_raw_latent_health_metrics = raw_latent_health_metrics
_piece_reconstruction_accuracy_metrics = piece_reconstruction_accuracy_metrics
_simsiam_loss = simsiam_loss
_latent_health_metrics = latent_health_metrics
_ValidatedCheckpoint = ValidatedCheckpoint
_MAX_CONSECUTIVE_AMP_SKIPS = 16

__all__ = [
    "LunaNetwork",
    "PreparedBatch",
    "RecurrentBatchResult",
    "RepresentationCollapseError",
    "TrainingPhaseProvenance",
]


class LunaNetwork(NetworkCheckpointMixin):
    """EfficientZeroV2 learner with persistent optimizer and latent MCTS inference."""

    _COLLAPSE_GUARD_THRESHOLD = 0.05
    _COLLAPSE_GUARD_PATIENCE = 3
    _COLLAPSE_GUARD_START_STEP = 100

    _learner: EzV2LearnerConfig
    _game: ChessGame
    _amp_dtype: torch.dtype
    _global_step: int
    _trainer_iteration: int
    _lr_schedule_total_steps: int
    _lr_schedule_mismatch_warned: bool
    _loaded_checkpoint_path: Path | None
    _training_phase_provenance: TrainingPhaseProvenance | None
    _low_diversity_reports: int
    _mcts_inference_compiled: bool
    _training_compiled: bool
    _action_plane_lookup: torch.Tensor | None
    _mcts_initial_inference: InitialInference
    _mcts_recurrent_inference: RecurrentInference
    _training_initial_inference: InitialInference
    _training_representation: RepresentationInference
    _training_dynamics: DynamicsInference
    _training_prediction: PredictionInference
    _prefetch_executor: ThreadPoolExecutor | None
    board_x: int
    board_y: int
    board_z: int
    action_size: int
    device: torch.device
    nnet: EZV2Networks
    optimizer: optim.AdamW
    scaler: torch.GradScaler

    def __init__(self, game: ChessGame, learner: EzV2LearnerConfig | None = None) -> None:
        initialize_network(self._runtime(), game, learner)

    def _new_optimizer(self) -> optim.AdamW:
        return new_optimizer(self._runtime())

    def _new_grad_scaler(self) -> torch.GradScaler:
        return new_grad_scaler(self._runtime())

    @property
    def training_phase_provenance(self) -> TrainingPhaseProvenance | None:
        return self._training_phase_provenance

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def trainer_iteration(self) -> int:
        return self._trainer_iteration

    def warmup_mcts_inference(self, game: ChessGame) -> None:
        if not self._mcts_inference_compiled:
            return
        board = game.get_init_board()
        canonical = game.get_canonical_form(board, 1)
        observation = game.to_array(canonical)
        valid_moves = game.get_valid_moves(canonical, 1)
        policy, _value, latent = self.predict_with_latent(observation, valid_moves)
        _result = self.recurrent_predict(latent, int(np.argmax(policy)))
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _lr_schedule(self, step_in_run: int, total_steps: int) -> float:
        learner = self._learner
        warmup = min(max(learner.lr_warmup_steps, 0), max(total_steps - 1, 0))
        if warmup > 0 and step_in_run <= warmup:
            return learner.lr * step_in_run / warmup
        progress = (step_in_run - warmup) / max(total_steps - warmup, 1)
        cosine = 1.0 + math.cos(math.pi * min(progress, 1.0))
        return learner.lr_min + 0.5 * (learner.lr - learner.lr_min) * cosine

    def _check_representation_diversity(self, root_batch_feature_std: float, training_step: int) -> None:
        if not self._collapse_guard_enabled(training_step):
            return
        if not math.isfinite(root_batch_feature_std):
            raise RepresentationCollapseError("Root latent diversity is non-finite")
        if root_batch_feature_std >= self._COLLAPSE_GUARD_THRESHOLD:
            self._low_diversity_reports = 0
            return
        self._low_diversity_reports += 1
        logger.warning(
            "Representation diversity canary {}/{}: root batch-feature std {:.6f} < {:.3f}",
            self._low_diversity_reports,
            self._COLLAPSE_GUARD_PATIENCE,
            root_batch_feature_std,
            self._COLLAPSE_GUARD_THRESHOLD,
        )
        if self._low_diversity_reports >= self._COLLAPSE_GUARD_PATIENCE:
            raise RepresentationCollapseError(
                "Stopping state-anchored training after "
                f"{self._low_diversity_reports} consecutive collapsed-latent reports "
                f"(root batch-feature std={root_batch_feature_std:.6f})"
            )

    def _collapse_guard_enabled(self, training_step: int) -> bool:
        return (
            self._learner.model_name == "balanced_reconstruction"
            and self._learner.reconstruction_loss_weight > 0.0
            and training_step >= self._COLLAPSE_GUARD_START_STEP
        )

    def _resolve_lr_schedule_total(self, requested_total: int, current_call_steps: int) -> int:
        candidate = requested_total if requested_total > 0 else self._global_step + current_call_steps
        if self._lr_schedule_total_steps == 0:
            self._lr_schedule_total_steps = candidate
        elif self._changed_lr_horizon(requested_total):
            logger.warning(
                "Ignoring changed LR horizon {} and preserving checkpoint horizon {} steps.",
                requested_total,
                self._lr_schedule_total_steps,
            )
            self._lr_schedule_mismatch_warned = True
        return self._lr_schedule_total_steps

    def _changed_lr_horizon(self, requested_total: int) -> bool:
        return (
            requested_total > 0
            and requested_total != self._lr_schedule_total_steps
            and not self._lr_schedule_mismatch_warned
        )

    def _async_batch_prefetch(self, upcoming_steps: int = 0) -> bool:
        return async_batch_prefetch(self._runtime(), upcoming_steps)

    def _prepare_batch(
        self,
        replay: PrioritizedReplayBuffer,
        bs: int,
        unroll: int,
        td: int,
        discount: float,
        training_step: int,
        mcts_for_reanalyze: MCTSParams | None,
    ) -> PreparedBatch:
        return prepare_batch(
            self._runtime(),
            replay,
            bs,
            unroll,
            td,
            discount,
            training_step,
            mcts_for_reanalyze,
        )

    def _validate_training_inputs(
        self,
        replay: PrioritizedReplayBuffer,
        steps: int,
        bs: int,
        unroll: int,
        td: int,
    ) -> None:
        validate_training_inputs(replay, steps, bs, unroll, td)

    def _create_reanalysis_search(self, params: MCTSParams) -> ReanalysisSearch:
        return cast(ReanalysisSearch, BatchedMCTS(self._game, self, params))

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
        request = TrainingRequest(
            steps,
            total_train_steps,
            discount,
            mcts_for_reanalyze,
            TrainingProfilerConfig(
                torch_profile_steps,
                torch_profile_dir,
                torch_profile_iter,
                torch_profile_export_chrome,
                torch_profile_tensorboard_dir,
                torch_profile_with_stack,
            ),
        )
        functions = TrainingFunctions(
            _soft_ce_with_support,
            _simsiam_loss,
            _piece_reconstruction_loss,
            _piece_class_targets,
            _raw_latent_health_metrics,
            _piece_reconstruction_accuracy_metrics,
            _latent_health_metrics,
            _has_non_finite_gradients,
            _MAX_CONSECUTIVE_AMP_SKIPS,
        )
        return train_ezv2(self._runtime(), replay, request, functions)

    def _persist_compiled_mcts_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return persist_compiled_latent(self._runtime(), latent)

    def _encode_action_planes(self, actions: torch.Tensor) -> torch.Tensor:
        return encode_action_planes(self._runtime(), actions)

    def batched_initial_inference(
        self,
        obs_batch: np.ndarray,
        valid_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        return batched_initial_inference(self._runtime(), obs_batch, valid_batch)

    def batched_recurrent_inference(
        self,
        latents: torch.Tensor,
        actions: list[int],
        *,
        valid_masks: list[np.ndarray | None] | None = None,
        policy_topk: int | None = None,
    ) -> RecurrentBatchResult:
        return batched_recurrent_inference(
            self._runtime(),
            latents,
            actions,
            valid_masks=valid_masks,
            policy_topk=policy_topk,
        )

    def predict_with_latent(
        self,
        board: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, float, torch.Tensor]:
        return predict_with_latent(self._runtime(), board, valid)

    def recurrent_predict(
        self,
        latent: torch.Tensor,
        action: int,
        valid_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float, torch.Tensor]:
        return recurrent_predict(self._runtime(), latent, action, valid_mask)

    def log_model_summary(self) -> None:
        total = sum(parameter.numel() for parameter in self.nnet.parameters())
        logger.info(
            "Model: {} | {} parameters | observation={} | actions={} | channels={} | "
            "representation_blocks={} | dynamics_blocks={}",
            self._learner.model_name,
            f"{total:,}",
            (self.board_x, self.board_y, self.board_z),
            self.action_size,
            self._learner.num_channels,
            self._learner.repr_blocks,
            self._learner.dyn_blocks,
        )

    def _runtime(self) -> NetworkRuntime:
        return cast(NetworkRuntime, self)


_InitialInference = InitialInference
_RecurrentInference = RecurrentInference
