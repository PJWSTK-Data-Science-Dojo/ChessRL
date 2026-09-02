r"""Shared runtime and training value types for Luna's network wrapper.

These types are intentionally free of imports from :mod:\`luna.network\` so the
public facade can compose the implementation modules without import cycles.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import chess
import numpy as np
import torch
import torch.optim as optim

from luna.config import EzV2LearnerConfig, MCTSParams
from luna.ezv2_networks import EZV2Networks
from luna.game.chess_game import ChessGame
from luna.lc0_dataset import Lc0Batch
from luna.replay_buffer import PrioritizedReplayBuffer


class RecurrentBatchResult(NamedTuple):
    """Batched recurrent forward with either dense or sparse policy rows."""

    policy_full: np.ndarray | None
    topk_indices: np.ndarray | None
    topk_probs: np.ndarray | None
    values: np.ndarray
    rewards: np.ndarray
    next_latent: torch.Tensor


class ReanalysisBatchStats(NamedTuple):
    """Work performed while refreshing one sampled replay batch."""

    selected_samples: int
    searched_positions: int
    duration_seconds: float


class PreparedBatch(NamedTuple):
    """Collated replay batch and its observable reanalysis work."""

    collated: dict[str, np.ndarray]
    is_weights: np.ndarray
    tree_indices: list[int]
    reanalysis: ReanalysisBatchStats
    expert_anchor: Lc0Batch | None = None


class RepresentationCollapseError(RuntimeError):
    """Repeated diversity canaries detected a collapsed representation."""


@dataclass(frozen=True, slots=True)
class TrainingPhaseProvenance:
    """Immutable identity of the checkpoint that started a training phase."""

    source_checkpoint_sha256: str
    source_trainer_iteration: int
    source_global_step: int

    def as_config(self) -> dict[str, str | int]:
        return {
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_trainer_iteration": self.source_trainer_iteration,
            "source_global_step": self.source_global_step,
        }


class ValidatedCheckpoint(NamedTuple):
    state_dict: dict[str, torch.Tensor]
    global_step: int
    trainer_iteration: int
    lr_schedule_total_steps: int
    training_phase_provenance: TrainingPhaseProvenance | None


InitialInference = Callable[
    [torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
RecurrentInference = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]
RepresentationInference = Callable[[torch.Tensor], torch.Tensor]
DynamicsInference = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
PredictionInference = Callable[
    [torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor],
]


class NetworkRuntime(Protocol):
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

    def _new_optimizer(self) -> optim.AdamW: ...

    def _new_grad_scaler(self) -> torch.GradScaler: ...

    def _lr_schedule(self, step_in_run: int, total_steps: int) -> float: ...

    def _resolve_lr_schedule_total(self, requested_total: int, current_call_steps: int) -> int: ...

    def _async_batch_prefetch(self, upcoming_steps: int = 0) -> bool: ...

    def _prepare_batch(
        self,
        replay: PrioritizedReplayBuffer,
        bs: int,
        unroll: int,
        td: int,
        discount: float,
        training_step: int,
        mcts_for_reanalyze: MCTSParams | None,
    ) -> PreparedBatch: ...

    def _validate_training_inputs(
        self,
        replay: PrioritizedReplayBuffer,
        steps: int,
        batch_size: int,
        unroll: int,
        td_steps: int,
    ) -> None: ...

    def _encode_action_planes(self, actions: torch.Tensor) -> torch.Tensor: ...

    def _check_representation_diversity(self, root_batch_feature_std: float, training_step: int) -> None: ...

    def _create_reanalysis_search(self, params: MCTSParams) -> ReanalysisSearch: ...

    def batched_initial_inference(
        self,
        obs_batch: np.ndarray,
        valid_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor]: ...

    def batched_recurrent_inference(
        self,
        latent_batch: torch.Tensor,
        actions: list[int],
        *,
        valid_masks: list[np.ndarray | None] | None = None,
        policy_topk: int | None = None,
    ) -> RecurrentBatchResult: ...


class ReanalysisSearch(Protocol):
    def search_batch(
        self,
        boards: list[chess.Board],
        *,
        temp: float,
        add_exploration_noise: bool,
    ) -> list[tuple[np.ndarray, float, np.ndarray, np.ndarray]]: ...
