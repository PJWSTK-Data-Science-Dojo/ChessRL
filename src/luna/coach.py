"""EfficientZeroV2 self-play and training orchestration."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Never

import chess
import numpy as np
import wandb
from loguru import logger
from tqdm import tqdm

import luna.coach_batched_self_play as coach_batched_self_play
import luna.coach_checkpoints as coach_checkpoints
import luna.coach_evaluation as coach_evaluation
import luna.coach_metrics as coach_metrics
import luna.coach_self_play as coach_self_play
import luna.coach_training as coach_training
from luna.config import (
    TrainingRunConfig,
    WandbResumeMode,
    validate_training_configuration,
    validate_wandb_resume,
    validate_wandb_run_id,
    validate_wandb_run_name,
)
from luna.game.benchmark_state import BenchmarkState
from luna.game.chess_game import ChessGame
from luna.game.stockfish_eval import (
    StockfishEvalOutcome,
    StockfishEvalScores,
    stockfish_evaluation_protocol,
)
from luna.game.stockfish_ladder import fairy_ladder_protocol
from luna.network import LunaNetwork
from luna.profiling import IterProfileStats, SelfPlayMCTSTimings
from luna.replay_buffer import PrioritizedReplayBuffer, Trajectory
from luna.self_play_actors import SelfPlayActorPool

_self_play_exploration_enabled = coach_self_play.self_play_exploration_enabled
_optimizer_steps_for_positions = coach_training.optimizer_steps_for_positions
_select_self_play_action = coach_self_play.select_self_play_action
_enables_threefold_claim = coach_self_play.enables_threefold_claim
_non_repetition_actions = coach_self_play.non_repetition_actions
validate_fresh_checkpoint_target = coach_checkpoints.validate_fresh_checkpoint_target
validate_resume_checkpoint_target = coach_checkpoints.validate_resume_checkpoint_target

__all__ = [
    "Coach",
    "_enables_threefold_claim",
    "_non_repetition_actions",
    "_optimizer_steps_for_positions",
    "_select_self_play_action",
    "_self_play_exploration_enabled",
    "validate_fresh_checkpoint_target",
    "validate_resume_checkpoint_target",
]


def _configure_wandb_metrics() -> None:
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("iteration")
    wandb.define_metric("replay_buffer_size", step_metric="iteration")
    wandb.define_metric("selfplay/*", step_metric="iteration")
    wandb.define_metric("performance/*", step_metric="iteration")
    wandb.define_metric("replay/*", step_metric="iteration")
    wandb.define_metric("benchmark/*", step_metric="iteration")
    wandb.define_metric("ladder/evaluation_step")
    wandb.define_metric("ladder/*", step_metric="ladder/evaluation_step")


class Coach:
    """Orchestrates EZV2 self-play, replay storage, and unroll learning."""

    game: ChessGame
    nnet: LunaNetwork
    run: TrainingRunConfig
    replay: PrioritizedReplayBuffer

    def __init__(
        self,
        game: ChessGame,
        nnet: LunaNetwork,
        run: TrainingRunConfig,
        wandb_project: str | None = None,
        wandb_run_id: str | None = None,
        wandb_run_name: str | None = None,
        wandb_resume: WandbResumeMode = "allow",
        initialize_evaluation_state: bool = False,
        seed: int = 0,
    ) -> None:
        self.game = game
        self.nnet = nnet
        self.run = run
        validate_training_configuration(run, nnet._learner)
        if not math.isclose(run.discount, nnet._learner.discount, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("run.discount and learner.discount must match")
        self.replay = PrioritizedReplayBuffer(
            capacity=run.replay_capacity,
            alpha=run.per_alpha,
            beta=run.per_beta,
        )
        self._profile_mcts_timings: SelfPlayMCTSTimings | None = None
        self._profile_sp_env_s = 0.0
        self._checkpoint_lineage_iteration: int | None = None
        self._checkpoint_target_validated = False
        self._initialize_evaluation_state = initialize_evaluation_state
        self._seed = seed
        validate_wandb_run_id(wandb_run_id)
        validate_wandb_run_name(wandb_run_name)
        validate_wandb_resume(wandb_resume)

        if wandb_project:
            self._initialize_wandb(
                wandb_project,
                wandb_run_id=wandb_run_id,
                wandb_run_name=wandb_run_name,
                wandb_resume=wandb_resume,
            )

    def _initialize_wandb(
        self,
        project: str,
        *,
        wandb_run_id: str | None,
        wandb_run_name: str | None,
        wandb_resume: WandbResumeMode,
    ) -> None:
        phase_provenance = self.nnet.training_phase_provenance
        init_config = {
            "seed": self._seed,
            "run": asdict(self.run),
            "learner": asdict(self.nnet._learner),
            "training_phase_provenance": (phase_provenance.as_config() if phase_provenance is not None else None),
        }
        if self.run.stockfish_eval_every > 0:
            init_config["benchmark_protocol"] = asdict(stockfish_evaluation_protocol(self.run))
        if self.run.ladder_eval_every > 0:
            init_config["ladder_protocol"] = fairy_ladder_protocol(self.run)
        if wandb_run_id is None:
            wandb.init(
                project=project,
                name=wandb_run_name,
                config=init_config,
                tags=["chess", "ezv2"],
            )
        else:
            wandb.init(
                project=project,
                id=wandb_run_id,
                name=wandb_run_name,
                resume=wandb_resume,
                config=init_config,
                tags=["chess", "ezv2"],
            )
        _configure_wandb_metrics()
        logger.info("WandB initialized for project: {}", project)

    def execute_episode(self) -> Trajectory:
        return coach_self_play.execute_episode(self)

    def execute_episodes_batched(self, num_episodes: int, *, progress: bool = True) -> list[Trajectory]:
        return coach_batched_self_play.execute_episodes_batched(self, num_episodes, progress=progress)

    def _run_self_play_pool(
        self,
        num_episodes: int,
        pool_size: int,
        pbar: tqdm[Never],
    ) -> list[Trajectory]:
        return coach_batched_self_play.run_self_play_pool(self, num_episodes, pool_size, pbar)

    def _trajectory_with_terminal_rewards(
        self,
        observations: list[np.ndarray],
        actions: list[int],
        root_policies: list[np.ndarray],
        root_values: list[float],
        valids_list: list[np.ndarray],
        terminal_value_for_next_player: float,
        truncated: bool = False,
        termination: chess.Termination | None = None,
        repetition_guard_attempts: int = 0,
        repetition_guard_interventions: int = 0,
        repetition_guard_forced_fallbacks: int = 0,
        repetition_guard_excluded_actions: int = 0,
    ) -> Trajectory:
        return coach_self_play.trajectory_with_terminal_rewards(
            observations,
            actions,
            root_policies,
            root_values,
            valids_list,
            terminal_value_for_next_player,
            truncated,
            termination,
            repetition_guard_attempts,
            repetition_guard_interventions,
            repetition_guard_forced_fallbacks,
            repetition_guard_excluded_actions,
        )

    def _external_checkpoint_path(self, iteration: int) -> Path:
        return coach_evaluation.external_checkpoint_path(self, iteration)

    def _initialize_external_evaluation_sidecars(self, iteration: int) -> None:
        coach_evaluation.initialize_external_evaluation_sidecars(self, iteration)

    def _run_fixed_benchmark(
        self,
        iteration: int,
        checkpoint_path: Path,
        checkpoint_sha256: str,
    ) -> BenchmarkState:
        return coach_evaluation.run_fixed_benchmark(self, iteration, checkpoint_path, checkpoint_sha256)

    def _reconcile_current_evaluations(self, iteration: int) -> None:
        coach_evaluation.reconcile_current_evaluations(self, iteration)

    def learn(self) -> None:
        coach_training.learn(self)

    def _learn_iterations(
        self,
        start_iteration: int,
        actor_pool: SelfPlayActorPool | None,
    ) -> None:
        coach_training.learn_iterations(self, start_iteration, actor_pool)

    def _configure_replay_beta_annealing(self, iteration: int, optimizer_steps: int) -> None:
        coach_training.configure_replay_beta_annealing(self, iteration, optimizer_steps)

    def _log_iteration_metrics(
        self,
        iteration: int,
        trajectories: list[Trajectory],
        stats: IterProfileStats,
        optimizer_steps: int = 0,
    ) -> None:
        coach_metrics.log_iteration_metrics(self, iteration, trajectories, stats, optimizer_steps)

    @staticmethod
    def _stockfish_normalized_score(scores: StockfishEvalScores) -> float:
        return coach_checkpoints.stockfish_normalized_score(scores)

    @staticmethod
    def _checkpoint_sha256(path: Path) -> str:
        return coach_checkpoints.checkpoint_sha256(path)

    @staticmethod
    def _validate_best_record(
        value: object,
        *,
        protocol: dict[str, object],
        best_path: Path,
        trainer_iteration: int,
    ) -> dict[str, object]:
        return coach_checkpoints.validate_best_record(
            value,
            protocol=protocol,
            best_path=best_path,
            trainer_iteration=trainer_iteration,
        )

    @staticmethod
    def _write_best_metadata(metadata_path: Path, record: dict[str, object]) -> None:
        coach_checkpoints.write_best_metadata(metadata_path, record)

    @classmethod
    def _best_evaluation_record(
        cls,
        folder: Path,
        protocol: dict[str, object],
    ) -> dict[str, object] | None:
        return coach_checkpoints.best_evaluation_record(folder, protocol)

    @classmethod
    def _previous_best_score(cls, folder: Path, protocol: dict[str, object]) -> float:
        return coach_checkpoints.previous_best_score(folder, protocol)

    @classmethod
    def validate_best_evaluation_contract(cls, run: TrainingRunConfig) -> dict[str, object] | None:
        return coach_checkpoints.validate_best_evaluation_contract(run)

    def _checkpoint_dir_usable(self) -> bool:
        return coach_checkpoints.checkpoint_dir_usable(self)

    def _assert_checkpoint_target(self) -> None:
        coach_checkpoints.assert_checkpoint_target(self)

    @staticmethod
    def _numbered_checkpoints(folder: Path) -> list[tuple[int, Path]]:
        return coach_checkpoints.numbered_checkpoints(folder)

    def _assert_checkpoint_lineage(self) -> None:
        coach_checkpoints.assert_checkpoint_lineage(self)

    def _managed_checkpoint_iteration(self, folder: Path) -> int:
        return coach_checkpoints.managed_checkpoint_iteration(self, folder)

    def _prune_checkpoint_files(self) -> None:
        coach_checkpoints.prune_checkpoint_files(self)

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        coach_checkpoints.atomic_copy(source, destination)

    def _update_best_from_stockfish(
        self,
        iteration: int,
        outcome: StockfishEvalOutcome,
        *,
        checkpoint_path: Path | None = None,
    ) -> None:
        coach_checkpoints.update_best_from_stockfish(
            self,
            iteration,
            outcome,
            checkpoint_path=checkpoint_path,
        )

    def _publish_checkpoint(self, iteration: int) -> None:
        coach_checkpoints.publish_checkpoint(self, iteration)
