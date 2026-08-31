"""EfficientZeroV2 Coach: self-play -> replay buffer -> unroll training loop.

Self-play uses a sliding pool of up to ``parallel_games`` episodes with
:class:`~luna.mcts.BatchedMCTS`. Training publishes every completed iteration;
periodic Stockfish matches provide an external, fixed-opponent benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Never, cast

import numpy as np
import wandb
from loguru import logger
from tqdm import tqdm

from luna.config import (
    TrainingRunConfig,
    WandbResumeMode,
    validate_training_configuration,
    validate_wandb_resume,
    validate_wandb_run_id,
    validate_wandb_run_name,
)
from luna.game.benchmark_state import (
    BENCHMARK_STATE_NAME,
    BenchmarkState,
    load_benchmark_state,
    record_benchmark_result,
    write_benchmark_state,
)
from luna.game.chess_game import ChessGame
from luna.game.stockfish_eval import (
    StockfishEvalOutcome,
    StockfishEvalScores,
    StockfishEvalSkipped,
    _wandb_metrics,
    run_stockfish_eval,
    stockfish_evaluation_protocol,
    validate_ladder_configuration,
    validate_stockfish_configuration,
)
from luna.game.stockfish_ladder import (
    LADDER_STATE_NAME,
    fairy_ladder_protocol,
    load_fairy_ladder_state,
    run_fairy_ladder_eval,
    write_fairy_ladder_state,
)
from luna.mcts import MCTS, BatchedMCTS
from luna.network import LunaNetwork
from luna.profiling import IterProfileStats, SelfPlayMCTSTimings, write_iter_summaries_json
from luna.replay_buffer import PrioritizedReplayBuffer, Trajectory
from luna.self_play_actors import SelfPlayActorPool

_BEST_EVAL_NAME = "best_eval.json"
_BEST_EVAL_FIELD = "best_evaluation"
_BEST_EVAL_SCHEMA_VERSION = 1


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


def _managed_checkpoint_conflicts(folder: Path) -> list[str]:
    managed = list(folder.glob("checkpoint_*.pth.tar"))
    managed.extend(
        folder / name
        for name in ("latest.pth.tar", "best.pth.tar", _BEST_EVAL_NAME, BENCHMARK_STATE_NAME, LADDER_STATE_NAME)
    )
    return sorted(path.name for path in managed if path.exists())


def validate_fresh_checkpoint_target(run: TrainingRunConfig) -> None:
    """Refuse to start a new run in a directory containing managed training state."""
    if not str(run.checkpoint).strip():
        return
    folder = Path(run.checkpoint).resolve()
    conflicts = _managed_checkpoint_conflicts(folder)
    if conflicts:
        raise FileExistsError(
            f"Fresh training would overwrite managed files in {folder}: {conflicts}. "
            "Choose a new --run.checkpoint directory or resume latest.pth.tar."
        )


def validate_resume_checkpoint_target(
    run: TrainingRunConfig,
    source_checkpoint: str | Path,
    *,
    allow_evaluation_artifacts_only: bool = False,
) -> None:
    """Prevent a resume checkpoint from being merged into another managed run."""
    if not str(run.checkpoint).strip():
        return
    target = Path(run.checkpoint).resolve()
    if Path(source_checkpoint).expanduser().resolve().parent == target:
        return
    conflicts = _managed_checkpoint_conflicts(target)
    evaluation_artifacts = {BENCHMARK_STATE_NAME, LADDER_STATE_NAME, "best.pth.tar", _BEST_EVAL_NAME}
    if allow_evaluation_artifacts_only and set(conflicts) <= evaluation_artifacts:
        return
    if conflicts:
        raise FileExistsError(
            f"Resume target {target} contains managed files from another checkpoint lineage: {conflicts}. "
            "Resume in the source directory or choose a new, empty --run.checkpoint directory."
        )


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
        self._profile_sp_env_s: float = 0.0
        self._checkpoint_lineage_iteration: int | None = None
        self._checkpoint_target_validated = False
        self._replay_beta_annealing_configured = False
        self._initialize_evaluation_state = initialize_evaluation_state
        self._seed = seed
        validate_wandb_run_id(wandb_run_id)
        validate_wandb_run_name(wandb_run_name)
        validate_wandb_resume(wandb_resume)

        if wandb_project:
            phase_provenance = nnet.training_phase_provenance
            init_config = {
                "seed": seed,
                "run": asdict(run),
                "learner": asdict(nnet._learner),
                "training_phase_provenance": (phase_provenance.as_config() if phase_provenance is not None else None),
            }
            if run.stockfish_eval_every > 0:
                init_config["benchmark_protocol"] = asdict(stockfish_evaluation_protocol(run))
            if run.ladder_eval_every > 0:
                init_config["ladder_protocol"] = fairy_ladder_protocol(run)
            if wandb_run_id is None:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config=init_config,
                    tags=["chess", "ezv2"],
                )
            else:
                wandb.init(
                    project=wandb_project,
                    id=wandb_run_id,
                    name=wandb_run_name,
                    resume=wandb_resume,
                    config=init_config,
                    tags=["chess", "ezv2"],
                )
            _configure_wandb_metrics()
            logger.info("WandB initialized for project: {}", wandb_project)

    def execute_episode(self) -> Trajectory:
        """Run one self-play game using latent MCTS, collecting a full trajectory."""
        mcts = MCTS(self.game, self.nnet, self.run)

        observations: list[np.ndarray] = []
        actions: list[int] = []
        root_policies: list[np.ndarray] = []
        root_values: list[float] = []
        valids_list: list[np.ndarray] = []

        board = self.game.get_init_board()
        current_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, current_player)
            explore = episode_step < self.run.temp_threshold

            # Store the untempered visit distribution as the policy target.
            # Temperature affects only which action is played.
            pi, root_v = mcts.search_latent(
                canonical_board,
                temp=1.0,
                add_exploration_noise=explore,
            )

            obs = self.game.to_array(canonical_board)
            valid = self.game.get_valid_moves(canonical_board, 1)

            observations.append(obs)
            root_policies.append(np.array(pi, dtype=np.float32))
            root_values.append(root_v)
            valids_list.append(valid)

            if self.run.search_mode == "gumbel":
                if mcts.last_action is None:
                    raise RuntimeError("Gumbel search did not propose an action")
                action = mcts.last_action
            elif explore:
                action = int(np.random.choice(len(pi), p=pi))
            else:
                action = int(np.argmax(pi))

            current_player = self.game.push_action(board, current_player, action)
            actions.append(action)

            outcome = self.game.get_game_outcome(board, current_player)
            if outcome is not None:
                return self._trajectory_with_terminal_rewards(
                    observations,
                    actions,
                    root_policies,
                    root_values,
                    valids_list,
                    terminal_value_for_next_player=outcome,
                )

            if self.run.max_ply is not None and episode_step >= self.run.max_ply:
                return self._trajectory_with_terminal_rewards(
                    observations,
                    actions,
                    root_policies,
                    root_values,
                    valids_list,
                    terminal_value_for_next_player=0.0,
                    truncated=True,
                )

    def execute_episodes_batched(self, num_episodes: int, *, progress: bool = True) -> list[Trajectory]:
        """Run ``num_episodes`` self-play games using batched parallel MCTS.

        Uses a sliding pool of up to ``parallel_games`` games so that whenever a
        game finishes, another starts immediately, keeping GPU batch size high
        until all episodes are collected.
        """
        if num_episodes <= 0:
            return []
        if self.run.profile:
            self._profile_mcts_timings = SelfPlayMCTSTimings()
            self._profile_sp_env_s = 0.0
        pool_size = min(self.run.parallel_games, num_episodes)
        with tqdm(total=num_episodes, desc="Self Play (batched)", disable=not progress) as pbar:
            return self._run_self_play_pool(num_episodes, pool_size, pbar)

    def _run_self_play_pool(self, num_episodes: int, pool_size: int, pbar: tqdm[Never]) -> list[Trajectory]:
        """Keep the inference batch full by replacing each finished game immediately."""
        mcts_timings = self._profile_mcts_timings if self.run.profile else None
        bmcts = BatchedMCTS(self.game, self.nnet, self.run, timings=mcts_timings)
        p = pool_size

        boards = [self.game.get_init_board() for _ in range(p)]
        players = [1] * p
        steps = [0] * p
        alive = [True] * p

        obs_lists: list[list[np.ndarray]] = [[] for _ in range(p)]
        action_lists: list[list[int]] = [[] for _ in range(p)]
        policy_lists: list[list[np.ndarray]] = [[] for _ in range(p)]
        value_lists: list[list[float]] = [[] for _ in range(p)]
        valid_lists: list[list[np.ndarray]] = [[] for _ in range(p)]
        terminal_rewards: list[float] = [0.0] * p

        completed: list[Trajectory] = []

        def reset_slot(i: int) -> None:
            boards[i] = self.game.get_init_board()
            players[i] = 1
            steps[i] = 0
            obs_lists[i].clear()
            action_lists[i].clear()
            policy_lists[i].clear()
            value_lists[i].clear()
            valid_lists[i].clear()
            terminal_rewards[i] = 0.0

        while len(completed) < num_episodes:
            if self.run.profile:
                _t_env0 = time.perf_counter()

            active_indices = [i for i in range(p) if alive[i]]
            if not active_indices:
                if self.run.profile:
                    self._profile_sp_env_s += time.perf_counter() - _t_env0
                break

            explore = [steps[i] + 1 < self.run.temp_threshold for i in active_indices]

            if self.run.profile:
                _t_env1 = time.perf_counter()
                self._profile_sp_env_s += _t_env1 - _t_env0

            canonical_boards = [self.game.get_canonical_form(boards[i], players[i]) for i in active_indices]
            batch_out = bmcts.search_batch(
                canonical_boards,
                temp=1.0,
                add_exploration_noise=explore,
            )
            results_by_idx = dict(zip(active_indices, batch_out))

            if self.run.profile:
                _t_env2 = time.perf_counter()

            for j, idx in enumerate(active_indices):
                steps[idx] += 1
                pi, root_v, obs_row, valid_row = results_by_idx[idx]

                obs_lists[idx].append(obs_row)
                policy_lists[idx].append(pi)
                value_lists[idx].append(root_v)
                valid_lists[idx].append(valid_row)

                if self.run.search_mode == "gumbel":
                    proposed_action = bmcts.last_actions[j]
                    if proposed_action is None:
                        raise RuntimeError("Batched Gumbel search did not propose an action")
                    action = proposed_action
                elif explore[j]:
                    action = int(np.random.choice(len(pi), p=pi))
                else:
                    action = int(np.argmax(pi))

                players[idx] = self.game.push_action(boards[idx], players[idx], action)
                action_lists[idx].append(action)

                outcome = self.game.get_game_outcome(boards[idx], players[idx])
                if outcome is not None:
                    terminal_rewards[idx] = outcome
                    traj = self._trajectory_with_terminal_rewards(
                        obs_lists[idx],
                        action_lists[idx],
                        policy_lists[idx],
                        value_lists[idx],
                        valid_lists[idx],
                        terminal_value_for_next_player=terminal_rewards[idx],
                    )
                    if len(completed) < num_episodes:
                        completed.append(traj)
                        pbar.update(1)
                    if len(completed) >= num_episodes:
                        alive[idx] = False
                    else:
                        reset_slot(idx)
                elif self.run.max_ply is not None and steps[idx] >= self.run.max_ply:
                    terminal_rewards[idx] = 0.0
                    traj = self._trajectory_with_terminal_rewards(
                        obs_lists[idx],
                        action_lists[idx],
                        policy_lists[idx],
                        value_lists[idx],
                        valid_lists[idx],
                        terminal_value_for_next_player=terminal_rewards[idx],
                        truncated=True,
                    )
                    if len(completed) < num_episodes:
                        completed.append(traj)
                        pbar.update(1)
                    if len(completed) >= num_episodes:
                        alive[idx] = False
                    else:
                        reset_slot(idx)

            if self.run.profile:
                self._profile_sp_env_s += time.perf_counter() - _t_env2

        return completed

    def _trajectory_with_terminal_rewards(
        self,
        observations: list[np.ndarray],
        actions: list[int],
        root_policies: list[np.ndarray],
        root_values: list[float],
        valids_list: list[np.ndarray],
        terminal_value_for_next_player: float,
        truncated: bool = False,
    ) -> Trajectory:
        game_len = len(actions)
        rewards = [0.0] * game_len
        # Transition rewards use the acting (parent) player's perspective. The
        # terminal environment value is for the next player, hence the sign flip.
        rewards[-1] = -float(terminal_value_for_next_player)
        return Trajectory(
            observations=observations,
            actions=actions,
            rewards=rewards,
            root_policies=root_policies,
            root_values=root_values,
            valids=valids_list,
            truncated=truncated,
        )

    def _external_checkpoint_path(self, iteration: int) -> Path:
        candidates: list[Path] = []
        if self.nnet._loaded_checkpoint_path is not None:
            candidates.append(self.nnet._loaded_checkpoint_path)
        folder = Path(self.run.checkpoint).expanduser().resolve()
        candidates.extend((folder / f"checkpoint_{iteration}.pth.tar", folder / "latest.pth.tar"))
        for candidate in candidates:
            if candidate.is_file() and LunaNetwork.checkpoint_trainer_iteration(candidate) == iteration:
                return candidate
        raise FileNotFoundError(
            f"No immutable checkpoint is available for scheduled evaluation at iteration {iteration}"
        )

    def _initialize_external_evaluation_sidecars(self, iteration: int) -> None:
        folder = Path(self.run.checkpoint).expanduser().resolve()
        if self._initialize_evaluation_state:
            folder.mkdir(parents=True, exist_ok=True)

        if self.run.stockfish_eval_every > 0:
            benchmark_path = folder / BENCHMARK_STATE_NAME
            benchmark_protocol = asdict(stockfish_evaluation_protocol(self.run))
            required = not self._initialize_evaluation_state and iteration > self.run.stockfish_eval_every
            benchmark_state = load_benchmark_state(benchmark_path, benchmark_protocol, required=required)
            if benchmark_state.last_iteration is not None and benchmark_state.last_iteration > iteration:
                raise RuntimeError("Fixed benchmark state is newer than the loaded checkpoint")
            if self._initialize_evaluation_state and not benchmark_path.exists():
                write_benchmark_state(benchmark_path, benchmark_state)

        if self.run.ladder_eval_every > 0:
            ladder_path = folder / LADDER_STATE_NAME
            required = not self._initialize_evaluation_state and iteration > self.run.ladder_eval_every
            ladder_state = load_fairy_ladder_state(ladder_path, self.run, required=required)
            if ladder_state.last_iteration is not None and ladder_state.last_iteration > iteration:
                raise RuntimeError("Fairy ladder state is newer than the loaded checkpoint")
            if self._initialize_evaluation_state and not ladder_path.exists():
                write_fairy_ladder_state(ladder_path, ladder_state)

    def _run_fixed_benchmark(
        self,
        iteration: int,
        checkpoint_path: Path,
        checkpoint_sha256: str,
    ) -> BenchmarkState:
        folder = Path(self.run.checkpoint).expanduser().resolve()
        state_path = folder / BENCHMARK_STATE_NAME
        protocol = asdict(stockfish_evaluation_protocol(self.run))
        state = load_benchmark_state(
            state_path,
            protocol,
            required=state_path.exists() or iteration > self.run.stockfish_eval_every,
        )
        if state.last_iteration is not None and state.last_iteration > iteration:
            raise RuntimeError("Fixed benchmark state is newer than the loaded checkpoint")
        if state.last_iteration == iteration:
            if state.last_checkpoint_sha256 != checkpoint_sha256 or state.last_scores is None:
                raise RuntimeError("Fixed benchmark checkpoint identity differs from its durable result")
            scores = state.last_scores
            logger.info("Fixed benchmark iteration {} already completed; reconciling outputs", iteration)
            duration_seconds = None
        else:
            started_at = time.perf_counter()
            outcome = run_stockfish_eval(
                self.game,
                self.nnet,
                self.run,
                iteration=iteration,
                metric_prefix=None,
            )
            if isinstance(outcome, StockfishEvalSkipped):
                raise RuntimeError(f"External evaluation did not complete ({outcome.reason}): {outcome.message}")
            scores = outcome
            state = record_benchmark_result(
                state_path,
                protocol,
                iteration=iteration,
                checkpoint_sha256=checkpoint_sha256,
                scores=scores,
            )
            duration_seconds = time.perf_counter() - started_at
        if wandb.run is not None:
            metrics = _wandb_metrics(
                scores,
                iteration,
                opponent_elo=self.run.stockfish_elo,
                duration_seconds=duration_seconds,
            )
            metrics["benchmark/evaluation_step"] = state.evaluation_step
            wandb.log(metrics)
        self._update_best_from_stockfish(iteration, scores, checkpoint_path=checkpoint_path)
        return state

    def _reconcile_current_evaluations(self, iteration: int) -> None:
        if iteration < 1:
            return
        fixed_due = self.run.stockfish_eval_every > 0 and iteration % self.run.stockfish_eval_every == 0
        ladder_due = self.run.ladder_eval_every > 0 and iteration % self.run.ladder_eval_every == 0
        if not fixed_due and not ladder_due:
            return
        checkpoint_path = self._external_checkpoint_path(iteration)
        checkpoint_sha256 = self._checkpoint_sha256(checkpoint_path)
        if fixed_due:
            self._run_fixed_benchmark(iteration, checkpoint_path, checkpoint_sha256)
        if ladder_due:
            run_fairy_ladder_eval(
                self.game,
                self.nnet,
                self.run,
                iteration=iteration,
                checkpoint_sha256=checkpoint_sha256,
                state_required=(Path(self.run.checkpoint).expanduser().resolve() / LADDER_STATE_NAME).exists(),
            )

    def learn(self) -> None:
        """Full EZV2 training loop: self-play -> store in replay -> train from replay -> evaluate."""
        self._assert_checkpoint_target()
        self._assert_checkpoint_lineage()

        current_iteration = self.nnet._trainer_iteration
        start_iteration = current_iteration + 1
        evaluation_interval = self.run.stockfish_eval_every
        fixed_due_now = (
            evaluation_interval > 0 and current_iteration > 0 and current_iteration % evaluation_interval == 0
        )
        next_evaluation = (
            ((start_iteration + evaluation_interval - 1) // evaluation_interval) * evaluation_interval
            if evaluation_interval > 0
            else self.run.num_iters + 1
        )
        if fixed_due_now or next_evaluation <= self.run.num_iters:
            validate_stockfish_configuration(self.run)

        ladder_interval = self.run.ladder_eval_every
        ladder_due_now = ladder_interval > 0 and current_iteration > 0 and current_iteration % ladder_interval == 0
        next_ladder_eval = (
            ((start_iteration + ladder_interval - 1) // ladder_interval) * ladder_interval
            if ladder_interval > 0
            else self.run.num_iters + 1
        )
        if ladder_due_now or next_ladder_eval <= self.run.num_iters:
            validate_ladder_configuration(self.run)

        self._initialize_external_evaluation_sidecars(current_iteration)
        if start_iteration <= self.run.num_iters or fixed_due_now or ladder_due_now:
            self.nnet.warmup_mcts_inference(self.game)
        self._reconcile_current_evaluations(current_iteration)

        if start_iteration > self.run.num_iters:
            logger.info(
                "Checkpoint is already at iteration {}; requested total is {}. Nothing to train.",
                current_iteration,
                self.run.num_iters,
            )
            return
        if start_iteration > 1:
            logger.info("Resuming training at iteration {} of {}", start_iteration, self.run.num_iters)

        worker_count = min(self.run.self_play_workers, self.run.num_episodes)
        if worker_count <= 1:
            self._learn_iterations(start_iteration, actor_pool=None)
            return

        logger.info(
            "Starting {} persistent self-play actors with up to {} batched games each",
            worker_count,
            self.run.parallel_games,
        )
        with SelfPlayActorPool(
            self.nnet,
            self.run,
            worker_count=worker_count,
            base_seed=self._seed,
        ) as actor_pool:
            self._learn_iterations(start_iteration, actor_pool=actor_pool)

    def _learn_iterations(
        self,
        start_iteration: int,
        actor_pool: SelfPlayActorPool | None,
    ) -> None:
        profile_rows: list[IterProfileStats] = []
        if self.run.profile:
            os.makedirs(self.run.profile_dir, exist_ok=True)
            logger.info(
                "Profiling enabled: dir={} | Kineto steps: iter {} x {} | chrome={} tb_logdir={} with_stack={}",
                os.path.abspath(self.run.profile_dir),
                self.run.profile_torch_iter,
                self.run.profile_torch_steps,
                self.run.profile_export_chrome,
                self.run.profile_tensorboard_logdir,
                self.run.profile_with_stack,
            )

        for i in range(start_iteration, self.run.num_iters + 1):
            logger.info("Starting Iter #{} ...", i)
            iter_t0 = time.perf_counter()
            stats = IterProfileStats(iter_index=i)

            t0 = time.perf_counter()
            if actor_pool is None:
                trajectories = self.execute_episodes_batched(self.run.num_episodes)
            else:
                trajectories = actor_pool.collect(self.run.num_episodes, generation=i)
            stats.self_play_s = time.perf_counter() - t0
            if self.run.profile and self._profile_mcts_timings is not None:
                mt = self._profile_mcts_timings
                stats.self_play_env_s = self._profile_sp_env_s
                stats.self_play_mcts_encode_s = mt.encode_s
                stats.self_play_mcts_initial_inf_s = mt.initial_inf_s
                stats.self_play_mcts_selection_s = mt.selection_s
                stats.self_play_mcts_recurrent_inf_s = mt.recurrent_inf_s
                stats.self_play_mcts_expand_backup_s = mt.expand_backup_s
                stats.self_play_mcts_finalize_s = mt.finalize_s
                stats.self_play_search_batch_calls = mt.search_batch_calls

            t0 = time.perf_counter()
            for traj in trajectories:
                self.replay.save_trajectory(traj)
            stats.replay_save_s = time.perf_counter() - t0

            learner_batch_size = self.nnet._learner.batch_size
            if self.replay.size < learner_batch_size:
                logger.warning("Replay buffer too small ({}), skipping training.", self.replay.size)
                stats.total_s = time.perf_counter() - iter_t0
                self._log_iteration_metrics(i, trajectories, stats)
                if self.run.profile:
                    profile_rows.append(stats)
                    logger.info("\n{}\n", stats.to_log_lines())
                continue

            self._configure_replay_beta_annealing(i)

            do_kineto = (
                self.run.profile
                and self.run.profile_torch_steps > 0
                and i == self.run.profile_torch_iter
                and (self.run.profile_export_chrome or bool(self.run.profile_tensorboard_logdir))
            )
            if (
                self.run.profile
                and self.run.profile_torch_steps > 0
                and i == self.run.profile_torch_iter
                and not (self.run.profile_export_chrome or self.run.profile_tensorboard_logdir)
            ):
                logger.warning(
                    "profile_torch_steps>0 but both profile_export_chrome=False and no "
                    "profile_tensorboard_logdir — no Kineto export will be produced."
                )
            logger.info("Training from replay buffer ({} positions) ...", self.replay.size)
            t0 = time.perf_counter()
            lr_schedule_total_steps = self.nnet._lr_schedule_total_steps
            if lr_schedule_total_steps == 0:
                remaining_iterations = self.run.num_iters - i + 1
                lr_schedule_total_steps = self.nnet._global_step + remaining_iterations * self.run.train_steps_per_iter
            loss_info = self.nnet.train_ezv2(
                self.replay,
                steps=self.run.train_steps_per_iter,
                total_train_steps=lr_schedule_total_steps,
                discount=self.run.discount,
                mcts_for_reanalyze=self.run,
                torch_profile_steps=self.run.profile_torch_steps if do_kineto else 0,
                torch_profile_dir=self.run.profile_dir if do_kineto else None,
                torch_profile_iter=i,
                torch_profile_export_chrome=self.run.profile_export_chrome,
                torch_profile_tensorboard_dir=self.run.profile_tensorboard_logdir if do_kineto else None,
                torch_profile_with_stack=self.run.profile_with_stack,
            )
            stats.train_s = time.perf_counter() - t0
            logger.info("Training done: {}", loss_info)

            t0 = time.perf_counter()
            self._publish_checkpoint(i)
            stats.checkpoint_publish_s = time.perf_counter() - t0

            self._reconcile_current_evaluations(i)

            stats.total_s = time.perf_counter() - iter_t0
            self._log_iteration_metrics(i, trajectories, stats)
            if self.run.profile:
                profile_rows.append(stats)
                logger.info("\n{}\n", stats.to_log_lines())

        if self.run.profile and profile_rows:
            summary_path = Path(self.run.profile_dir) / self.run.profile_summary_json
            write_iter_summaries_json(str(summary_path), profile_rows)
            logger.info("Wrote aggregated phase timings to {}", summary_path.resolve())

    def _configure_replay_beta_annealing(self, iteration: int) -> None:
        if self._replay_beta_annealing_configured:
            return
        remaining_iterations = self.run.num_iters - iteration + 1
        remaining_sample_calls = remaining_iterations * self.run.train_steps_per_iter
        self.replay.configure_beta_annealing(remaining_sample_calls)
        self._replay_beta_annealing_configured = True
        logger.info(
            "PER beta annealing starts at iteration {} over {} optimizer sample calls",
            iteration,
            remaining_sample_calls,
        )

    def _log_iteration_metrics(
        self,
        iteration: int,
        trajectories: list[Trajectory],
        stats: IterProfileStats,
    ) -> None:
        if wandb.run is None:
            return
        games = len(trajectories)
        positions = sum(trajectory.game_length for trajectory in trajectories)
        average_ply = positions / games if games else 0.0
        truncated_games = sum(trajectory.truncated for trajectory in trajectories)
        white_wins = 0
        black_wins = 0
        draws = 0
        policy_entropy_sum = 0.0
        for trajectory in trajectories:
            probabilities = trajectory.root_policies.astype(np.float32)
            positive = probabilities > 0.0
            policy_entropy_sum -= float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
            if trajectory.truncated:
                continue
            terminal_reward = float(trajectory.rewards[-1])
            if terminal_reward == 0.0:
                draws += 1
                continue
            white_reward = terminal_reward if trajectory.game_length % 2 else -terminal_reward
            if white_reward > 0.0:
                white_wins += 1
            else:
                black_wins += 1

        max_ply_fraction = truncated_games / games if games else 0.0
        decisive_games = white_wins + black_wins
        policy_entropy = policy_entropy_sum / positions if positions else 0.0
        positions_per_second = positions / stats.self_play_s if stats.self_play_s > 0.0 else 0.0
        wandb.log(
            {
                "iteration": iteration,
                "replay_buffer_size": self.replay.size,
                "selfplay/games": games,
                "selfplay/positions": positions,
                "selfplay/avg_ply": average_ply,
                "selfplay/max_ply_fraction": max_ply_fraction,
                "selfplay/truncated_fraction": max_ply_fraction,
                "selfplay/decisive_fraction": decisive_games / games if games else 0.0,
                "selfplay/draw_fraction": draws / games if games else 0.0,
                "selfplay/white_win_fraction": white_wins / games if games else 0.0,
                "selfplay/black_win_fraction": black_wins / games if games else 0.0,
                "selfplay/policy_entropy": policy_entropy,
                "performance/self_play_seconds": stats.self_play_s,
                "performance/self_play_positions_per_second": positions_per_second,
                "performance/train_seconds": stats.train_s,
                "performance/iteration_seconds": stats.total_s,
                "replay/size": self.replay.size,
                "replay/beta": self.replay.beta,
            }
        )

    @staticmethod
    def _stockfish_normalized_score(scores: StockfishEvalScores) -> float:
        """Map Stockfish matchup to ``[0, 1]`` (draws weighted 0.5)."""

        total = scores.model_wins + scores.draws + scores.stockfish_wins
        if total <= 0:
            raise ValueError("A completed Stockfish evaluation must contain at least one game")
        return (scores.model_wins + 0.5 * scores.draws) / float(total)

    @staticmethod
    def _checkpoint_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _validate_best_record(
        value: object,
        *,
        protocol: dict[str, object],
        best_path: Path,
        trainer_iteration: int,
    ) -> dict[str, object]:
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise RuntimeError(f"Best checkpoint has no valid external-evaluation record: {best_path}")
        record = dict(value)
        iteration = record.get("iteration")
        score = record.get("score")
        source_sha256 = record.get("source_checkpoint_sha256")
        if record.get("schema_version") != _BEST_EVAL_SCHEMA_VERSION:
            raise RuntimeError(f"Best checkpoint has an unsupported external-evaluation record: {best_path}")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
            raise RuntimeError(f"Best checkpoint evaluation iteration is invalid: {best_path}")
        if iteration != trainer_iteration:
            raise RuntimeError(f"Best checkpoint evaluation iteration differs from its trainer state: {best_path}")
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise RuntimeError(f"Best checkpoint evaluation score must be finite and between zero and one: {best_path}")
        if record.get("protocol") != protocol:
            raise RuntimeError(
                f"External-evaluation protocol differs from the score in {best_path}; "
                "use a new checkpoint directory for this benchmark contract"
            )
        if (
            not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)
        ):
            raise RuntimeError(f"Best checkpoint source SHA-256 is invalid: {best_path}")
        return record

    @staticmethod
    def _write_best_metadata(metadata_path: Path, record: dict[str, object]) -> None:
        temporary = metadata_path.with_name(f".{metadata_path.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, metadata_path)
            directory_fd = os.open(metadata_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _best_evaluation_record(
        cls,
        folder: Path,
        protocol: dict[str, object],
    ) -> dict[str, object] | None:
        best_path = folder / "best.pth.tar"
        metadata_path = folder / _BEST_EVAL_NAME
        if not best_path.exists():
            if metadata_path.exists():
                raise RuntimeError(f"External-evaluation metadata exists without its best checkpoint: {metadata_path}")
            return None
        checkpoint = LunaNetwork._read_checkpoint(best_path)
        trainer_iteration = checkpoint.get("trainer_iteration")
        if isinstance(trainer_iteration, bool) or not isinstance(trainer_iteration, int):
            raise RuntimeError(f"Best checkpoint trainer iteration is invalid: {best_path}")
        record = cls._validate_best_record(
            checkpoint.get(_BEST_EVAL_FIELD),
            protocol=protocol,
            best_path=best_path,
            trainer_iteration=trainer_iteration,
        )
        try:
            metadata: object = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            metadata = None
        if metadata != record:
            cls._write_best_metadata(metadata_path, record)
        return record

    @classmethod
    def _previous_best_score(cls, folder: Path, protocol: dict[str, object]) -> float:
        record = cls._best_evaluation_record(folder, protocol)
        if record is None:
            return float("-inf")
        return float(cast(float | int, record["score"]))

    @classmethod
    def validate_best_evaluation_contract(cls, run: TrainingRunConfig) -> dict[str, object] | None:
        """Validate and reconcile the authoritative best-checkpoint record before training."""
        if not str(run.checkpoint).strip():
            return None
        folder = Path(run.checkpoint).expanduser().resolve()
        protocol = asdict(stockfish_evaluation_protocol(run))
        return cls._best_evaluation_record(folder, protocol)

    def _checkpoint_dir_usable(self) -> bool:
        return bool(str(self.run.checkpoint).strip())

    def _assert_checkpoint_target(self) -> None:
        if self._checkpoint_target_validated or not self._checkpoint_dir_usable():
            return
        source_checkpoint = self.nnet._loaded_checkpoint_path
        if source_checkpoint is None:
            validate_fresh_checkpoint_target(self.run)
        else:
            validate_resume_checkpoint_target(
                self.run,
                source_checkpoint,
                allow_evaluation_artifacts_only=self._initialize_evaluation_state,
            )
        self._checkpoint_target_validated = True

    @staticmethod
    def _numbered_checkpoints(folder: Path) -> list[tuple[int, Path]]:
        numbered: list[tuple[int, Path]] = []
        for path in folder.glob("checkpoint_*.pth.tar"):
            try:
                iteration = int(path.name.removeprefix("checkpoint_").removesuffix(".pth.tar"))
            except ValueError:
                logger.warning("Ignoring checkpoint with an invalid iteration name: {}", path)
                continue
            numbered.append((iteration, path))
        return numbered

    def _assert_checkpoint_lineage(self) -> None:
        if not self._checkpoint_dir_usable():
            return
        managed_iteration = self._managed_checkpoint_iteration(Path(self.run.checkpoint).resolve())
        self._checkpoint_lineage_iteration = managed_iteration
        if managed_iteration > self.nnet._trainer_iteration:
            raise RuntimeError(
                "The checkpoint directory contains newer training state than the loaded checkpoint; "
                "load its highest-iteration managed checkpoint or resume into a new directory"
            )

    def _managed_checkpoint_iteration(self, folder: Path) -> int:
        numbered_iteration = max(
            (iteration for iteration, _path in self._numbered_checkpoints(folder)),
            default=0,
        )
        latest_path = folder / "latest.pth.tar"
        if not latest_path.exists():
            return numbered_iteration
        latest_iteration = LunaNetwork.checkpoint_trainer_iteration(latest_path)
        return max(numbered_iteration, latest_iteration)

    def _prune_checkpoint_files(self) -> None:
        top_k = self.run.checkpoint_top_k
        if top_k is None or top_k <= 0:
            return
        if not self._checkpoint_dir_usable():
            return

        folder = Path(self.run.checkpoint).resolve()
        numbered = [path for _, path in sorted(self._numbered_checkpoints(folder), reverse=True)]
        for fp in numbered[max(1, int(top_k)) :]:
            try:
                fp.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not remove old checkpoint {}: {}", fp, exc)

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            shutil.copy2(source, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _update_best_from_stockfish(
        self,
        iteration: int,
        outcome: StockfishEvalOutcome,
        *,
        checkpoint_path: Path | None = None,
    ) -> None:
        if isinstance(outcome, StockfishEvalSkipped):
            raise RuntimeError(f"External evaluation did not complete ({outcome.reason}): {outcome.message}")
        if not self._checkpoint_dir_usable():
            return

        folder = Path(self.run.checkpoint).resolve()
        fp = checkpoint_path if checkpoint_path is not None else folder / f"checkpoint_{iteration}.pth.tar"
        if not fp.is_file():
            raise FileNotFoundError(f"Evaluated checkpoint is missing: {fp}")
        checkpoint_iteration = LunaNetwork.checkpoint_trainer_iteration(fp)
        if checkpoint_iteration != iteration:
            raise RuntimeError(
                f"Evaluated checkpoint iteration {checkpoint_iteration} differs from requested iteration {iteration}: {fp}"
            )
        if self.nnet._trainer_iteration != iteration:
            raise RuntimeError("In-memory model differs from the externally evaluated checkpoint iteration")

        sf_score = self._stockfish_normalized_score(outcome)
        protocol = asdict(stockfish_evaluation_protocol(self.run))
        previous_score = self._previous_best_score(folder, protocol)
        if sf_score <= previous_score:
            return
        record: dict[str, object] = {
            "schema_version": _BEST_EVAL_SCHEMA_VERSION,
            "iteration": iteration,
            "score": sf_score,
            "protocol": protocol,
            "source_checkpoint_sha256": self._checkpoint_sha256(fp),
        }
        self.nnet.save_checkpoint(
            folder=self.run.checkpoint,
            filename="best.pth.tar",
            extra_state={_BEST_EVAL_FIELD: record},
        )
        self._write_best_metadata(folder / _BEST_EVAL_NAME, record)
        logger.info("New best external score {:.3f} at iteration {}", sf_score, iteration)

    def _publish_checkpoint(self, iteration: int) -> None:
        if not self._checkpoint_dir_usable():
            logger.warning(
                'run.checkpoint "" or unset-like; skipping checkpoint_{} and best.pth.tar writes.',
                iteration,
            )
            return

        ck_name = f"checkpoint_{iteration}.pth.tar"
        folder = Path(self.run.checkpoint).resolve()
        checkpoint_path = folder / ck_name
        if checkpoint_path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable numbered checkpoint: {checkpoint_path}")
        if self._checkpoint_lineage_iteration is None:
            self._checkpoint_lineage_iteration = self._managed_checkpoint_iteration(folder)
        numbered_iteration = max(
            (saved_iteration for saved_iteration, _path in self._numbered_checkpoints(folder)),
            default=0,
        )
        latest_existing = max(self._checkpoint_lineage_iteration, numbered_iteration)
        if latest_existing >= iteration:
            raise FileExistsError(
                f"Refusing non-monotonic checkpoint {checkpoint_path}; "
                f"directory already contains iteration {latest_existing}"
            )
        previous_iteration = self.nnet._trainer_iteration
        self.nnet._trainer_iteration = iteration
        numbered_saved = False
        try:
            self.nnet.save_checkpoint(
                folder=self.run.checkpoint,
                filename=ck_name,
            )
            numbered_saved = True
        finally:
            if not numbered_saved:
                self.nnet._trainer_iteration = previous_iteration
        self._atomic_copy(folder / ck_name, folder / "latest.pth.tar")
        self._checkpoint_lineage_iteration = iteration
        self._prune_checkpoint_files()
