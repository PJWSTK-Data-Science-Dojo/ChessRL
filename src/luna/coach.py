"""EfficientZeroV2 Coach: self-play -> replay buffer -> unroll training loop.

Self-play uses a sliding pool of up to ``parallel_games`` episodes with
:class:`~luna.mcts.BatchedMCTS`. Training publishes every completed iteration;
periodic Stockfish matches provide an external, fixed-opponent benchmark.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Never

import numpy as np
import wandb
from loguru import logger
from tqdm import tqdm

from luna.config import TrainingRunConfig
from luna.game.chess_game import ChessGame
from luna.game.stockfish_eval import (
    StockfishEvalOutcome,
    StockfishEvalScores,
    StockfishEvalSkipped,
    run_stockfish_eval,
    validate_stockfish_configuration,
)
from luna.mcts import MCTS, BatchedMCTS
from luna.network import LunaNetwork
from luna.profiling import IterProfileStats, SelfPlayMCTSTimings, write_iter_summaries_json
from luna.replay_buffer import PrioritizedReplayBuffer, Trajectory

_BEST_EVAL_NAME = "best_eval.json"


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
    ) -> None:
        self.game = game
        self.nnet = nnet
        self.run = run
        self.replay = PrioritizedReplayBuffer(
            capacity=run.replay_capacity,
            alpha=run.per_alpha,
            beta=run.per_beta,
        )
        if abs(self.nnet._learner.discount - self.run.discount) > 1e-9:
            logger.warning(
                "learner.discount ({}) != run.discount ({}); using run.discount for both MCTS and TD targets.",
                self.nnet._learner.discount,
                self.run.discount,
            )
        self.nnet._learner.discount = self.run.discount
        self._profile_mcts_timings: SelfPlayMCTSTimings | None = None
        self._profile_sp_env_s: float = 0.0

        if wandb_project:
            wandb.init(
                project=wandb_project,
                config=asdict(run),
                tags=["chess", "ezv2"],
            )
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

            board, current_player = self.game.get_next_state(board, current_player, action)
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
                )

    def execute_episodes_batched(self, num_episodes: int) -> list[Trajectory]:
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
        with tqdm(total=num_episodes, desc="Self Play (batched)") as pbar:
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

                boards[idx], players[idx] = self.game.get_next_state(boards[idx], players[idx], action)
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
        )

    def learn(self) -> None:
        """Full EZV2 training loop: self-play -> store in replay -> train from replay -> evaluate."""
        train_steps_per_iter = self.run.train_steps_per_iter
        total_train_steps = self.run.num_iters * train_steps_per_iter

        start_iteration = self.nnet._trainer_iteration + 1
        if start_iteration > self.run.num_iters:
            logger.info(
                "Checkpoint is already at iteration {}; requested total is {}. Nothing to train.",
                self.nnet._trainer_iteration,
                self.run.num_iters,
            )
            return
        if start_iteration > 1:
            logger.info("Resuming training at iteration {} of {}", start_iteration, self.run.num_iters)

        evaluation_interval = self.run.stockfish_eval_every
        if evaluation_interval > 0:
            next_evaluation = ((start_iteration + evaluation_interval - 1) // evaluation_interval) * evaluation_interval
            if next_evaluation <= self.run.num_iters:
                validate_stockfish_configuration(self.run)

        self.nnet.warmup_mcts_inference(self.game)

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
            trajectories = self.execute_episodes_batched(self.run.num_episodes)
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
                if self.run.profile:
                    stats.total_s = time.perf_counter() - iter_t0
                    profile_rows.append(stats)
                    logger.info("\n{}\n", stats.to_log_lines())
                continue

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
            loss_info = self.nnet.train_ezv2(
                self.replay,
                steps=train_steps_per_iter,
                total_train_steps=total_train_steps,
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

            if wandb.run is not None:
                wandb.log(
                    {
                        "iteration": i,
                        "replay_buffer_size": self.replay.size,
                    }
                )

            t0 = time.perf_counter()
            self._publish_checkpoint(i)
            stats.checkpoint_publish_s = time.perf_counter() - t0

            if self.run.stockfish_eval_every > 0 and i % self.run.stockfish_eval_every == 0:
                sf_outcome = run_stockfish_eval(self.game, self.nnet, self.run, iteration=i)
                self._update_best_from_stockfish(i, sf_outcome)

            stats.total_s = time.perf_counter() - iter_t0
            if self.run.profile:
                profile_rows.append(stats)
                logger.info("\n{}\n", stats.to_log_lines())

        if self.run.profile and profile_rows:
            summary_path = Path(self.run.profile_dir) / self.run.profile_summary_json
            write_iter_summaries_json(str(summary_path), profile_rows)
            logger.info("Wrote aggregated phase timings to {}", summary_path.resolve())

    @staticmethod
    def _stockfish_normalized_score(scores: StockfishEvalScores) -> float:
        """Map Stockfish matchup to ``[0, 1]`` (draws weighted 0.5)."""

        total = scores.model_wins + scores.draws + scores.stockfish_wins
        if total <= 0:
            raise ValueError("A completed Stockfish evaluation must contain at least one game")
        return (scores.model_wins + 0.5 * scores.draws) / float(total)

    @staticmethod
    def _previous_best_score(metadata_path: Path) -> float:
        if not metadata_path.exists():
            return float("-inf")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            score: object = payload["score"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Could not read external-evaluation metadata: {metadata_path}") from exc
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise RuntimeError(f"External-evaluation score is not numeric: {metadata_path}")
        return float(score)

    @staticmethod
    def _write_best_metadata(metadata_path: Path, iteration: int, score: float) -> None:
        temporary = metadata_path.with_name(f".{metadata_path.name}.tmp-{os.getpid()}")
        try:
            temporary.write_text(
                json.dumps({"iteration": iteration, "score": score}, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, metadata_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _checkpoint_dir_usable(self) -> bool:
        return bool(str(self.run.checkpoint).strip())

    def _prune_checkpoint_files(self) -> None:
        top_k = self.run.checkpoint_top_k
        if top_k is None or top_k <= 0:
            return
        if not self._checkpoint_dir_usable():
            return

        folder = Path(self.run.checkpoint).resolve()
        numbered_with_iterations: list[tuple[int, Path]] = []
        for path in folder.glob("checkpoint_*.pth.tar"):
            try:
                iteration = int(path.name.removeprefix("checkpoint_").removesuffix(".pth.tar"))
            except ValueError:
                logger.warning("Ignoring checkpoint with an invalid iteration name: {}", path)
                continue
            numbered_with_iterations.append((iteration, path))
        numbered = [path for _, path in sorted(numbered_with_iterations, reverse=True)]
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
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _update_best_from_stockfish(self, iteration: int, outcome: StockfishEvalOutcome) -> None:
        if isinstance(outcome, StockfishEvalSkipped):
            raise RuntimeError(f"External evaluation did not complete ({outcome.reason}): {outcome.message}")
        if not self._checkpoint_dir_usable():
            return

        folder = Path(self.run.checkpoint).resolve()
        fp = folder / f"checkpoint_{iteration}.pth.tar"
        if not fp.is_file():
            raise FileNotFoundError(f"Evaluated checkpoint is missing: {fp}")

        sf_score = self._stockfish_normalized_score(outcome)
        metadata_path = folder / _BEST_EVAL_NAME
        previous_score = self._previous_best_score(metadata_path)
        if sf_score <= previous_score:
            return
        self._atomic_copy(fp, folder / "best.pth.tar")
        self._write_best_metadata(metadata_path, iteration, sf_score)
        logger.info("New best external score {:.3f} at iteration {}", sf_score, iteration)

    def _publish_checkpoint(self, iteration: int) -> None:
        if not self._checkpoint_dir_usable():
            logger.warning(
                'run.checkpoint "" or unset-like; skipping checkpoint_{} and best.pth.tar writes.',
                iteration,
            )
            return

        ck_name = f"checkpoint_{iteration}.pth.tar"
        self.nnet._trainer_iteration = iteration
        self.nnet.save_checkpoint(
            folder=self.run.checkpoint,
            filename=ck_name,
        )
        folder = Path(self.run.checkpoint).resolve()
        self._atomic_copy(folder / ck_name, folder / "latest.pth.tar")
        self._prune_checkpoint_files()
