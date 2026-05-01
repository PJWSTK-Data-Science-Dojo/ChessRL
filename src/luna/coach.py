"""EfficientZeroV2 Coach: self-play -> replay buffer -> unroll training loop.

Self-play uses a sliding pool of up to ``parallel_games`` episodes with
:class:`~luna.mcts.BatchedMCTS`, refilling finished games so recurrent inference
stays batched. Arena pits batch up to ``arena_parallel_games`` games per ply.
"""

import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

from .config import MCTSParams, TrainingRunConfig
from .game.chess_game import DRAW_VALUE, ChessGame
from .mcts import MCTS, BatchedMCTS
from .network import LunaNetwork
from .profiling import IterProfileStats, SelfPlayMCTSTimings, write_iter_summaries_json
from .replay_buffer import PrioritizedReplayBuffer, Trajectory


class Coach:
    """Orchestrates EZV2 self-play data collection, replay storage, and learning."""

    game: ChessGame
    nnet: LunaNetwork
    pnet: LunaNetwork
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
        self.pnet = nnet.__class__(game, nnet._learner)
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

        # Initialize WandB if project name provided
        if wandb_project and wandb is not None:
            wandb.init(
                project=wandb_project,
                config=asdict(run),
                tags=["chess", "ezv2"],
            )
            logger.info("WandB initialized for project: {}", wandb_project)
        elif wandb_project and wandb is None:
            logger.warning("WandB project specified but wandb not installed. Install with: uv add wandb")

    # ------------------------------------------------------------------
    # Single-game self-play (fallback / arena)
    # ------------------------------------------------------------------
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
            temp = 1.0 if episode_step < self.run.temp_threshold else 0.0

            pi, root_v = mcts.search_latent(canonical_board)

            obs = self.game.to_array(canonical_board)
            valid = self.game.get_valid_moves(canonical_board, 1)

            observations.append(obs)
            root_policies.append(np.array(pi, dtype=np.float32))
            root_values.append(root_v)
            valids_list.append(valid)

            if temp == 0:
                action = int(np.argmax(pi))
            else:
                action = int(np.random.choice(len(pi), p=pi))

            board, current_player = self.game.get_next_state(board, current_player, action)
            actions.append(action)

            r = self.game.get_game_ended(board, current_player)
            if abs(r) > 1e-8:
                return self._trajectory_with_terminal_rewards(
                    observations, actions, root_policies, root_values, valids_list, terminal_r=r,
                )

            if self.run.max_ply is not None and episode_step >= self.run.max_ply:
                return self._trajectory_with_terminal_rewards(
                    observations, actions, root_policies, root_values, valids_list, terminal_r=DRAW_VALUE,
                )

    # ------------------------------------------------------------------
    # Batched parallel self-play
    # ------------------------------------------------------------------
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

    def _run_self_play_pool(self, num_episodes: int, pool_size: int, pbar: tqdm) -> list[Trajectory]:
        """Execute self-play with sliding game pool for batched MCTS inference.

        Maintains a pool of up to pool_size active episodes, refilling finished games
        immediately to keep GPU batches full. Uses BatchedMCTS to amortize network overhead
        across parallel positions.

        Args:
            num_episodes: Total episodes to generate.
            pool_size: Max concurrent games in the pool.
            pbar: Progress bar for tracking completion.

        Returns:
            List of completed game trajectories with (obs, policy, reward) per timestep.
        """
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

            temps = [
                1.0 if steps[i] + 1 < self.run.temp_threshold else 0.0
                for i in active_indices
            ]

            if self.run.profile:
                _t_env1 = time.perf_counter()
                self._profile_sp_env_s += _t_env1 - _t_env0

            temp_groups: dict[float, list[int]] = {}
            for j, idx in enumerate(active_indices):
                temp_groups.setdefault(temps[j], []).append(idx)

            results_by_idx: dict[int, tuple[np.ndarray, float, np.ndarray, np.ndarray]] = {}
            for temp_key, idxs in temp_groups.items():
                canonical_boards = [
                    self.game.get_canonical_form(boards[i], players[i]) for i in idxs
                ]
                batch_out = bmcts.search_batch(canonical_boards, temp=temp_key)
                for local_j, game_i in enumerate(idxs):
                    results_by_idx[game_i] = batch_out[local_j]

            if self.run.profile:
                _t_env2 = time.perf_counter()

            for j, idx in enumerate(active_indices):
                steps[idx] += 1
                pi, root_v, obs_row, valid_row = results_by_idx[idx]

                obs_lists[idx].append(obs_row)
                policy_lists[idx].append(pi)
                value_lists[idx].append(root_v)
                valid_lists[idx].append(valid_row)

                t = temps[j]
                if t == 0:
                    action = int(np.argmax(pi))
                else:
                    action = int(np.random.choice(len(pi), p=pi))

                boards[idx], players[idx] = self.game.get_next_state(boards[idx], players[idx], action)
                action_lists[idx].append(action)

                r = self.game.get_game_ended(boards[idx], players[idx])
                if abs(r) > 1e-8:
                    terminal_rewards[idx] = r
                    traj = self._trajectory_with_terminal_rewards(
                        obs_lists[idx],
                        action_lists[idx],
                        policy_lists[idx],
                        value_lists[idx],
                        valid_lists[idx],
                        terminal_r=terminal_rewards[idx],
                    )
                    if len(completed) < num_episodes:
                        completed.append(traj)
                        pbar.update(1)
                    if len(completed) >= num_episodes:
                        alive[idx] = False
                    else:
                        reset_slot(idx)
                elif self.run.max_ply is not None and steps[idx] >= self.run.max_ply:
                    terminal_rewards[idx] = DRAW_VALUE
                    traj = self._trajectory_with_terminal_rewards(
                        obs_lists[idx],
                        action_lists[idx],
                        policy_lists[idx],
                        value_lists[idx],
                        valid_lists[idx],
                        terminal_r=terminal_rewards[idx],
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

    @staticmethod
    def _arena_mcts_params(run: TrainingRunConfig) -> MCTSParams:
        sims = run.arena_num_mcts_sims if run.arena_num_mcts_sims is not None else run.num_mcts_sims
        return MCTSParams(
            num_mcts_sims=sims,
            cpuct=run.cpuct,
            dir_noise=False,
            dir_alpha=run.dir_alpha,
            discount=run.discount,
            recurrent_policy_topk=run.recurrent_policy_topk,
        )

    def _play_arena_games_batched(
        self,
        white: LunaNetwork,
        black: LunaNetwork,
        mcts_params: MCTSParams,
        num_games: int,
    ) -> list[float]:
        """Pit current vs previous model with parallel game execution.

        Runs num_games head-to-head matches with batched inference for speed.
        Each model plays both colors to reduce variance.

        Args:
            white: Network playing white pieces.
            black: Network playing black pieces.
            mcts_params: MCTS search parameters for both players.
            num_games: Number of parallel games to play.

        Returns:
            List of game results from white's perspective (1.0 = win, 0.0 = loss, 0.5 = draw).
        """
        game = self.game
        max_ply = self.run.max_ply
        white_bm = BatchedMCTS(game, white, mcts_params)
        black_bm = BatchedMCTS(game, black, mcts_params)

        boards = [game.get_init_board() for _ in range(num_games)]
        players = [1] * num_games
        steps = [0] * num_games
        done = [False] * num_games
        results = [0.0] * num_games

        while not all(done):
            active = [i for i in range(num_games) if not done[i]]
            p_side = players[active[0]]
            for i in active:
                if players[i] != p_side:
                    raise RuntimeError("internal: arena batch games desynchronized")
            bm = white_bm if p_side == 1 else black_bm
            canonicals = [game.get_canonical_form(boards[i], players[i]) for i in active]
            batch_res = bm.search_batch(canonicals, temp=0.0)
            for j, idx in enumerate(active):
                pi, _rv, _obs, _valid = batch_res[j]
                action = int(np.argmax(pi))
                steps[idx] += 1
                boards[idx], players[idx] = game.get_next_state(boards[idx], players[idx], action)
                r = game.get_game_ended(boards[idx], players[idx])
                if abs(r) > 1e-8:
                    results[idx] = float(players[idx] * r)
                    done[idx] = True
                elif max_ply is not None and steps[idx] >= max_ply:
                    results[idx] = 0.0
                    done[idx] = True

        return results

    def _trajectory_with_terminal_rewards(
        self,
        observations: list[np.ndarray],
        actions: list[int],
        root_policies: list[np.ndarray],
        root_values: list[float],
        valids_list: list[np.ndarray],
        terminal_r: float,
    ) -> Trajectory:
        game_len = len(actions)
        rewards = [0.0] * game_len
        # Terminal reward is already from the correct player's perspective
        # (ChessGame.get_terminal_r returns +1 for winner, -1 for loser)
        rewards[-1] = float(terminal_r)
        return Trajectory(
            observations=observations,
            actions=actions,
            rewards=rewards,
            root_policies=root_policies,
            root_values=root_values,
            valids=valids_list,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def learn(self) -> None:
        """Full EZV2 training loop: self-play -> store in replay -> train from replay -> evaluate."""
        train_steps_per_iter = self.run.train_steps_per_iter
        total_train_steps = self.run.num_iters * train_steps_per_iter

        self.nnet.warmup_mcts_inference(self.game)
        self.pnet.warmup_mcts_inference(self.game)

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

        steps_completed = 0
        for i in range(1, self.run.num_iters + 1):
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

            if self.replay.size < self.run.batch_size:
                logger.warning("Replay buffer too small ({}), skipping training.", self.replay.size)
                if self.run.profile:
                    stats.total_s = time.perf_counter() - iter_t0
                    profile_rows.append(stats)
                    logger.info("\n{}\n", stats.to_log_lines())
                continue

            t0 = time.perf_counter()
            self.nnet.save_checkpoint(folder=self.run.checkpoint, filename="temp.pth.tar")
            self.pnet.load_checkpoint(folder=self.run.checkpoint, filename="temp.pth.tar")
            stats.checkpoint_io_s = time.perf_counter() - t0

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
                start_step=steps_completed,
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
            steps_completed += train_steps_per_iter
            logger.info("Training done: {}", loss_info)

            logger.info("PITTING AGAINST PREVIOUS VERSION")
            arena_params = self._arena_mcts_params(self.run)
            batch_cap = max(1, self.run.arena_parallel_games)

            num_arena_pits = max(1, int(self.run.arena_compare / 2))
            nwins = 0
            pwins = 0
            draws = 0

            t_arena0 = time.perf_counter()
            with tqdm(total=num_arena_pits * 2, desc="Arena (batched)") as aprog:
                remaining = num_arena_pits
                while remaining > 0:
                    b = min(batch_cap, remaining)
                    for result in self._play_arena_games_batched(self.pnet, self.nnet, arena_params, b):
                        if result > 0.5:
                            pwins += 1
                        elif result < -0.5:
                            nwins += 1
                        else:
                            draws += 1
                        aprog.update(1)
                    remaining -= b

                remaining = num_arena_pits
                while remaining > 0:
                    b = min(batch_cap, remaining)
                    for result in self._play_arena_games_batched(self.nnet, self.pnet, arena_params, b):
                        if result > 0.5:
                            nwins += 1
                        elif result < -0.5:
                            pwins += 1
                        else:
                            draws += 1
                        aprog.update(1)
                    remaining -= b
            stats.arena_s = time.perf_counter() - t_arena0

            logger.info("NEW/PREV WINS: {} / {} ; DRAWS: {}", nwins, pwins, draws)

            # Log iteration metrics to WandB
            if wandb is not None and wandb.run is not None:
                total_games = nwins + pwins + draws
                win_rate = nwins / total_games if total_games > 0 else 0.0
                wandb.log({
                    "arena/new_wins": nwins,
                    "arena/prev_wins": pwins,
                    "arena/draws": draws,
                    "arena/win_rate": win_rate,
                    "arena/total_games": total_games,
                    "iteration": i,
                    "replay_buffer_size": self.replay.size,
                })

            t0 = time.perf_counter()
            if self.run.save_anyway:
                logger.warning("save_anyway=True, accepting new model unconditionally.")
                self._accept_model(i)
            else:
                total_decisive = pwins + nwins
                if total_decisive == 0 or float(nwins) / total_decisive < self.run.update_threshold:
                    logger.info("REJECTING NEW MODEL")
                    self.nnet.load_checkpoint(folder=self.run.checkpoint, filename="temp.pth.tar")
                else:
                    logger.info("ACCEPTING NEW MODEL")
                    self._accept_model(i)
            stats.accept_s = time.perf_counter() - t0

            stats.total_s = time.perf_counter() - iter_t0
            if self.run.profile:
                profile_rows.append(stats)
                logger.info("\n{}\n", stats.to_log_lines())

        if self.run.profile and profile_rows:
            summary_path = Path(self.run.profile_dir) / self.run.profile_summary_json
            write_iter_summaries_json(str(summary_path), profile_rows)
            logger.info("Wrote aggregated phase timings to {}", summary_path.resolve())

    def _accept_model(self, iteration: int) -> None:
        self.nnet.save_checkpoint(folder=self.run.checkpoint, filename=f"checkpoint_{iteration}.pth.tar")
        self.nnet.save_checkpoint(folder=self.run.checkpoint, filename="best.pth.tar")
