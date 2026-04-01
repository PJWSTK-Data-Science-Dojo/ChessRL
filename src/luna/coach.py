"""EfficientZeroV2 Coach: self-play -> replay buffer -> unroll training loop."""

from __future__ import annotations

import logging

import numpy as np
from tqdm import tqdm

from .game.arena import Arena
from .game.luna_game import ChessGame
from .mcts import MCTS
from .network import LunaNetwork
from .replay_buffer import PrioritizedReplayBuffer, Trajectory
from .utils import dotdict

log = logging.getLogger(__name__)


class Coach:
    """Orchestrates EZV2 self-play data collection, replay storage, and learning."""

    game: ChessGame
    nnet: LunaNetwork
    pnet: LunaNetwork
    args: dotdict
    replay: PrioritizedReplayBuffer

    def __init__(self, game: ChessGame, nnet: LunaNetwork, args: dotdict) -> None:
        self.game = game
        self.nnet = nnet
        self.pnet = nnet.__class__(game)
        self.args = args
        self.replay = PrioritizedReplayBuffer(
            capacity=args.get("replay_capacity", 100_000),
            alpha=args.get("per_alpha", 0.6),
            beta=args.get("per_beta", 0.4),
        )

    # ------------------------------------------------------------------
    # Self-play
    # ------------------------------------------------------------------
    def execute_episode(self) -> Trajectory:
        """Run one self-play game using latent MCTS, collecting a full trajectory."""
        mcts = MCTS(self.game, self.nnet, self.args)

        observations: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        root_policies: list[np.ndarray] = []
        root_values: list[float] = []
        valids_list: list[np.ndarray] = []

        board = self.game.get_init_board()
        current_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, current_player)
            temp = 1.0 if episode_step < self.args.tempThreshold else 0.0

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
            if r != 0:
                game_len = len(actions)
                for i in range(game_len):
                    sign = 1.0 if (game_len - 1 - i) % 2 == 0 else -1.0
                    rewards.append(sign * r)

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
        train_steps_per_iter = self.args.get("train_steps_per_iter", 200)

        for i in range(1, self.args.numIters + 1):
            log.info("Starting Iter #%d ...", i)

            for ep in tqdm(range(self.args.numEps), desc="Self Play"):
                trajectory = self.execute_episode()
                self.replay.save_trajectory(trajectory)

            if self.replay.size < self.args.batch_size:
                log.warning("Replay buffer too small (%d), skipping training.", self.replay.size)
                continue

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")

            log.info("Training from replay buffer (%d positions) ...", self.replay.size)
            loss_info = self.nnet.train_ezv2(self.replay, steps=train_steps_per_iter)
            log.info("Training done: %s", loss_info)

            log.info("PITTING AGAINST PREVIOUS VERSION")
            pmcts = MCTS(self.game, self.pnet, self.args)
            nmcts = MCTS(self.game, self.nnet, self.args)

            arena = Arena(
                lambda x, _pm=pmcts: np.argmax(_pm.get_action_prob(x, temp=0)),
                lambda x, _nm=nmcts: np.argmax(_nm.get_action_prob(x, temp=0)),
                self.game,
            )

            num_arena_pits = max(1, int(self.args.arenaCompare / 2))
            nwins = 0
            pwins = 0
            draws = 0

            for _ in tqdm(range(num_arena_pits), desc="Arena (1)"):
                result = arena.play_game(verbose=False)
                if result == 1:
                    pwins += 1
                elif result == -1:
                    nwins += 1
                else:
                    draws += 1

            arena_rev = Arena(
                lambda x, _nm=nmcts: np.argmax(_nm.get_action_prob(x, temp=0)),
                lambda x, _pm=pmcts: np.argmax(_pm.get_action_prob(x, temp=0)),
                self.game,
            )

            for _ in tqdm(range(num_arena_pits), desc="Arena (2)"):
                result = arena_rev.play_game(verbose=False)
                if result == 1:
                    nwins += 1
                elif result == -1:
                    pwins += 1
                else:
                    draws += 1

            log.info("NEW/PREV WINS: %d / %d ; DRAWS: %d", nwins, pwins, draws)

            if self.args.get("save_anyway", False):
                log.warning("save_anyway=True, accepting new model unconditionally.")
                self._accept_model(i)
            else:
                total_decisive = pwins + nwins
                if total_decisive == 0 or float(nwins) / total_decisive < self.args.updateThreshold:
                    log.info("REJECTING NEW MODEL")
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")
                else:
                    log.info("ACCEPTING NEW MODEL")
                    self._accept_model(i)

    def _accept_model(self, iteration: int) -> None:
        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=f"checkpoint_{iteration}.pth.tar")
        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="best.pth.tar")

