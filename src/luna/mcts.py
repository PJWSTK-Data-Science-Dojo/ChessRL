"""EfficientZeroV2 MCTS -- latent-space search with dynamics/prediction networks.

Falls back to real-board search when no latent inference is available
(e.g. during arena evaluation with the old Coach path).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import chess
import numpy as np
import torch
import torch.nn.functional as F

from .game.luna_game import ChessGame
from .utils import dotdict

if TYPE_CHECKING:
    from .network import LunaNetwork

EPS = 1e-8
log = logging.getLogger(__name__)


class _LatentNode:
    """A node in the latent MCTS tree."""

    __slots__ = (
        "latent",
        "prior",
        "value_sum",
        "visit_count",
        "reward",
        "children",
        "expanded",
    )

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.value_sum = 0.0
        self.visit_count = 0
        self.reward = 0.0
        self.latent: torch.Tensor | None = None
        self.children: dict[int, _LatentNode] = {}
        self.expanded = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """Hybrid MCTS supporting both latent-space (EZV2) and real-board (AlphaZero) search."""

    game: ChessGame
    args: dotdict

    def __init__(self, game: ChessGame, nnet: LunaNetwork, args: dotdict) -> None:
        self.game = game
        self.nnet = nnet
        self.args = args

        self.Qsa: dict[tuple[str, int], float] = {}
        self.Nsa: dict[tuple[str, int], int] = {}
        self.Ns: dict[str, int] = {}
        self.Ps: dict[str, np.ndarray] = {}
        self.Es: dict[str, float] = {}
        self.Vs: dict[str, np.ndarray] = {}
        self.Va: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_action_prob(self, canonical_board: chess.Board, temp: float = 1) -> list[float]:
        for _ in range(self.args.numMCTSSims):
            self.search(canonical_board)

        s = self.game.string_representation(canonical_board)
        action_size = self.game.get_action_size()
        counts = [self.Nsa.get((s, a), 0) for a in range(action_size)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0.0] * len(counts)
            probs[bestA] = 1.0
            return probs

        counts_arr = np.array(counts, dtype=np.float64)
        counts_arr = counts_arr ** (1.0 / temp)
        total = counts_arr.sum()
        if total > 0:
            probs = (counts_arr / total).tolist()
        else:
            probs = [0.0] * len(counts)
        return probs

    def get_action_prob_and_value(
        self, canonical_board: chess.Board, temp: float = 1
    ) -> tuple[list[float], float]:
        """Run MCTS and return (policy, root_value)."""
        for _ in range(self.args.numMCTSSims):
            self.search(canonical_board)

        s = self.game.string_representation(canonical_board)
        action_size = self.game.get_action_size()
        counts = [self.Nsa.get((s, a), 0) for a in range(action_size)]

        root_q_values = []
        for a in range(action_size):
            if (s, a) in self.Qsa and self.Nsa.get((s, a), 0) > 0:
                root_q_values.append(self.Qsa[(s, a)] * self.Nsa[(s, a)])
            else:
                root_q_values.append(0.0)
        total_visits = sum(counts)
        root_value = sum(root_q_values) / max(total_visits, 1)

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0.0] * len(counts)
            probs[bestA] = 1.0
            return probs, root_value

        counts_arr = np.array(counts, dtype=np.float64)
        counts_arr = counts_arr ** (1.0 / temp)
        total = counts_arr.sum()
        probs = (counts_arr / total).tolist() if total > 0 else [0.0] * len(counts)
        return probs, root_value

    # ------------------------------------------------------------------
    # Latent-space MCTS (primary EZV2 path)
    # ------------------------------------------------------------------
    def search_latent(self, canonical_board: chess.Board, num_sims: int | None = None) -> tuple[list[float], float]:
        """Full latent-space MCTS. Returns (policy, root_value)."""
        if num_sims is None:
            num_sims = self.args.numMCTSSims

        valids = self.game.get_valid_moves(canonical_board, 1)
        obs = self.game.to_array(canonical_board)
        pi_np, root_v, latent = self.nnet.predict_with_latent(obs, valids)

        valid_indices = np.flatnonzero(valids)
        root = _LatentNode(prior=0.0)
        root.latent = latent
        root.expanded = True

        if self.args.dir_noise:
            noise = np.random.dirichlet([self.args.dir_alpha] * len(valid_indices))
            for i, a in enumerate(valid_indices):
                blended_prior = 0.75 * pi_np[a] + 0.25 * noise[i]
                root.children[int(a)] = _LatentNode(prior=float(blended_prior))
        else:
            for a in valid_indices:
                root.children[int(a)] = _LatentNode(prior=float(pi_np[a]))

        for _ in range(num_sims):
            self._latent_simulate(root)

        action_size = self.game.get_action_size()
        counts = np.zeros(action_size, dtype=np.float64)
        q_sum = np.zeros(action_size, dtype=np.float64)
        for a, child in root.children.items():
            counts[a] = child.visit_count
            q_sum[a] = child.value_sum

        total_visits = counts.sum()
        root_value = float(q_sum.sum() / max(total_visits, 1))

        temp = 1.0
        if counts.sum() > 0:
            counts_temp = counts ** (1.0 / max(temp, 1e-8))
            probs = (counts_temp / counts_temp.sum()).tolist()
        else:
            probs = [0.0] * action_size

        return probs, root_value

    def _latent_simulate(self, node: _LatentNode) -> float:
        if not node.expanded:
            return 0.0

        best_action = -1
        best_ucb = -float("inf")
        total_visits = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(total_visits + EPS)

        for action, child in node.children.items():
            if child.visit_count == 0:
                ucb = self.args.cpuct * child.prior * sqrt_total
            else:
                ucb = child.value() + self.args.cpuct * child.prior * sqrt_total / (1 + child.visit_count)
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        child = node.children[best_action]
        discount = float(self.args.get("discount", 0.997))

        if not child.expanded and node.latent is not None:
            pi_np, value, reward, next_latent = self.nnet.recurrent_predict(node.latent, best_action)
            child.latent = next_latent
            child.reward = reward
            child.expanded = True

            action_size = self.game.get_action_size()
            for a in range(action_size):
                if pi_np[a] > 1e-6:
                    child.children[a] = _LatentNode(prior=float(pi_np[a]))

            q = child.reward - discount * value
            child.visit_count += 1
            child.value_sum += q
            return -q

        value = -self._latent_simulate(child)
        q = child.reward + discount * value
        child.visit_count += 1
        child.value_sum += q
        return -q

    # ------------------------------------------------------------------
    # Real-board MCTS (fallback / arena evaluation)
    # ------------------------------------------------------------------
    def search(self, canonical_board: chess.Board) -> float:
        s = self.game.string_representation(canonical_board)

        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonical_board, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            valids = self.game.get_valid_moves(canonical_board, 1)
            self.Ps[s], v = self.nnet.predict((self.game.to_array(canonical_board), valids))
            if self.args.dir_noise:
                self.Ps[s] = 0.75 * self.Ps[s] + 0.25 * np.random.dirichlet(
                    [self.args.dir_alpha] * len(self.Ps[s])
                )
            self.Ps[s] = self.Ps[s] * valids
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Va[s] = np.flatnonzero(valids)
            self.Ns[s] = 0
            return -v

        valid_actions = self.Va[s]
        cur_best = -float("inf")
        best_act = -1
        sqrt_Ns = math.sqrt(self.Ns[s] + EPS)

        for a in valid_actions:
            a = int(a)
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * sqrt_Ns / (1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * float(self.Ps[s][a]) * sqrt_Ns
            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_s, next_player = self.game.get_next_state(canonical_board, 1, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

