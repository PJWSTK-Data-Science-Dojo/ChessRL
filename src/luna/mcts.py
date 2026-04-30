"""EfficientZeroV2 MCTS -- latent-space search with dynamics/prediction networks.

Supports both single-game search (original) and batched parallel search
across N games for dramatically higher GPU utilisation during self-play.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import chess
import numpy as np
import torch

from .config import MCTSParams
from .game.chess_game import ChessGame
from .profiling import SelfPlayMCTSTimings

if TYPE_CHECKING:
    from .network import LunaNetwork

try:
    from numba import njit

    @njit(cache=True)
    def _puct_argmax_numba(
        cpuct: float,
        sqrt_total: float,
        actions: np.ndarray,
        priors: np.ndarray,
        visits: np.ndarray,
        vsum: np.ndarray,
    ) -> int:
        n = visits.shape[0]
        ucb = np.empty(n, dtype=np.float64)
        for i in range(n):
            vi = visits[i]
            if vi == 0.0:
                ucb[i] = cpuct * priors[i] * sqrt_total
            else:
                q = vsum[i] / vi
                ucb[i] = q + cpuct * priors[i] * sqrt_total / (1.0 + vi)
        bi = 0
        bv = ucb[0]
        for i in range(1, n):
            if ucb[i] > bv:
                bv = ucb[i]
                bi = i
        return int(actions[bi])

    _NUMBA_PUCT = True
except ImportError:
    _NUMBA_PUCT = False

EPS = 1e-8


def _puct_best_action(cpuct: float, node: _LatentNode) -> int:
    """Pick child with highest PUCT score (vectorized over legal children).

    Matches the tie-breaking of the original per-child Python loop: first child
    among equals wins (dict / array insertion order).
    """
    ch = node.children
    n = len(ch)
    if n == 0:
        return -1
    if n == 1:
        return int(next(iter(ch.keys())))

    sqrt_total = math.sqrt(node.total_child_visits + EPS)
    actions = np.empty(n, dtype=np.int32)
    priors = np.empty(n, dtype=np.float64)
    visits = np.empty(n, dtype=np.float64)
    vsum = np.empty(n, dtype=np.float64)
    for i, (a, child) in enumerate(ch.items()):
        actions[i] = int(a)
        priors[i] = child.prior
        visits[i] = child.visit_count
        vsum[i] = child.value_sum

    if _NUMBA_PUCT and n >= 4:
        return _puct_argmax_numba(float(cpuct), sqrt_total, actions, priors, visits, vsum)

    q = np.divide(vsum, visits, out=np.zeros(n, dtype=np.float64), where=visits > 0)
    ucb0 = cpuct * priors * sqrt_total
    ucb1 = q + cpuct * priors * sqrt_total / (1.0 + visits)
    ucb = np.where(visits == 0, ucb0, ucb1)
    return int(actions[int(np.argmax(ucb))])


class _LatentNode:
    """A node in the latent MCTS tree."""

    __slots__ = (
        "children",
        "expanded",
        "latent",
        "prior",
        "reward",
        "total_child_visits",
        "value_sum",
        "visit_count",
    )

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.value_sum = 0.0
        self.visit_count = 0
        self.total_child_visits = 0
        self.reward = 0.0
        self.latent: torch.Tensor | None = None
        self.children: dict[int, _LatentNode] = {}
        self.expanded = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """Latent-space MCTS (EfficientZeroV2)."""

    game: ChessGame
    params: MCTSParams

    def __init__(self, game: ChessGame, nnet: LunaNetwork, params: MCTSParams) -> None:
        self.game = game
        self.nnet = nnet
        self.params = params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_action_prob(self, canonical_board: chess.Board, temp: float = 1) -> list[float]:
        probs, _ = self.search_latent(canonical_board, temp=temp)
        return probs

    def search_latent(
        self, canonical_board: chess.Board, num_sims: int | None = None, temp: float = 1.0
    ) -> tuple[list[float], float]:
        """Full latent-space MCTS. Returns (policy, root_value)."""
        if num_sims is None:
            num_sims = self.params.num_mcts_sims

        valids = self.game.get_valid_moves(canonical_board, 1)
        obs = self.game.to_array(canonical_board)
        pi_np, _root_v, latent = self.nnet.predict_with_latent(obs, valids)

        valid_indices = np.flatnonzero(valids)
        action_size = self.game.get_action_size()

        if len(valid_indices) == 0:
            return [0.0] * action_size, 0.0

        root = _LatentNode(prior=0.0)
        root.latent = latent
        root.expanded = True

        if self.params.dir_noise:
            noise = np.random.dirichlet([self.params.dir_alpha] * len(valid_indices))
            for i, a in enumerate(valid_indices):
                blended_prior = 0.75 * pi_np[a] + 0.25 * noise[i]
                root.children[int(a)] = _LatentNode(prior=float(blended_prior))
        else:
            for a in valid_indices:
                root.children[int(a)] = _LatentNode(prior=float(pi_np[a]))

        cp = self.params.cpuct
        for _ in range(num_sims):
            self._latent_simulate(root, cp)

        counts = np.zeros(action_size, dtype=np.float64)
        q_sum = np.zeros(action_size, dtype=np.float64)
        for action_key, child in root.children.items():
            idx: int = int(action_key)
            counts[idx] = float(child.visit_count)
            q_sum[idx] = child.value_sum

        total_visits = counts.sum()
        root_value = float(q_sum.sum() / max(total_visits, 1))

        if counts.sum() > 0:
            if temp == 0:
                best_mask = counts == counts.max()
                best_indices = np.flatnonzero(best_mask)
                best_a = int(np.random.choice(best_indices))
                probs = [0.0] * action_size
                probs[best_a] = 1.0
            else:
                counts_temp = counts ** (1.0 / max(temp, 1e-8))
                probs = (counts_temp / counts_temp.sum()).tolist()
        else:
            probs = [0.0] * action_size

        return probs, root_value

    def _latent_simulate(self, node: _LatentNode, cpuct: float) -> float:
        if not node.expanded or not node.children:
            return 0.0

        best_action = _puct_best_action(cpuct, node)
        child = node.children[best_action]
        discount = float(self.params.discount)

        if not child.expanded and node.latent is not None:
            pi_np, value, reward, next_latent = self.nnet.recurrent_predict(node.latent, best_action)
            child.latent = next_latent
            child.reward = reward
            child.expanded = True

            child_indices = np.flatnonzero(pi_np > 1e-6)
            for a in child_indices:
                child.children[int(a)] = _LatentNode(prior=float(pi_np[a]))

            q = child.reward + discount * (-value)
            child.visit_count += 1
            child.value_sum += q
            node.total_child_visits += 1
            return -q

        leaf_value = -self._latent_simulate(child, cpuct)
        q = child.reward + discount * leaf_value
        child.visit_count += 1
        child.value_sum += q
        node.total_child_visits += 1
        return -q


# ======================================================================
# Batched parallel MCTS for self-play
# ======================================================================

class _PendingExpansion:
    """Tracks a leaf that needs NN evaluation before backprop."""

    __slots__ = ("ancestors", "child")

    def __init__(self, ancestors: list[_LatentNode], child: _LatentNode) -> None:
        """ancestors = [root, ..., parent] along the path to the unexpanded *child*."""
        self.ancestors = ancestors
        self.child = child


def _backup_latent_path(ancestors: list[_LatentNode], leaf: _LatentNode, q_leaf: float, discount: float) -> None:
    """Match :meth:`MCTS._latent_simulate` backup along the full path (leaf first, then up)."""
    leaf.visit_count += 1
    leaf.value_sum += q_leaf
    ancestors[-1].total_child_visits += 1
    q = q_leaf
    for j in range(len(ancestors) - 1, 0, -1):
        child = ancestors[j]
        parent = ancestors[j - 1]
        q = child.reward + discount * q
        child.visit_count += 1
        child.value_sum += q
        parent.total_child_visits += 1


class BatchedMCTS:
    """Run MCTS for N games in parallel, batching all leaf expansions into one GPU call.

    Each simulation step:
      1. For each game, select a leaf via tree traversal (CPU)
      2. Batch all pending leaf latents into one recurrent_inference call (GPU)
      3. Backpropagate results into each tree (CPU)

    This turns N*sims individual GPU calls into N*sims / batch_factor calls,
    massively improving GPU utilisation.
    """

    def __init__(
        self,
        game: ChessGame,
        nnet: LunaNetwork,
        params: MCTSParams,
        timings: SelfPlayMCTSTimings | None = None,
    ) -> None:
        self.game = game
        self.nnet = nnet
        self.params = params
        self._timings = timings
        self._pending: list[_PendingExpansion] = []
        self._parent_latents: list[torch.Tensor] = []
        self._pending_actions: list[int] = []

    def search_batch(
        self,
        canonical_boards: list[chess.Board],
        num_sims: int | None = None,
        temp: float = 1.0,
    ) -> list[tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
        """Run batched latent MCTS for multiple positions.

        Returns one tuple per board: ``(policy, root_value, obs, valid)``. *policy* is a
        float32 vector summing to 1 (``numpy.ndarray``, shape ``(action_size,)``). *obs* and *valid*
        are copies of the rows used for root inference.
        """
        if num_sims is None:
            num_sims = self.params.num_mcts_sims

        N = len(canonical_boards)
        if N == 0:
            return []
        action_size = self.game.get_action_size()
        discount = float(self.params.discount)
        cpuct = self.params.cpuct
        tm = self._timings

        if tm is not None:
            tm.search_batch_calls += 1
            t0 = time.perf_counter()

        sample_obs = self.game.to_array(canonical_boards[0])
        obs_batch = np.empty((N, *sample_obs.shape), dtype=np.float32)
        valid_batch = np.empty((N, action_size), dtype=np.float32)
        for i, b in enumerate(canonical_boards):
            obs_batch[i] = self.game.to_array(b)
            valid_batch[i] = self.game.get_valid_moves(b, 1)

        if tm is not None:
            tm.encode_s += time.perf_counter() - t0
            t0 = time.perf_counter()

        policies_np, _values_np, latents = self.nnet.batched_initial_inference(obs_batch, valid_batch)

        if tm is not None:
            tm.initial_inf_s += time.perf_counter() - t0

        roots: list[_LatentNode] = []
        for i in range(N):
            root = _LatentNode(prior=0.0)
            root.latent = latents[i : i + 1]
            root.expanded = True

            valid_indices = np.flatnonzero(valid_batch[i])
            pi = policies_np[i]

            if self.params.dir_noise and len(valid_indices) > 0:
                noise = np.random.dirichlet([self.params.dir_alpha] * len(valid_indices))
                for j, a in enumerate(valid_indices):
                    root.children[int(a)] = _LatentNode(prior=float(0.75 * pi[a] + 0.25 * noise[j]))
            else:
                for a in valid_indices:
                    root.children[int(a)] = _LatentNode(prior=float(pi[a]))

            roots.append(root)

        for _ in range(num_sims):
            if tm is not None:
                t_sel = time.perf_counter()

            pending = self._pending
            parent_latents = self._parent_latents
            pending_actions = self._pending_actions
            pending.clear()
            parent_latents.clear()
            pending_actions.clear()

            for root in roots:
                if not root.children:
                    continue
                result = self._select_leaf(root, cpuct)
                if result is not None:
                    ancestors, child, action = result
                    pending.append(_PendingExpansion(ancestors, child))
                    parent_latents.append(ancestors[-1].latent)  # type: ignore[arg-type]
                    pending_actions.append(action)

            if tm is not None:
                tm.selection_s += time.perf_counter() - t_sel

            if not pending:
                continue

            if tm is not None:
                t_rec = time.perf_counter()

            batched_latent = torch.cat(parent_latents, dim=0)
            rb = self.nnet.batched_recurrent_inference(
                batched_latent,
                pending_actions,
                policy_topk=self.params.recurrent_policy_topk,
            )

            if tm is not None:
                tm.recurrent_inf_s += time.perf_counter() - t_rec
                t_bu = time.perf_counter()

            v_f = np.asarray(rb.values, dtype=np.float64)
            r_f = np.asarray(rb.rewards, dtype=np.float64)
            next_latents = rb.next_latent
            q_all = r_f + discount * (-v_f)

            if rb.policy_full is not None:
                pi_batch = rb.policy_full
                for j, pe in enumerate(pending):
                    child = pe.child
                    child.latent = next_latents[j : j + 1]
                    child.reward = float(r_f[j])
                    child.expanded = True

                    pi_row = pi_batch[j]
                    child_indices = np.flatnonzero(pi_row > 1e-6)
                    for a in child_indices:
                        child.children[int(a)] = _LatentNode(prior=float(pi_row[a]))

                    _backup_latent_path(pe.ancestors, child, float(q_all[j]), discount)
            else:
                idx_bt = rb.topk_indices
                prob_bt = rb.topk_probs
                assert idx_bt is not None and prob_bt is not None
                k_w = idx_bt.shape[1]
                for j, pe in enumerate(pending):
                    child = pe.child
                    child.latent = next_latents[j : j + 1]
                    child.reward = float(r_f[j])
                    child.expanded = True

                    for t in range(k_w):
                        a = int(idx_bt[j, t])
                        p = float(prob_bt[j, t])
                        if p > 1e-6:
                            child.children[a] = _LatentNode(prior=p)

                    _backup_latent_path(pe.ancestors, child, float(q_all[j]), discount)

            if tm is not None:
                tm.expand_backup_s += time.perf_counter() - t_bu

        if tm is not None:
            t_fin = time.perf_counter()

        results: list[tuple[np.ndarray, float, np.ndarray, np.ndarray]] = []
        for i, root in enumerate(roots):
            counts = np.zeros(action_size, dtype=np.float64)
            q_sum = np.zeros(action_size, dtype=np.float64)
            for ak, ch in root.children.items():
                counts[int(ak)] = float(ch.visit_count)
                q_sum[int(ak)] = ch.value_sum

            total_visits = counts.sum()
            root_value = float(q_sum.sum() / max(total_visits, 1))

            if total_visits > 0:
                if temp == 0:
                    best_mask = counts == counts.max()
                    best_indices = np.flatnonzero(best_mask)
                    best_a = int(np.random.choice(best_indices))
                    probs_arr = np.zeros(action_size, dtype=np.float32)
                    probs_arr[best_a] = 1.0
                else:
                    counts_temp = counts ** (1.0 / max(temp, 1e-8))
                    probs_arr = (counts_temp / counts_temp.sum()).astype(np.float32, copy=False)
            else:
                probs_arr = np.zeros(action_size, dtype=np.float32)

            results.append((probs_arr, root_value, obs_batch[i].copy(), valid_batch[i].copy()))

        if tm is not None:
            tm.finalize_s += time.perf_counter() - t_fin

        return results

    def _select_leaf(
        self, root: _LatentNode, cpuct: float
    ) -> tuple[list[_LatentNode], _LatentNode, int] | None:
        """Walk from *root* to an unexpanded leaf. Returns (ancestors, leaf, action) or None.

        *ancestors* is ``[root, ..., parent]`` such that *leaf* is a child of ``ancestors[-1]``.
        """
        if not root.expanded or not root.children:
            return None

        ancestors: list[_LatentNode] = [root]
        current = root
        while True:
            best_action = _puct_best_action(cpuct, current)
            child = current.children[best_action]

            if not child.expanded and current.latent is not None:
                return ancestors, child, best_action

            if not child.expanded or not child.children:
                return None

            ancestors.append(child)
            current = child
