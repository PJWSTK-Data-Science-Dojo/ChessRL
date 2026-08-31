"""EfficientZeroV2 latent search with Gumbel MuZero and classic PUCT.

The default root policy uses Gumbel top-m sampling without replacement and
Sequential Halving. Interior nodes use the deterministic Full Gumbel MuZero
policy-improvement rule. Classic MuZero PUCT remains available as an option.
Both algorithms share the same single-game and batched tree semantics.
"""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable, Collection, Sequence
from typing import TYPE_CHECKING, cast

import chess
import numpy as np
import torch

from luna.config import MCTSParams
from luna.game.chess_game import ChessGame, player_from_turn
from luna.profiling import SelfPlayMCTSTimings

if TYPE_CHECKING:
    from luna.network import LunaNetwork

_PuctArgmax = Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], int]


def _puct_argmax_impl(
    exploration_scale: float,
    actions: np.ndarray,
    priors: np.ndarray,
    visits: np.ndarray,
    vsum: np.ndarray,
) -> int:
    n = visits.shape[0]
    ucb = np.empty(n, dtype=np.float64)
    for i in range(n):
        visit_count = visits[i]
        if visit_count == 0.0:
            ucb[i] = exploration_scale * priors[i]
        else:
            q_value = vsum[i] / visit_count
            ucb[i] = q_value + exploration_scale * priors[i] / (1.0 + visit_count)
    best_index = 0
    best_value = ucb[0]
    for i in range(1, n):
        if ucb[i] > best_value:
            best_value = ucb[i]
            best_index = i
    return int(actions[best_index])


_puct_argmax_numba: _PuctArgmax = _puct_argmax_impl
try:
    numba_module = importlib.import_module("numba")
    _puct_argmax_numba = cast(_PuctArgmax, numba_module.njit(cache=True)(_puct_argmax_impl))
    _NUMBA_PUCT = True
except (AttributeError, ImportError):
    # Numba is an optional acceleration. Some valid environments expose an
    # incompatible coverage API during Numba import; search must still work.
    _NUMBA_PUCT = False

EPS = 1e-8


def _puct_best_action(cpuct: float, pb_c_base: float, node: _LatentNode) -> int:
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
    exploration_scale = (math.log((node.total_child_visits + pb_c_base + 1.0) / pb_c_base) + cpuct) * sqrt_total
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
        return int(_puct_argmax_numba(exploration_scale, actions, priors, visits, vsum))

    q = np.divide(vsum, visits, out=np.zeros(n, dtype=np.float64), where=visits > 0)
    ucb0 = exploration_scale * priors
    ucb1 = q + exploration_scale * priors / (1.0 + visits)
    ucb = np.where(visits == 0, ucb0, ucb1)
    return int(actions[int(np.argmax(ucb))])


class _LatentNode:
    """Search node whose edge reward is represented from its parent perspective."""

    __slots__ = (
        "board",
        "children",
        "expanded",
        "latent",
        "prior",
        "raw_value",
        "reward",
        "terminal",
        "total_child_visits",
        "value_sum",
        "visit_count",
    )

    def __init__(self, prior: float, board: chess.Board | None = None) -> None:
        self.prior = prior
        self.value_sum = 0.0
        self.visit_count = 0
        self.total_child_visits = 0
        self.reward = 0.0
        self.raw_value = 0.0
        self.terminal = False
        self.board = board
        self.latent: torch.Tensor | None = None
        self.children: dict[int, _LatentNode] = {}
        self.expanded = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _validate_search(params: MCTSParams, num_sims: int, temp: float) -> None:
    """Fail early on search settings that cannot define a valid policy."""
    if num_sims <= 0:
        raise ValueError("num_sims must be positive")
    if not math.isfinite(temp) or temp < 0:
        raise ValueError("temp must be finite and non-negative")
    if params.search_mode not in {"gumbel", "puct"}:
        raise ValueError(f"unknown MCTS search_mode: {params.search_mode!r}")
    if params.pb_c_base <= 0:
        raise ValueError("pb_c_base must be positive")
    if params.search_mode == "gumbel":
        if params.gumbel_max_considered_actions <= 0:
            raise ValueError("gumbel_max_considered_actions must be positive")
        if params.gumbel_scale < 0:
            raise ValueError("gumbel_scale must be non-negative")
        if params.gumbel_value_scale < 0:
            raise ValueError("gumbel_value_scale must be non-negative")
        if params.gumbel_maxvisit_init < 0:
            raise ValueError("gumbel_maxvisit_init must be non-negative")


def _get_sequence_of_considered_visits(max_num_considered_actions: int, num_simulations: int) -> tuple[int, ...]:
    """Return the exact Sequential Halving visit schedule used by Gumbel MuZero.

    At simulation ``i``, only root actions with the returned visit count are
    eligible. The first round therefore samples top-m actions without
    replacement; later rounds repeatedly halve the surviving set.
    """
    if max_num_considered_actions <= 0:
        raise ValueError("max_num_considered_actions must be positive")
    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive")
    if max_num_considered_actions == 1:
        return tuple(range(num_simulations))

    log2max = math.ceil(math.log2(max_num_considered_actions))
    sequence: list[int] = []
    visits = [0] * max_num_considered_actions
    num_considered = max_num_considered_actions
    while len(sequence) < num_simulations:
        extra_visits = max(1, int(num_simulations / (log2max * num_considered)))
        for _ in range(extra_visits):
            sequence.extend(visits[:num_considered])
            for i in range(num_considered):
                visits[i] += 1
        num_considered = max(2, num_considered // 2)
    return tuple(sequence[:num_simulations])


def _child_statistics(
    node: _LatentNode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return actions, normalized priors, visits, and mean Q in child order."""
    size = len(node.children)
    actions = np.empty(size, dtype=np.int32)
    priors = np.empty(size, dtype=np.float64)
    visits = np.empty(size, dtype=np.int64)
    qvalues = np.zeros(size, dtype=np.float64)
    for i, (action, child) in enumerate(node.children.items()):
        actions[i] = int(action)
        priors[i] = max(0.0, float(child.prior))
        visits[i] = int(child.visit_count)
        if child.visit_count > 0:
            qvalues[i] = child.value_sum / child.visit_count

    prior_sum = float(priors.sum())
    if size and prior_sum > 0.0:
        priors /= prior_sum
    elif size:
        priors.fill(1.0 / size)
    return actions, priors, visits, qvalues


def _completed_qvalues(node: _LatentNode, params: MCTSParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Complete and transform child Q values as specified by Gumbel MuZero.

    Unvisited actions receive the prior-weighted mixed value. Values are then
    min-max normalized and scaled by ``(maxvisit_init + max_visits) * value_scale``.
    """
    actions, priors, visits, qvalues = _child_statistics(node)
    if not len(actions):
        return actions, priors, visits, qvalues

    visited = visits > 0
    sum_visits = int(visits.sum())
    if np.any(visited):
        safe_priors = np.maximum(priors, np.finfo(np.float64).tiny)
        visited_prior_mass = float(safe_priors[visited].sum())
        weighted_q = float(np.dot(safe_priors[visited], qvalues[visited])) / visited_prior_mass
    else:
        weighted_q = float(node.raw_value)
    mixed_value = (float(node.raw_value) + sum_visits * weighted_q) / (sum_visits + 1)
    completed = np.where(visited, qvalues, mixed_value)

    min_q = float(completed.min())
    max_q = float(completed.max())
    normalized = (completed - min_q) / max(max_q - min_q, EPS)
    visit_scale = params.gumbel_maxvisit_init + float(visits.max())
    transformed = visit_scale * params.gumbel_value_scale * normalized
    return actions, priors, visits, transformed


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    probabilities = np.exp(shifted)
    return probabilities / float(probabilities.sum())


def _visit_count_policy(counts: np.ndarray, temp: float) -> np.ndarray:
    """Apply temperature to positive visit counts without overflow."""
    positive = counts > 0
    if not np.any(positive):
        raise ValueError("visit-count policy requires at least one positive count")
    log_counts = np.log(counts[positive])
    with np.errstate(over="ignore"):
        logits = (log_counts - float(log_counts.max())) / temp
    probabilities = np.exp(logits)
    policy = np.zeros_like(counts, dtype=np.float64)
    policy[positive] = probabilities / float(probabilities.sum())
    return policy


def _gumbel_interior_best_action(node: _LatentNode, params: MCTSParams) -> int:
    """Deterministically match visits to the current improved policy."""
    actions, priors, visits, completed_q = _completed_qvalues(node, params)
    if len(actions) == 0:
        return -1
    if len(actions) == 1:
        return int(actions[0])
    prior_logits = np.log(np.maximum(priors, np.finfo(np.float64).tiny))
    improved_policy = _softmax(prior_logits + completed_q)
    scores = improved_policy - visits / (1.0 + float(visits.sum()))
    return int(actions[int(np.argmax(scores))])


def _interior_best_action(node: _LatentNode, params: MCTSParams) -> int:
    if params.search_mode == "gumbel":
        return _gumbel_interior_best_action(node, params)
    return _puct_best_action(params.cpuct, params.pb_c_base, node)


def _gumbel_improved_policy(root: _LatentNode, action_size: int, params: MCTSParams) -> np.ndarray:
    """Produce the non-degenerate completed-Q policy target from a search tree."""
    policy = np.zeros(action_size, dtype=np.float32)
    actions, priors, _visits, completed_q = _completed_qvalues(root, params)
    if len(actions) == 0:
        return policy
    prior_logits = np.log(np.maximum(priors, np.finfo(np.float64).tiny))
    policy[actions] = _softmax(prior_logits + completed_q).astype(np.float32, copy=False)
    return policy


class _GumbelRootState:
    """Per-root Gumbel sample and Sequential Halving budget schedule."""

    __slots__ = ("actions", "gumbel", "params", "prior_logits", "schedule")

    def __init__(
        self,
        root: _LatentNode,
        num_simulations: int,
        params: MCTSParams,
        *,
        add_noise: bool,
    ) -> None:
        actions, priors, _visits, _qvalues = _child_statistics(root)
        num_considered = min(params.gumbel_max_considered_actions, len(actions))
        if num_considered <= 0:
            raise ValueError("Gumbel root requires at least one legal action")
        self.actions = actions
        self.params = params
        self.prior_logits = np.log(np.maximum(priors, np.finfo(np.float64).tiny))
        if add_noise and params.gumbel_scale > 0.0:
            self.gumbel = params.gumbel_scale * np.random.gumbel(size=len(actions))
        else:
            self.gumbel = np.zeros(len(actions), dtype=np.float64)
        self.schedule = _get_sequence_of_considered_visits(num_considered, num_simulations)

    def select_action(self, root: _LatentNode) -> int:
        """Choose the next root action under the current halving round."""
        actions, _priors, visits, completed_q = _completed_qvalues(root, self.params)
        if not np.array_equal(actions, self.actions):
            raise RuntimeError("root children changed during Sequential Halving")
        simulation_index = int(visits.sum())
        if simulation_index >= len(self.schedule):
            raise RuntimeError("Sequential Halving simulation budget was exceeded")
        considered_visit = self.schedule[simulation_index]
        eligible = visits == considered_visit
        if not np.any(eligible):
            raise RuntimeError("Sequential Halving visit schedule is inconsistent with the tree")
        scores = np.maximum(-1e9, self.gumbel + self.prior_logits + completed_q)
        scores = np.where(eligible, scores, -np.inf)
        return int(actions[int(np.argmax(scores))])

    def proposed_action(self, root: _LatentNode) -> int:
        """Return the Gumbel MuZero action proposed after Sequential Halving."""
        actions, _priors, visits, completed_q = _completed_qvalues(root, self.params)
        if not np.array_equal(actions, self.actions):
            raise RuntimeError("root children changed during Sequential Halving")
        considered_visit = int(visits.max())
        eligible = visits == considered_visit
        scores = np.maximum(-1e9, self.gumbel + self.prior_logits + completed_q)
        scores = np.where(eligible, scores, -np.inf)
        return int(actions[int(np.argmax(scores))])


class MCTS:
    """Latent-space MCTS (EfficientZeroV2)."""

    game: ChessGame
    params: MCTSParams

    def __init__(self, game: ChessGame, nnet: LunaNetwork, params: MCTSParams) -> None:
        self.game = game
        self.nnet = nnet
        self.params = params
        self.last_action: int | None = None
        self.last_simulations = 0

    def get_action_prob(
        self,
        canonical_board: chess.Board,
        temp: float = 1,
        *,
        add_exploration_noise: bool | None = None,
    ) -> list[float]:
        probs, _ = self.search_latent(
            canonical_board,
            temp=temp,
            add_exploration_noise=add_exploration_noise,
        )
        return probs

    def search_latent(
        self,
        canonical_board: chess.Board,
        num_sims: int | None = None,
        temp: float = 1.0,
        *,
        add_exploration_noise: bool | None = None,
        should_stop: Callable[[], bool] | None = None,
        allowed_root_actions: Collection[int] | None = None,
    ) -> tuple[list[float], float]:
        """Run latent search and return ``(policy_target, root_value)``.

        For Gumbel search, ``last_action`` is the actual Sequential Halving
        proposal. At positive temperature the returned policy is the soft
        completed-Q training target; at zero temperature it is a one-hot view of
        that proposal. Exploration defaults to ``temp > 0`` but can be controlled
        independently when a caller needs both a target and deterministic action.
        ``allowed_root_actions`` intersects the legal root mask without affecting
        legal actions at recurrent nodes.
        """
        self.last_action = None
        self.last_simulations = 0
        if num_sims is None:
            num_sims = self.params.num_mcts_sims
        _validate_search(self.params, num_sims, temp)
        if add_exploration_noise is None:
            add_exploration_noise = temp > 0.0

        root_board = canonical_board.copy(stack=canonical_board.halfmove_clock)
        terminal_value = self.game.get_game_outcome(root_board, 1)
        if terminal_value is not None:
            return [0.0] * self.game.get_action_size(), float(terminal_value)

        action_size = self.game.get_action_size()
        valids = self.game.get_valid_moves(root_board, 1)
        if allowed_root_actions is not None:
            root_mask = np.zeros(action_size, dtype=valids.dtype)
            for action in allowed_root_actions:
                if not 0 <= action < action_size:
                    raise ValueError(f"Root action must be in [0, {action_size}), got {action}")
                root_mask[action] = 1
            valids *= root_mask
        valid_indices = np.flatnonzero(valids)
        if len(valid_indices) == 0:
            return [0.0] * action_size, 0.0

        obs = self.game.to_array(canonical_board)
        pi_np, root_prediction, latent = self.nnet.predict_with_latent(obs, valids)

        root = _LatentNode(prior=0.0, board=root_board)
        root.latent = latent
        root.raw_value = float(root_prediction)
        root.expanded = True

        if self.params.search_mode == "puct" and self.params.dir_noise and add_exploration_noise:
            noise = np.random.dirichlet([self.params.dir_alpha] * len(valid_indices))
            for i, a in enumerate(valid_indices):
                blended_prior = (1.0 - self.params.dir_fraction) * pi_np[a] + self.params.dir_fraction * noise[i]
                root.children[int(a)] = _LatentNode(prior=float(blended_prior))
        else:
            for a in valid_indices:
                root.children[int(a)] = _LatentNode(prior=float(pi_np[a]))

        gumbel_state = (
            _GumbelRootState(
                root,
                num_sims,
                self.params,
                add_noise=add_exploration_noise,
            )
            if self.params.search_mode == "gumbel"
            else None
        )
        for _ in range(num_sims):
            if should_stop is not None and should_stop():
                break
            root_action = gumbel_state.select_action(root) if gumbel_state is not None else None
            self._latent_simulate(root, root_action=root_action)
            self.last_simulations += 1

        counts = np.zeros(action_size, dtype=np.float64)
        q_sum = np.zeros(action_size, dtype=np.float64)
        for action_key, child in root.children.items():
            idx: int = int(action_key)
            counts[idx] = float(child.visit_count)
            q_sum[idx] = child.value_sum

        total_visits = counts.sum()
        root_value = float(q_sum.sum() / max(total_visits, 1))

        if self.params.search_mode == "gumbel":
            if gumbel_state is None:
                raise RuntimeError("Gumbel search state was not initialized")
            self.last_action = gumbel_state.proposed_action(root)
            if temp == 0:
                probs = [0.0] * action_size
                probs[self.last_action] = 1.0
            else:
                probs = _gumbel_improved_policy(root, action_size, self.params).tolist()
        elif counts.sum() > 0:
            self.last_action = int(np.flatnonzero(counts == counts.max())[0])
            if temp == 0:
                probs = [0.0] * action_size
                probs[self.last_action] = 1.0
            else:
                probs = _visit_count_policy(counts, temp).tolist()
        else:
            actions = np.fromiter(root.children, dtype=np.int64)
            priors = np.array([root.children[int(action)].prior for action in actions], dtype=np.float64)
            self.last_action = int(actions[int(np.argmax(priors))])
            probs_array = np.zeros(action_size, dtype=np.float64)
            if temp == 0:
                probs_array[self.last_action] = 1.0
            else:
                probs_array[actions] = _visit_count_policy(np.maximum(priors, EPS), temp)
            probs = probs_array.tolist()

        return probs, root_value

    def _latent_simulate(self, node: _LatentNode, root_action: int | None = None) -> float:
        """Simulate one path and return its value from ``node``'s perspective."""
        if not node.expanded or not node.children:
            return 0.0

        best_action = root_action if root_action is not None else _interior_best_action(node, self.params)
        if best_action not in node.children:
            raise RuntimeError(f"search selected absent action {best_action}")
        child = node.children[best_action]
        discount = float(self.params.discount)

        if not child.expanded and node.latent is not None:
            child_valid_mask = None
            terminal_value: float | None = None
            if node.board is not None:
                try:
                    parent_player = player_from_turn(node.board.turn)
                    child_board, child_player = self.game.get_next_search_state(node.board, parent_player, best_action)
                    terminal_value = self.game.get_game_outcome(child_board, child_player)
                    child.board = child_board
                    if terminal_value is None:
                        child_valid_mask = self.game.get_valid_moves(child_board, child_player)
                except ValueError as exc:
                    raise RuntimeError(f"MCTS selected invalid action {best_action} at {node.board.fen()}") from exc

            child.expanded = True
            if terminal_value is not None:
                # The environment value is for the next player. Edge rewards and
                # child Q statistics are represented from the acting parent player.
                child.reward = -float(terminal_value)
                child.terminal = True
                q = child.reward
            else:
                pi_np, value, reward, next_latent = self.nnet.recurrent_predict(
                    node.latent, best_action, valid_mask=child_valid_mask
                )
                child.latent = next_latent
                child.reward = reward
                child.raw_value = float(value)

                child_indices = (
                    np.flatnonzero(child_valid_mask) if child_valid_mask is not None else np.flatnonzero(pi_np > 0.0)
                )
                for a in child_indices:
                    child.children[int(a)] = _LatentNode(prior=float(pi_np[a]))

                q = child.reward - discount * value
            child.visit_count += 1
            child.value_sum += q
            node.total_child_visits += 1
            return q

        child_value = self._latent_simulate(child)
        q = child.reward - discount * child_value
        child.visit_count += 1
        child.value_sum += q
        node.total_child_visits += 1
        return q


class _PendingExpansion:
    __slots__ = ("ancestors", "child")

    def __init__(self, ancestors: list[_LatentNode], child: _LatentNode) -> None:
        self.ancestors = ancestors
        self.child = child


def _backup_latent_path(ancestors: list[_LatentNode], leaf: _LatentNode, q_leaf: float, discount: float) -> None:
    """Back up values with alternating player perspective from leaf to root.

    ``q_leaf`` is the value of the leaf edge from its parent perspective. Each
    preceding edge combines its parent-perspective reward with the negated value
    of the selected continuation at the child node.
    """
    leaf.visit_count += 1
    leaf.value_sum += q_leaf
    ancestors[-1].total_child_visits += 1
    q = q_leaf
    for j in range(len(ancestors) - 1, 0, -1):
        child = ancestors[j]
        parent = ancestors[j - 1]
        q = child.reward - discount * q
        child.visit_count += 1
        child.value_sum += q
        parent.total_child_visits += 1


class BatchedMCTS:
    """Batch leaf expansion across independent search trees."""

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
        self._pending_parent_boards: list[chess.Board | None] = []  # For computing child valid masks
        self.last_actions: list[int | None] = []

    def search_batch(
        self,
        canonical_boards: list[chess.Board],
        num_sims: int | None = None,
        temp: float = 1.0,
        *,
        add_exploration_noise: bool | Sequence[bool] | None = None,
    ) -> list[tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
        """Run batched latent MCTS for multiple positions.

        Returns one tuple per board: ``(policy, root_value, obs, valid)``. *policy* is a
        float32 vector summing to 1 (``numpy.ndarray``, shape ``(action_size,)``). *obs* and *valid*
        are copies of the rows used for root inference. ``last_actions`` contains
        the actual Sequential Halving proposals; see :meth:`MCTS.search_latent`.
        Exploration noise accepts either one flag for all roots or one flag per
        root for sliding self-play pools at different ply counts.
        """
        self.last_actions = []
        if num_sims is None:
            num_sims = self.params.num_mcts_sims

        N = len(canonical_boards)
        if N == 0:
            return []
        _validate_search(self.params, num_sims, temp)
        if add_exploration_noise is None:
            exploration_noise = [temp > 0.0] * N
        elif isinstance(add_exploration_noise, bool):
            exploration_noise = [add_exploration_noise] * N
        else:
            exploration_noise = [bool(enabled) for enabled in add_exploration_noise]
            if len(exploration_noise) != N:
                raise ValueError("add_exploration_noise must contain one flag per batched root")
        action_size = self.game.get_action_size()
        root_boards = [board.copy(stack=board.halfmove_clock) for board in canonical_boards]
        root_outcomes = [self.game.get_game_outcome(board, 1) for board in root_boards]
        discount = float(self.params.discount)
        cpuct = self.params.cpuct
        tm = self._timings

        if tm is not None:
            tm.search_batch_calls += 1
            t0 = time.perf_counter()

        sample_obs = self.game.to_array(canonical_boards[0])
        obs_batch = np.empty((N, *sample_obs.shape), dtype=np.float32)
        valid_batch = np.empty((N, action_size), dtype=np.float32)
        for i, (canonical_board, root_board) in enumerate(zip(canonical_boards, root_boards)):
            obs_batch[i] = self.game.to_array(canonical_board)
            valid_batch[i] = self.game.get_valid_moves(root_board, 1)

        if tm is not None:
            tm.encode_s += time.perf_counter() - t0
            t0 = time.perf_counter()

        active_indices = [i for i, outcome in enumerate(root_outcomes) if outcome is None]
        policies_np = np.zeros((N, action_size), dtype=np.float32)
        root_predictions = np.zeros(N, dtype=np.float32)
        root_latents: list[torch.Tensor | None] = [None] * N
        if active_indices:
            active_policies, active_predictions, active_latents = self.nnet.batched_initial_inference(
                obs_batch[active_indices],
                valid_batch[active_indices],
            )
            for batch_index, root_index in enumerate(active_indices):
                policies_np[root_index] = active_policies[batch_index]
                root_predictions[root_index] = float(np.asarray(active_predictions[batch_index]).item())
                root_latents[root_index] = active_latents[batch_index : batch_index + 1]

        if tm is not None:
            tm.initial_inf_s += time.perf_counter() - t0

        roots: list[_LatentNode] = []
        for i in range(N):
            root = _LatentNode(prior=0.0, board=root_boards[i])
            root_outcome = root_outcomes[i]
            root.raw_value = float(root_outcome) if root_outcome is not None else float(root_predictions[i])
            root.expanded = True
            if root_outcome is not None:
                roots.append(root)
                continue

            root_latent = root_latents[i]
            if root_latent is None:
                raise RuntimeError("Initial inference returned no latent state for a non-terminal root")
            root.latent = root_latent

            valid_indices = np.flatnonzero(valid_batch[i])
            pi = policies_np[i]

            if (
                self.params.search_mode == "puct"
                and self.params.dir_noise
                and exploration_noise[i]
                and len(valid_indices) > 0
            ):
                noise = np.random.dirichlet([self.params.dir_alpha] * len(valid_indices))
                for j, a in enumerate(valid_indices):
                    root.children[int(a)] = _LatentNode(
                        prior=float((1.0 - self.params.dir_fraction) * pi[a] + self.params.dir_fraction * noise[j])
                    )
            else:
                for a in valid_indices:
                    root.children[int(a)] = _LatentNode(prior=float(pi[a]))

            roots.append(root)

        gumbel_states: list[_GumbelRootState | None] = []
        for root, add_noise in zip(roots, exploration_noise):
            state = (
                _GumbelRootState(
                    root,
                    num_sims,
                    self.params,
                    add_noise=add_noise,
                )
                if self.params.search_mode == "gumbel" and root.children
                else None
            )
            gumbel_states.append(state)

        for _ in range(num_sims):
            if tm is not None:
                t_sel = time.perf_counter()

            pending = self._pending
            parent_latents = self._parent_latents
            pending_actions = self._pending_actions
            parent_boards = self._pending_parent_boards
            pending.clear()
            parent_latents.clear()
            pending_actions.clear()
            parent_boards.clear()

            for root, gumbel_state in zip(roots, gumbel_states):
                if not root.children:
                    continue
                root_action = gumbel_state.select_action(root) if gumbel_state is not None else None
                result = self._select_leaf(root, cpuct, root_action=root_action)
                if result is not None:
                    ancestors, child, action = result
                    parent_node = ancestors[-1]
                    parent_latent = parent_node.latent
                    if parent_latent is None:
                        raise RuntimeError("Selected a parent node without a latent state")
                    pending.append(_PendingExpansion(ancestors, child))
                    parent_latents.append(parent_latent)
                    pending_actions.append(action)
                    parent_boards.append(parent_node.board)

            if tm is not None:
                tm.selection_s += time.perf_counter() - t_sel

            if not pending:
                continue

            if tm is not None:
                t_rec = time.perf_counter()

            child_boards_list: list[chess.Board | None] = []
            valid_masks_list: list[np.ndarray | None] = []
            terminal_values: list[float | None] = []
            for parent_board, action in zip(parent_boards, pending_actions):
                if parent_board is not None:
                    try:
                        parent_player = player_from_turn(parent_board.turn)
                        child_board, child_player = self.game.get_next_search_state(parent_board, parent_player, action)
                        terminal_value = self.game.get_game_outcome(child_board, child_player)
                        child_boards_list.append(child_board)
                        child_valid_mask = (
                            self.game.get_valid_moves(child_board, child_player) if terminal_value is None else None
                        )
                        valid_masks_list.append(child_valid_mask)
                        terminal_values.append(terminal_value)
                    except ValueError as exc:
                        raise RuntimeError(
                            f"Batched MCTS selected invalid action {action} at {parent_board.fen()}"
                        ) from exc
                else:
                    child_boards_list.append(None)
                    valid_masks_list.append(None)
                    terminal_values.append(None)

            inference_indices = [j for j, terminal_value in enumerate(terminal_values) if terminal_value is None]
            rb = None
            if inference_indices:
                batched_latent = torch.cat([parent_latents[j] for j in inference_indices], dim=0)
                rb = self.nnet.batched_recurrent_inference(
                    batched_latent,
                    [pending_actions[j] for j in inference_indices],
                    valid_masks=[valid_masks_list[j] for j in inference_indices],
                    policy_topk=self.params.recurrent_policy_topk,
                )

            if tm is not None:
                tm.recurrent_inf_s += time.perf_counter() - t_rec
                t_bu = time.perf_counter()

            for j, terminal_value in enumerate(terminal_values):
                if terminal_value is None:
                    continue
                pe = pending[j]
                child = pe.child
                child.latent = None
                child.raw_value = float(terminal_value)
                child.reward = -float(terminal_value)
                child.terminal = True
                child.expanded = True
                child.board = child_boards_list[j]
                _backup_latent_path(pe.ancestors, child, child.reward, discount)

            if rb is None:
                if tm is not None:
                    tm.expand_backup_s += time.perf_counter() - t_bu
                continue

            v_f = np.asarray(rb.values, dtype=np.float64)
            r_f = np.asarray(rb.rewards, dtype=np.float64)
            next_latents = rb.next_latent
            q_all = r_f - discount * v_f

            if rb.policy_full is not None:
                pi_batch = rb.policy_full
                for output_index, pending_index in enumerate(inference_indices):
                    pe = pending[pending_index]
                    child = pe.child
                    child.latent = next_latents[output_index : output_index + 1]
                    child.raw_value = float(v_f[output_index])
                    child.reward = float(r_f[output_index])
                    child.expanded = True

                    # Reuse precomputed child board (no redundant computation)
                    child.board = child_boards_list[pending_index]

                    pi_row = pi_batch[output_index]
                    valid_mask = valid_masks_list[pending_index]
                    child_indices = (
                        np.flatnonzero(valid_mask) if valid_mask is not None else np.flatnonzero(pi_row > 0.0)
                    )
                    for a in child_indices:
                        child.children[int(a)] = _LatentNode(prior=float(pi_row[a]))

                    _backup_latent_path(pe.ancestors, child, float(q_all[output_index]), discount)
            else:
                idx_bt = rb.topk_indices
                prob_bt = rb.topk_probs
                if idx_bt is None or prob_bt is None:
                    raise RuntimeError("Sparse recurrent inference returned no policy candidates")
                k_w = idx_bt.shape[1]
                for output_index, pending_index in enumerate(inference_indices):
                    pe = pending[pending_index]
                    child = pe.child
                    child.latent = next_latents[output_index : output_index + 1]
                    child.raw_value = float(v_f[output_index])
                    child.reward = float(r_f[output_index])
                    child.expanded = True

                    # Reuse precomputed child board (no redundant computation)
                    child.board = child_boards_list[pending_index]

                    valid_mask = valid_masks_list[pending_index]
                    for t in range(k_w):
                        action = int(idx_bt[output_index, t])
                        probability = float(prob_bt[output_index, t])
                        if valid_mask is None:
                            if probability > 0.0:
                                child.children[action] = _LatentNode(prior=probability)
                        elif valid_mask[action] > 0.0:
                            child.children[action] = _LatentNode(prior=probability)
                    if valid_mask is not None and len(child.children) != int(np.count_nonzero(valid_mask)):
                        raise RuntimeError("top-K recurrent policy omitted a legal action")

                    _backup_latent_path(pe.ancestors, child, float(q_all[output_index]), discount)

            if tm is not None:
                tm.expand_backup_s += time.perf_counter() - t_bu

        if tm is not None:
            t_fin = time.perf_counter()

        self.last_actions = [None] * N
        results: list[tuple[np.ndarray, float, np.ndarray, np.ndarray]] = []
        for i, root in enumerate(roots):
            counts = np.zeros(action_size, dtype=np.float64)
            q_sum = np.zeros(action_size, dtype=np.float64)
            for ak, ch in root.children.items():
                counts[int(ak)] = float(ch.visit_count)
                q_sum[int(ak)] = ch.value_sum

            total_visits = counts.sum()
            root_outcome = root_outcomes[i]
            root_value = float(root_outcome) if root_outcome is not None else float(q_sum.sum() / max(total_visits, 1))

            if self.params.search_mode == "gumbel" and root.children:
                gumbel_state = gumbel_states[i]
                if gumbel_state is None:
                    raise RuntimeError("Gumbel search state was not initialized")
                proposed_action = gumbel_state.proposed_action(root)
                self.last_actions[i] = proposed_action
                if temp == 0:
                    probs_arr = np.zeros(action_size, dtype=np.float32)
                    probs_arr[proposed_action] = 1.0
                else:
                    probs_arr = _gumbel_improved_policy(root, action_size, self.params)
            elif total_visits > 0:
                proposed_action = int(np.flatnonzero(counts == counts.max())[0])
                self.last_actions[i] = proposed_action
                if temp == 0:
                    probs_arr = np.zeros(action_size, dtype=np.float32)
                    probs_arr[proposed_action] = 1.0
                else:
                    probs_arr = _visit_count_policy(counts, temp).astype(np.float32, copy=False)
            else:
                probs_arr = np.zeros(action_size, dtype=np.float32)

            results.append((probs_arr, root_value, obs_batch[i].copy(), valid_batch[i].copy()))

        if tm is not None:
            tm.finalize_s += time.perf_counter() - t_fin

        return results

    def _select_leaf(
        self, root: _LatentNode, cpuct: float, root_action: int | None = None
    ) -> tuple[list[_LatentNode], _LatentNode, int] | None:
        """Return the first unexpanded edge reachable under the selection policy."""
        if not root.expanded or not root.children:
            return None

        ancestors: list[_LatentNode] = [root]
        current = root
        while True:
            if current is root and root_action is not None:
                best_action = root_action
            elif self.params.search_mode == "gumbel":
                best_action = _gumbel_interior_best_action(current, self.params)
            else:
                best_action = _puct_best_action(cpuct, self.params.pb_c_base, current)
            if best_action not in current.children:
                raise RuntimeError(f"search selected absent action {best_action}")
            child = current.children[best_action]

            if not child.expanded and current.latent is not None:
                return ancestors, child, best_action

            if child.expanded and child.terminal:
                _backup_latent_path(
                    ancestors,
                    child,
                    child.reward,
                    float(self.params.discount),
                )
                return None

            if not child.expanded or not child.children:
                return None

            ancestors.append(child)
            current = child
