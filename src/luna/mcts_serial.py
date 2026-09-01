"""Single-position latent-space MCTS."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
import numpy as np

from luna.config import MCTSParams
from luna.game.chess_game import ChessGame, player_from_turn
from luna.mcts_gumbel import (
    _gumbel_improved_policy,
    _GumbelRootState,
    _interior_best_action,
)
from luna.mcts_tree import EPS, _LatentNode, _validate_search, _visit_count_policy

if TYPE_CHECKING:
    from luna.network import LunaNetwork


@dataclass(frozen=True, slots=True)
class _SearchOptions:
    num_simulations: int
    temperature: float
    add_exploration_noise: bool


@dataclass(frozen=True, slots=True)
class _RootPosition:
    canonical_board: chess.Board
    root_board: chess.Board
    valid_moves: np.ndarray
    valid_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class _PreparedRoot:
    node: _LatentNode
    action_size: int


@dataclass(frozen=True, slots=True)
class _CompletedSearch:
    policy: list[float]
    value: float


@dataclass(frozen=True, slots=True)
class _RootStatistics:
    counts: np.ndarray
    value: float


@dataclass(frozen=True, slots=True)
class _PolicySelection:
    policy: list[float]
    action: int


@dataclass(frozen=True, slots=True)
class _ChildTransition:
    valid_mask: np.ndarray | None
    terminal_value: float | None


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
        options = self._resolve_options(num_sims, temp, add_exploration_noise)
        prepared = self._prepare_root(canonical_board, allowed_root_actions, options)
        if isinstance(prepared, _CompletedSearch):
            return prepared.policy, prepared.value
        gumbel_state = self._create_gumbel_state(prepared.node, options)
        self._run_simulations(prepared.node, gumbel_state, options, should_stop)
        statistics = _root_statistics(prepared.node, prepared.action_size)
        selection = self._select_policy(prepared, gumbel_state, statistics, options)
        self.last_action = selection.action
        return selection.policy, statistics.value

    def _resolve_options(
        self,
        num_sims: int | None,
        temperature: float,
        add_exploration_noise: bool | None,
    ) -> _SearchOptions:
        num_simulations = self.params.num_mcts_sims if num_sims is None else num_sims
        _validate_search(self.params, num_simulations, temperature)
        exploration_noise = temperature > 0.0 if add_exploration_noise is None else add_exploration_noise
        return _SearchOptions(num_simulations, temperature, exploration_noise)

    def _prepare_root(
        self,
        canonical_board: chess.Board,
        allowed_root_actions: Collection[int] | None,
        options: _SearchOptions,
    ) -> _PreparedRoot | _CompletedSearch:
        root_board = canonical_board.copy(stack=canonical_board.halfmove_clock)
        terminal_value = self.game.get_game_outcome(root_board, 1)
        if terminal_value is not None:
            return _CompletedSearch([0.0] * self.game.get_action_size(), float(terminal_value))
        action_size = self.game.get_action_size()
        valid_moves = self._valid_root_moves(root_board, action_size, allowed_root_actions)
        valid_indices = np.flatnonzero(valid_moves)
        if len(valid_indices) == 0:
            return _CompletedSearch([0.0] * action_size, 0.0)
        position = _RootPosition(canonical_board, root_board, valid_moves, valid_indices)
        return _PreparedRoot(self._initialize_root(position, options), action_size)

    def _valid_root_moves(
        self,
        root_board: chess.Board,
        action_size: int,
        allowed_root_actions: Collection[int] | None,
    ) -> np.ndarray:
        valid_moves = self.game.get_valid_moves(root_board, 1)
        if allowed_root_actions is None:
            return valid_moves
        root_mask = np.zeros(action_size, dtype=valid_moves.dtype)
        for action in allowed_root_actions:
            if not 0 <= action < action_size:
                raise ValueError(f"Root action must be in [0, {action_size}), got {action}")
            root_mask[action] = 1
        valid_moves *= root_mask
        return valid_moves

    def _initialize_root(self, position: _RootPosition, options: _SearchOptions) -> _LatentNode:
        observation = self.game.to_array(position.canonical_board)
        policy, prediction, latent = self.nnet.predict_with_latent(observation, position.valid_moves)
        root = _LatentNode(prior=0.0, board=position.root_board)
        root.latent = latent
        root.raw_value = float(prediction)
        root.expanded = True
        self._add_root_children(root, policy, position.valid_indices, options.add_exploration_noise)
        return root

    def _add_root_children(
        self,
        root: _LatentNode,
        policy: np.ndarray,
        valid_indices: np.ndarray,
        add_exploration_noise: bool,
    ) -> None:
        if self.params.search_mode == "puct" and self.params.dir_noise and add_exploration_noise:
            noise = np.random.dirichlet([self.params.dir_alpha] * len(valid_indices))
            for index, action in enumerate(valid_indices):
                prior = (1.0 - self.params.dir_fraction) * policy[action]
                prior += self.params.dir_fraction * noise[index]
                root.children[int(action)] = _LatentNode(prior=float(prior))
            return
        for action in valid_indices:
            root.children[int(action)] = _LatentNode(prior=float(policy[action]))

    def _create_gumbel_state(
        self,
        root: _LatentNode,
        options: _SearchOptions,
    ) -> _GumbelRootState | None:
        if self.params.search_mode != "gumbel":
            return None
        return _GumbelRootState(
            root,
            options.num_simulations,
            self.params,
            add_noise=options.add_exploration_noise,
        )

    def _run_simulations(
        self,
        root: _LatentNode,
        gumbel_state: _GumbelRootState | None,
        options: _SearchOptions,
        should_stop: Callable[[], bool] | None,
    ) -> None:
        for _ in range(options.num_simulations):
            if should_stop is not None and should_stop():
                break
            root_action = gumbel_state.select_action(root) if gumbel_state is not None else None
            self._latent_simulate(root, root_action=root_action)
            self.last_simulations += 1

    def _select_policy(
        self,
        prepared: _PreparedRoot,
        gumbel_state: _GumbelRootState | None,
        statistics: _RootStatistics,
        options: _SearchOptions,
    ) -> _PolicySelection:
        if self.params.search_mode == "gumbel":
            if gumbel_state is None:
                raise RuntimeError("Gumbel search state was not initialized")
            return _gumbel_policy(prepared, gumbel_state, options.temperature, self.params)
        if statistics.counts.sum() > 0:
            return _visited_policy(prepared.action_size, statistics.counts, options.temperature)
        return _prior_policy(prepared, options.temperature)

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
            return self._expand_child(node, child, best_action, discount)
        child_value = self._latent_simulate(child)
        return _record_edge_value(node, child, child.reward - discount * child_value)

    def _expand_child(
        self,
        node: _LatentNode,
        child: _LatentNode,
        action: int,
        discount: float,
    ) -> float:
        transition = self._prepare_child_transition(node, child, action)
        child.expanded = True
        if transition.terminal_value is not None:
            child.reward = -float(transition.terminal_value)
            child.terminal = True
            return _record_edge_value(node, child, child.reward)
        q_value = self._infer_child(node, child, action, transition.valid_mask, discount)
        return _record_edge_value(node, child, q_value)

    def _prepare_child_transition(
        self,
        node: _LatentNode,
        child: _LatentNode,
        action: int,
    ) -> _ChildTransition:
        if node.board is None:
            return _ChildTransition(None, None)
        try:
            parent_player = player_from_turn(node.board.turn)
            child_board, child_player = self.game.get_next_search_state(node.board, parent_player, action)
            terminal_value = self.game.get_game_outcome(child_board, child_player)
            child.board = child_board
            valid_mask = self.game.get_valid_moves(child_board, child_player) if terminal_value is None else None
            return _ChildTransition(valid_mask, terminal_value)
        except ValueError as exc:
            raise RuntimeError(f"MCTS selected invalid action {action} at {node.board.fen()}") from exc

    def _infer_child(
        self,
        node: _LatentNode,
        child: _LatentNode,
        action: int,
        valid_mask: np.ndarray | None,
        discount: float,
    ) -> float:
        parent_latent = node.latent
        if parent_latent is None:
            raise RuntimeError("Cannot expand a child without its parent latent state")
        policy, value, reward, next_latent = self.nnet.recurrent_predict(
            parent_latent,
            action,
            valid_mask=valid_mask,
        )
        child.latent = next_latent
        child.reward = reward
        child.raw_value = float(value)
        child_indices = np.flatnonzero(valid_mask) if valid_mask is not None else np.flatnonzero(policy > 0.0)
        for child_action in child_indices:
            child.children[int(child_action)] = _LatentNode(prior=float(policy[child_action]))
        return child.reward - discount * value


def _root_statistics(root: _LatentNode, action_size: int) -> _RootStatistics:
    counts = np.zeros(action_size, dtype=np.float64)
    q_sum = np.zeros(action_size, dtype=np.float64)
    for action, child in root.children.items():
        index = int(action)
        counts[index] = float(child.visit_count)
        q_sum[index] = child.value_sum
    total_visits = counts.sum()
    value = float(root.raw_value if total_visits == 0 else q_sum.sum() / total_visits)
    return _RootStatistics(counts, value)


def _gumbel_policy(
    prepared: _PreparedRoot,
    gumbel_state: _GumbelRootState,
    temperature: float,
    params: MCTSParams,
) -> _PolicySelection:
    action = gumbel_state.proposed_action(prepared.node)
    if temperature == 0:
        policy = [0.0] * prepared.action_size
        policy[action] = 1.0
    else:
        policy = _gumbel_improved_policy(prepared.node, prepared.action_size, params).tolist()
    return _PolicySelection(policy, action)


def _visited_policy(action_size: int, counts: np.ndarray, temperature: float) -> _PolicySelection:
    action = int(np.flatnonzero(counts == counts.max())[0])
    if temperature == 0:
        policy = [0.0] * action_size
        policy[action] = 1.0
    else:
        policy = _visit_count_policy(counts, temperature).tolist()
    return _PolicySelection(policy, action)


def _prior_policy(prepared: _PreparedRoot, temperature: float) -> _PolicySelection:
    actions = np.fromiter(prepared.node.children, dtype=np.int64)
    priors = np.array([prepared.node.children[int(action)].prior for action in actions], dtype=np.float64)
    action = int(actions[int(np.argmax(priors))])
    policy = np.zeros(prepared.action_size, dtype=np.float64)
    if temperature == 0:
        policy[action] = 1.0
    else:
        policy[actions] = _visit_count_policy(np.maximum(priors, EPS), temperature)
    return _PolicySelection(policy.tolist(), action)


def _record_edge_value(parent: _LatentNode, child: _LatentNode, q_value: float) -> float:
    child.visit_count += 1
    child.value_sum += q_value
    parent.total_child_visits += 1
    return q_value
