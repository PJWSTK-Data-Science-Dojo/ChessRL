"""Exact chess-state expansion for serial MCTS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
import numpy as np

from luna.game.chess_game import ChessGame, player_from_turn
from luna.mcts_tree import _LatentNode

if TYPE_CHECKING:
    from luna.network import LunaNetwork


@dataclass(frozen=True, slots=True)
class _ExactTransition:
    board: chess.Board
    valid_mask: np.ndarray | None
    terminal_value: float | None


class ExactStateExpander:
    """Expand serial-search children by re-encoding exact chess positions."""

    def __init__(self, game: ChessGame, network: LunaNetwork, discount: float) -> None:
        self._game = game
        self._network = network
        self._discount = discount

    def expand(self, parent: _LatentNode, child: _LatentNode, action: int) -> float:
        transition = self._prepare_transition(parent, action)
        child.board = transition.board
        child.expanded = True
        if transition.terminal_value is not None:
            q_value = self._initialize_terminal(child, transition.terminal_value)
        else:
            q_value = -self._discount * self._infer_child(child, transition.valid_mask)
        return _record_edge_value(parent, child, q_value)

    def _prepare_transition(self, parent: _LatentNode, action: int) -> _ExactTransition:
        if parent.board is None:
            raise RuntimeError("Exact-state expansion requires a parent board")
        try:
            parent_player = player_from_turn(parent.board.turn)
            child_board, child_player = self._game.get_next_exact_search_state(parent.board, parent_player, action)
        except ValueError as exc:
            raise RuntimeError(f"MCTS selected invalid action {action} at {parent.board.fen()}") from exc
        terminal_value = self._game.get_game_outcome(child_board, child_player)
        valid_mask = self._game.get_valid_moves(child_board, child_player) if terminal_value is None else None
        return _ExactTransition(child_board, valid_mask, terminal_value)

    @staticmethod
    def _initialize_terminal(child: _LatentNode, terminal_value: float) -> float:
        child.raw_value = terminal_value
        child.reward = -terminal_value
        child.terminal = True
        return child.reward

    def _infer_child(self, child: _LatentNode, valid_mask: np.ndarray | None) -> float:
        if child.board is None or valid_mask is None:
            raise RuntimeError("Exact-state inference requires a non-terminal child board")
        player = player_from_turn(child.board.turn)
        canonical = self._game.get_canonical_form(child.board, player)
        policy, value, latent = self._network.predict_with_latent(self._game.to_array(canonical), valid_mask)
        child.latent = latent
        child.reward = 0.0
        child.raw_value = float(value)
        for action in np.flatnonzero(valid_mask):
            child.children[int(action)] = _LatentNode(prior=float(policy[action]))
        return float(value)


def _record_edge_value(parent: _LatentNode, child: _LatentNode, q_value: float) -> float:
    child.visit_count += 1
    child.value_sum += q_value
    parent.total_child_visits += 1
    return q_value
