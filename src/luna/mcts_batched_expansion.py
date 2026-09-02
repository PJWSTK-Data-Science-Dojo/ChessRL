"""Leaf expansion and backup for batched latent MCTS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import chess
import numpy as np
import torch

from luna.game.chess_game import ChessGame, player_from_turn
from luna.mcts_tree import _backup_latent_path, _LatentNode, _PendingExpansion

if TYPE_CHECKING:
    from luna.network import LunaNetwork
    from luna.network_types import RecurrentBatchResult


@dataclass(frozen=True, slots=True)
class PendingExpansionBatch:
    pending: list[_PendingExpansion]
    parent_latents: list[torch.Tensor]
    actions: list[int]
    parent_boards: list[chess.Board | None]


@dataclass(frozen=True, slots=True)
class ExpansionTransitions:
    child_boards: list[chess.Board | None]
    valid_masks: list[np.ndarray | None]
    terminal_values: list[float | None]
    inference_indices: list[int]


@dataclass(frozen=True, slots=True)
class ExactExpansionBatch:
    policies: np.ndarray
    values: np.ndarray
    latents: torch.Tensor


def prepare_expansion_transitions(
    game: ChessGame,
    batch: PendingExpansionBatch,
    tree_state_mode: Literal["latent", "exact"],
) -> ExpansionTransitions:
    child_boards: list[chess.Board | None] = []
    valid_masks: list[np.ndarray | None] = []
    terminal_values: list[float | None] = []
    for parent_board, action in zip(batch.parent_boards, batch.actions, strict=True):
        transition = _prepare_transition(game, parent_board, action, tree_state_mode)
        child_board, valid_mask, terminal_value = transition
        child_boards.append(child_board)
        valid_masks.append(valid_mask)
        terminal_values.append(terminal_value)
    inference_indices = [index for index, terminal_value in enumerate(terminal_values) if terminal_value is None]
    return ExpansionTransitions(child_boards, valid_masks, terminal_values, inference_indices)


def _prepare_transition(
    game: ChessGame,
    parent_board: chess.Board | None,
    action: int,
    tree_state_mode: Literal["latent", "exact"],
) -> tuple[chess.Board | None, np.ndarray | None, float | None]:
    if parent_board is None:
        return None, None, None
    try:
        parent_player = player_from_turn(parent_board.turn)
        transition = (
            game.get_next_exact_search_state if tree_state_mode == "exact" else game.get_next_latent_search_state
        )
        child_board, child_player = transition(parent_board, parent_player, action)
        terminal_value = game.get_game_outcome(child_board, child_player)
        valid_mask = game.get_valid_moves(child_board, child_player) if terminal_value is None else None
        return child_board, valid_mask, terminal_value
    except ValueError as exc:
        raise RuntimeError(f"Batched MCTS selected invalid action {action} at {parent_board.fen()}") from exc


def infer_recurrent_expansions(
    network: LunaNetwork,
    batch: PendingExpansionBatch,
    transitions: ExpansionTransitions,
    policy_topk: int | None,
) -> RecurrentBatchResult | None:
    indices = transitions.inference_indices
    if not indices:
        return None
    batched_latent = torch.cat([batch.parent_latents[index] for index in indices], dim=0)
    return network.batched_recurrent_inference(
        batched_latent,
        [batch.actions[index] for index in indices],
        valid_masks=[transitions.valid_masks[index] for index in indices],
        policy_topk=policy_topk,
    )


def infer_exact_expansions(
    network: LunaNetwork,
    game: ChessGame,
    transitions: ExpansionTransitions,
) -> ExactExpansionBatch | None:
    indices = transitions.inference_indices
    if not indices:
        return None
    boards = [_canonical_child(game, transitions.child_boards[index]) for index in indices]
    observations = np.stack([game.to_array(board) for board in boards])
    valid_masks = np.stack([_required_mask(transitions.valid_masks[index]) for index in indices])
    policies, values, latents = network.batched_initial_inference(observations, valid_masks)
    return ExactExpansionBatch(
        np.asarray(policies, dtype=np.float32),
        np.asarray(values, dtype=np.float32).reshape(len(indices)),
        latents,
    )


def _canonical_child(game: ChessGame, board: chess.Board | None) -> chess.Board:
    if board is None:
        raise RuntimeError("Exact-state expansion requires a child board")
    return game.get_canonical_form(board, player_from_turn(board.turn))


def _required_mask(mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        raise RuntimeError("Exact-state expansion requires a legal-action mask")
    return mask


def backup_expansions(
    batch: PendingExpansionBatch,
    transitions: ExpansionTransitions,
    recurrent: RecurrentBatchResult | None,
    discount: float,
) -> None:
    _backup_terminal_expansions(batch, transitions, discount)
    if recurrent is None:
        return
    values = np.asarray(recurrent.values, dtype=np.float64)
    rewards = np.asarray(recurrent.rewards, dtype=np.float64)
    q_values = rewards - discount * values
    if recurrent.policy_full is not None:
        _backup_dense_expansions(batch, transitions, recurrent, q_values, discount)
    else:
        _backup_sparse_expansions(batch, transitions, recurrent, q_values, discount)


def backup_exact_expansions(
    batch: PendingExpansionBatch,
    transitions: ExpansionTransitions,
    exact: ExactExpansionBatch | None,
    discount: float,
) -> None:
    _backup_terminal_expansions(batch, transitions, discount)
    if exact is None:
        return
    q_values = -discount * exact.values
    for output_index, pending_index in enumerate(transitions.inference_indices):
        child = _initialize_exact_child(batch, transitions, exact, output_index, pending_index)
        valid_mask = _required_mask(transitions.valid_masks[pending_index])
        for action in np.flatnonzero(valid_mask):
            child.children[int(action)] = _LatentNode(prior=float(exact.policies[output_index, action]))
        pending = batch.pending[pending_index]
        _backup_latent_path(pending.ancestors, child, float(q_values[output_index]), discount)


def _initialize_exact_child(
    batch: PendingExpansionBatch,
    transitions: ExpansionTransitions,
    exact: ExactExpansionBatch,
    output_index: int,
    pending_index: int,
) -> _LatentNode:
    child = batch.pending[pending_index].child
    child.latent = exact.latents[output_index : output_index + 1]
    child.raw_value = float(exact.values[output_index])
    child.reward = 0.0
    child.expanded = True
    child.board = transitions.child_boards[pending_index]
    return child


def _backup_terminal_expansions(
    batch: PendingExpansionBatch,
    transitions: ExpansionTransitions,
    discount: float,
) -> None:
    for index, terminal_value in enumerate(transitions.terminal_values):
        if terminal_value is None:
            continue
        pending = batch.pending[index]
        child = pending.child
        child.latent = None
        child.raw_value = float(terminal_value)
        child.reward = -float(terminal_value)
        child.terminal = True
        child.expanded = True
        child.board = transitions.child_boards[index]
        _backup_latent_path(pending.ancestors, child, child.reward, discount)


def _backup_dense_expansions(
    batch: PendingExpansionBatch,
    transitions: ExpansionTransitions,
    recurrent: RecurrentBatchResult,
    q_values: np.ndarray,
    discount: float,
) -> None:
    policy = recurrent.policy_full
    if policy is None:
        raise RuntimeError("Dense recurrent inference returned no policy")
    for output_index, pending_index in enumerate(transitions.inference_indices):
        child = _initialize_recurrent_child(batch, transitions, recurrent, output_index, pending_index)
        policy_row = policy[output_index]
        valid_mask = transitions.valid_masks[pending_index]
        child_indices = np.flatnonzero(valid_mask) if valid_mask is not None else np.flatnonzero(policy_row > 0.0)
        for action in child_indices:
            child.children[int(action)] = _LatentNode(prior=float(policy_row[action]))
        pending = batch.pending[pending_index]
        _backup_latent_path(pending.ancestors, child, float(q_values[output_index]), discount)


def _backup_sparse_expansions(
    batch: PendingExpansionBatch,
    transitions: ExpansionTransitions,
    recurrent: RecurrentBatchResult,
    q_values: np.ndarray,
    discount: float,
) -> None:
    indices = recurrent.topk_indices
    probabilities = recurrent.topk_probs
    if indices is None or probabilities is None:
        raise RuntimeError("Sparse recurrent inference returned no policy candidates")
    for output_index, pending_index in enumerate(transitions.inference_indices):
        child = _initialize_recurrent_child(batch, transitions, recurrent, output_index, pending_index)
        valid_mask = transitions.valid_masks[pending_index]
        _add_sparse_children(child, indices[output_index], probabilities[output_index], valid_mask)
        pending = batch.pending[pending_index]
        _backup_latent_path(pending.ancestors, child, float(q_values[output_index]), discount)


def _initialize_recurrent_child(
    batch: PendingExpansionBatch,
    transitions: ExpansionTransitions,
    recurrent: RecurrentBatchResult,
    output_index: int,
    pending_index: int,
) -> _LatentNode:
    pending = batch.pending[pending_index]
    child = pending.child
    child.latent = recurrent.next_latent[output_index : output_index + 1]
    child.raw_value = float(recurrent.values[output_index])
    child.reward = float(recurrent.rewards[output_index])
    child.expanded = True
    child.board = transitions.child_boards[pending_index]
    return child


def _add_sparse_children(
    child: _LatentNode,
    indices: np.ndarray,
    probabilities: np.ndarray,
    valid_mask: np.ndarray | None,
) -> None:
    for action_value, probability_value in zip(indices, probabilities, strict=True):
        action = int(action_value)
        probability = float(probability_value)
        if valid_mask is None:
            if probability > 0.0:
                child.children[action] = _LatentNode(prior=probability)
        elif valid_mask[action] > 0.0:
            child.children[action] = _LatentNode(prior=probability)
    if valid_mask is not None and len(child.children) != int(np.count_nonzero(valid_mask)):
        raise RuntimeError("top-K recurrent policy omitted a legal action")
