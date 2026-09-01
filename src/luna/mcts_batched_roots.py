"""Root preparation and finalization for batched latent MCTS."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
import numpy as np
import torch

from luna.config import MCTSParams
from luna.game.chess_game import ChessGame
from luna.mcts_gumbel import _gumbel_improved_policy, _GumbelRootState
from luna.mcts_search_contempt import SearchContemptState
from luna.mcts_tree import _LatentNode, _visit_count_policy

if TYPE_CHECKING:
    from luna.network import LunaNetwork

SearchResult = tuple[np.ndarray, float, np.ndarray, np.ndarray]


@dataclass(frozen=True, slots=True)
class EncodedRootBatch:
    action_size: int
    root_boards: list[chess.Board]
    root_outcomes: list[float | None]
    observations: np.ndarray
    legal_masks: np.ndarray
    search_masks: np.ndarray


@dataclass(frozen=True, slots=True)
class RootPredictions:
    policies: np.ndarray
    values: np.ndarray
    latents: list[torch.Tensor | None]


@dataclass(frozen=True, slots=True)
class RootSearchSettings:
    params: MCTSParams
    num_simulations: int
    exploration_noise: Sequence[bool]


@dataclass(frozen=True, slots=True)
class SearchRoots:
    encoded: EncodedRootBatch
    roots: list[_LatentNode]
    gumbel_states: list[_GumbelRootState | None]
    search_contempt_states: list[SearchContemptState]


@dataclass(frozen=True, slots=True)
class BatchedSearchResults:
    results: list[SearchResult]
    actions: list[int | None]


def encode_root_batch(
    game: ChessGame,
    canonical_boards: list[chess.Board],
    root_action_restrictions: Sequence[Collection[int] | None],
) -> EncodedRootBatch:
    action_size = game.get_action_size()
    root_boards = [board.copy(stack=board.halfmove_clock) for board in canonical_boards]
    root_outcomes = [game.get_game_outcome(board, 1) for board in root_boards]
    sample_observation = game.to_array(canonical_boards[0])
    observations = np.empty((len(canonical_boards), *sample_observation.shape), dtype=np.float32)
    legal_masks = np.empty((len(canonical_boards), action_size), dtype=np.float32)
    for index, (canonical_board, root_board) in enumerate(zip(canonical_boards, root_boards, strict=True)):
        observations[index] = game.to_array(canonical_board)
        legal_masks[index] = game.get_valid_moves(root_board, 1)

    search_masks = legal_masks.copy()
    for row, restriction in enumerate(root_action_restrictions):
        if restriction is None:
            continue
        root_mask = _root_action_mask(restriction, action_size)
        search_masks[row] *= root_mask
    return EncodedRootBatch(
        action_size,
        root_boards,
        root_outcomes,
        observations,
        legal_masks,
        search_masks,
    )


def _root_action_mask(restriction: Collection[int], action_size: int) -> np.ndarray:
    root_mask = np.zeros(action_size, dtype=np.float32)
    for action in restriction:
        if isinstance(action, bool) or not isinstance(action, int | np.integer):
            raise ValueError("Root actions must be integer action indices")
        normalized_action = int(action)
        if not 0 <= normalized_action < action_size:
            raise ValueError(f"Root action must be in [0, {action_size}), got {normalized_action}")
        root_mask[normalized_action] = 1.0
    return root_mask


def infer_root_batch(network: LunaNetwork, encoded: EncodedRootBatch) -> RootPredictions:
    batch_size = len(encoded.root_boards)
    active_indices = [index for index, outcome in enumerate(encoded.root_outcomes) if outcome is None]
    latents: list[torch.Tensor | None] = [None] * batch_size
    if len(active_indices) == batch_size:
        policies, values, active_latents = network.batched_initial_inference(
            encoded.observations,
            encoded.search_masks,
        )
        policy_array = np.asarray(policies, dtype=np.float32)
        value_array = np.asarray(values, dtype=np.float32).reshape(batch_size)
        latents = [active_latents[index : index + 1] for index in range(batch_size)]
        return RootPredictions(policy_array, value_array, latents)

    policy_array = np.zeros((batch_size, encoded.action_size), dtype=np.float32)
    value_array = np.zeros(batch_size, dtype=np.float32)
    if active_indices:
        policies, values, active_latents = network.batched_initial_inference(
            encoded.observations[active_indices],
            encoded.search_masks[active_indices],
        )
        for batch_index, root_index in enumerate(active_indices):
            policy_array[root_index] = policies[batch_index]
            value_array[root_index] = float(np.asarray(values[batch_index]).item())
            latents[root_index] = active_latents[batch_index : batch_index + 1]
    return RootPredictions(policy_array, value_array, latents)


def build_search_roots(
    encoded: EncodedRootBatch,
    predictions: RootPredictions,
    settings: RootSearchSettings,
) -> SearchRoots:
    roots = [_build_root(index, encoded, predictions, settings) for index in range(len(encoded.root_boards))]
    gumbel_states = [
        _GumbelRootState(
            root,
            settings.num_simulations,
            settings.params,
            add_noise=add_noise,
        )
        if settings.params.search_mode == "gumbel" and root.children
        else None
        for root, add_noise in zip(roots, settings.exploration_noise, strict=True)
    ]
    search_contempt_states = [SearchContemptState(settings.params.search_contempt_visit_limit) for _ in roots]
    return SearchRoots(encoded, roots, gumbel_states, search_contempt_states)


def _build_root(
    index: int,
    encoded: EncodedRootBatch,
    predictions: RootPredictions,
    settings: RootSearchSettings,
) -> _LatentNode:
    root = _LatentNode(prior=0.0, board=encoded.root_boards[index])
    root_outcome = encoded.root_outcomes[index]
    root.raw_value = float(root_outcome) if root_outcome is not None else float(predictions.values[index])
    root.expanded = True
    if root_outcome is not None:
        return root

    root_latent = predictions.latents[index]
    if root_latent is None:
        raise RuntimeError("Initial inference returned no latent state for a non-terminal root")
    root.latent = root_latent
    valid_indices = np.flatnonzero(encoded.search_masks[index])
    policy = predictions.policies[index]
    if (
        settings.params.search_mode == "puct"
        and settings.params.dir_noise
        and settings.exploration_noise[index]
        and len(valid_indices) > 0
    ):
        noise = np.random.dirichlet([settings.params.dir_alpha] * len(valid_indices))
        for noise_index, action in enumerate(valid_indices):
            prior = (1.0 - settings.params.dir_fraction) * policy[action] + settings.params.dir_fraction * noise[
                noise_index
            ]
            root.children[int(action)] = _LatentNode(prior=float(prior))
    else:
        for action in valid_indices:
            root.children[int(action)] = _LatentNode(prior=float(policy[action]))
    return root


def finalize_search_roots(search: SearchRoots, params: MCTSParams, temperature: float) -> BatchedSearchResults:
    actions: list[int | None] = [None] * len(search.roots)
    results = [_finalize_root(index, search, params, temperature, actions) for index in range(len(search.roots))]
    return BatchedSearchResults(results, actions)


def _finalize_root(
    index: int,
    search: SearchRoots,
    params: MCTSParams,
    temperature: float,
    actions: list[int | None],
) -> SearchResult:
    encoded = search.encoded
    root = search.roots[index]
    counts = np.zeros(encoded.action_size, dtype=np.float64)
    q_sum = np.zeros(encoded.action_size, dtype=np.float64)
    for action, child in root.children.items():
        counts[int(action)] = float(child.visit_count)
        q_sum[int(action)] = child.value_sum

    total_visits = counts.sum()
    root_outcome = encoded.root_outcomes[index]
    root_value = float(root_outcome) if root_outcome is not None else float(q_sum.sum() / max(total_visits, 1))
    if params.search_mode == "gumbel" and root.children:
        gumbel_state = search.gumbel_states[index]
        if gumbel_state is None:
            raise RuntimeError("Gumbel search state was not initialized")
        proposed_action = gumbel_state.proposed_action(root)
        actions[index] = proposed_action
        if temperature == 0:
            policy = np.zeros(encoded.action_size, dtype=np.float32)
            policy[proposed_action] = 1.0
        else:
            policy = _gumbel_improved_policy(root, encoded.action_size, params)
    elif total_visits > 0:
        proposed_action = int(np.flatnonzero(counts == counts.max())[0])
        actions[index] = proposed_action
        if temperature == 0:
            policy = np.zeros(encoded.action_size, dtype=np.float32)
            policy[proposed_action] = 1.0
        else:
            policy = _visit_count_policy(counts, temperature).astype(np.float32, copy=False)
    else:
        policy = np.zeros(encoded.action_size, dtype=np.float32)
    return policy, root_value, encoded.observations[index].copy(), encoded.legal_masks[index].copy()
