"""Batched latent MCTS orchestration."""

from __future__ import annotations

import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
import torch

from luna.config import MCTSParams
from luna.game.chess_game import ChessGame
from luna.mcts_batched_expansion import (
    PendingExpansionBatch,
    backup_exact_expansions,
    backup_expansions,
    infer_exact_expansions,
    infer_recurrent_expansions,
    prepare_expansion_transitions,
)
from luna.mcts_batched_roots import (
    RootSearchSettings,
    SearchResult,
    SearchRoots,
    build_search_roots,
    encode_root_batch,
    finalize_search_roots,
    infer_root_batch,
)
from luna.mcts_search_contempt import SearchContemptState, SearchContemptStats
from luna.mcts_tree import (
    _backup_latent_path,
    _LatentNode,
    _PendingExpansion,
    _validate_search,
)
from luna.profiling import SelfPlayMCTSTimings

if TYPE_CHECKING:
    from luna.network import LunaNetwork


@dataclass(frozen=True, slots=True)
class _BatchRequest:
    num_simulations: int
    temperature: float
    exploration_noise: list[bool]
    root_action_restrictions: list[Collection[int] | None]
    discount: float


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
        self._pending_parent_boards: list[chess.Board | None] = []
        self.last_actions: list[int | None] = []
        self.last_search_contempt_stats: list[SearchContemptStats] = []

    def search_batch(
        self,
        canonical_boards: list[chess.Board],
        num_sims: int | None = None,
        temp: float = 1.0,
        *,
        add_exploration_noise: bool | Sequence[bool] | None = None,
        allowed_root_actions: Sequence[Collection[int] | None] | None = None,
    ) -> list[SearchResult]:
        """Run latent MCTS for each position, batching network inference."""
        self.last_actions = []
        self.last_search_contempt_stats = []
        if not canonical_boards:
            return []
        request = self._build_request(
            len(canonical_boards),
            num_sims,
            temp,
            add_exploration_noise,
            allowed_root_actions,
        )
        search = self._prepare_roots(canonical_boards, request)
        for _ in range(request.num_simulations):
            self._run_simulation(search, request)
        return self._finalize(search, request.temperature)

    def _build_request(
        self,
        root_count: int,
        num_sims: int | None,
        temperature: float,
        exploration_noise: bool | Sequence[bool] | None,
        allowed_root_actions: Sequence[Collection[int] | None] | None,
    ) -> _BatchRequest:
        num_simulations = self.params.num_mcts_sims if num_sims is None else num_sims
        _validate_search(self.params, num_simulations, temperature)
        return _BatchRequest(
            num_simulations,
            temperature,
            _resolve_exploration_noise(root_count, temperature, exploration_noise),
            _resolve_root_action_restrictions(root_count, allowed_root_actions),
            float(self.params.discount),
        )

    def _prepare_roots(
        self,
        canonical_boards: list[chess.Board],
        request: _BatchRequest,
    ) -> SearchRoots:
        timings = self._timings
        started_at = time.perf_counter()
        if timings is not None:
            timings.search_batch_calls += 1
        encoded = encode_root_batch(
            self.game,
            canonical_boards,
            request.root_action_restrictions,
            self.params.tree_state_mode,
        )
        if timings is not None:
            timings.encode_s += time.perf_counter() - started_at
            started_at = time.perf_counter()
        predictions = infer_root_batch(self.nnet, encoded)
        if timings is not None:
            timings.initial_inf_s += time.perf_counter() - started_at
        settings = RootSearchSettings(self.params, request.num_simulations, request.exploration_noise)
        return build_search_roots(encoded, predictions, settings)

    def _run_simulation(self, search: SearchRoots, request: _BatchRequest) -> None:
        timings = self._timings
        started_at = time.perf_counter()
        batch = self._select_expansions(search)
        if timings is not None:
            timings.selection_s += time.perf_counter() - started_at
        if not batch.pending:
            return
        started_at = time.perf_counter()
        transitions = prepare_expansion_transitions(
            self.game,
            batch,
            self.params.tree_state_mode,
        )
        if timings is not None:
            timings.expand_backup_s += time.perf_counter() - started_at
            started_at = time.perf_counter()
        if self.params.tree_state_mode == "exact":
            exact = infer_exact_expansions(self.nnet, self.game, transitions)
        else:
            exact = None
            recurrent = infer_recurrent_expansions(
                self.nnet,
                batch,
                transitions,
                self.params.recurrent_policy_topk,
            )
        if timings is not None:
            timings.recurrent_inf_s += time.perf_counter() - started_at
            started_at = time.perf_counter()
        if self.params.tree_state_mode == "exact":
            backup_exact_expansions(batch, transitions, exact, request.discount)
        else:
            backup_expansions(batch, transitions, recurrent, request.discount)
        if timings is not None:
            timings.expand_backup_s += time.perf_counter() - started_at

    def _select_expansions(self, search: SearchRoots) -> PendingExpansionBatch:
        self._clear_pending()
        search_states = zip(
            search.roots,
            search.gumbel_states,
            search.search_contempt_states,
            strict=True,
        )
        for root, gumbel_state, search_contempt in search_states:
            if not root.children:
                continue
            root_action = gumbel_state.select_action(root) if gumbel_state is not None else None
            selected = self._select_leaf(root, search_contempt, root_action=root_action)
            if selected is not None:
                self._queue_expansion(selected)
        return PendingExpansionBatch(
            self._pending,
            self._parent_latents,
            self._pending_actions,
            self._pending_parent_boards,
        )

    def _clear_pending(self) -> None:
        self._pending.clear()
        self._parent_latents.clear()
        self._pending_actions.clear()
        self._pending_parent_boards.clear()

    def _queue_expansion(
        self,
        selected: tuple[list[_LatentNode], _LatentNode, int],
    ) -> None:
        ancestors, child, action = selected
        parent_node = ancestors[-1]
        parent_latent = parent_node.latent
        if parent_latent is None:
            raise RuntimeError("Selected a parent node without a latent state")
        self._pending.append(_PendingExpansion(ancestors, child))
        self._parent_latents.append(parent_latent)
        self._pending_actions.append(action)
        self._pending_parent_boards.append(parent_node.board)

    def _finalize(self, search: SearchRoots, temperature: float) -> list[SearchResult]:
        started_at = time.perf_counter()
        finalized = finalize_search_roots(search, self.params, temperature)
        self.last_actions = finalized.actions
        self.last_search_contempt_stats = [state.stats for state in search.search_contempt_states]
        if self._timings is not None:
            self._timings.finalize_s += time.perf_counter() - started_at
        return finalized.results

    def _select_leaf(
        self,
        root: _LatentNode,
        search_contempt: SearchContemptState,
        root_action: int | None = None,
    ) -> tuple[list[_LatentNode], _LatentNode, int] | None:
        """Return the first unexpanded edge reachable under the selection policy."""
        if not root.expanded or not root.children:
            return None
        ancestors: list[_LatentNode] = [root]
        current = root
        while True:
            depth = len(ancestors) - 1
            best_action = self._select_action(current, root, search_contempt, root_action, depth)
            if best_action not in current.children:
                raise RuntimeError(f"search selected absent action {best_action}")
            child = current.children[best_action]
            if not child.expanded and current.latent is not None:
                return ancestors, child, best_action
            if child.expanded and child.terminal:
                _backup_latent_path(ancestors, child, child.reward, float(self.params.discount))
                return None
            if not child.expanded or not child.children:
                return None
            ancestors.append(child)
            current = child

    def _select_action(
        self,
        current: _LatentNode,
        root: _LatentNode,
        search_contempt: SearchContemptState,
        root_action: int | None,
        depth: int,
    ) -> int:
        if current is root and root_action is not None:
            return root_action
        return search_contempt.select_action(current, depth, self.params)


def _resolve_exploration_noise(
    root_count: int,
    temperature: float,
    setting: bool | Sequence[bool] | None,
) -> list[bool]:
    if setting is None:
        return [temperature > 0.0] * root_count
    if isinstance(setting, bool):
        return [setting] * root_count
    resolved = [bool(enabled) for enabled in setting]
    if len(resolved) != root_count:
        raise ValueError("add_exploration_noise must contain one flag per batched root")
    return resolved


def _resolve_root_action_restrictions(
    root_count: int,
    setting: Sequence[Collection[int] | None] | None,
) -> list[Collection[int] | None]:
    if setting is None:
        return [None] * root_count
    restrictions = list(setting)
    if len(restrictions) != root_count:
        raise ValueError("allowed_root_actions must contain one entry per batched root")
    return restrictions
