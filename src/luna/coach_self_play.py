"""Single-game self-play and trajectory construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import chess
import numpy as np

from luna.config import TrainingRunConfig
from luna.game.chess_game import ChessGame
from luna.mcts import MCTS
from luna.mcts_search_contempt import SearchContemptStats
from luna.replay_buffer import Trajectory

if TYPE_CHECKING:
    from luna.coach import Coach


def self_play_exploration_enabled(board: chess.Board, ply: int, run: TrainingRunConfig) -> bool:
    """Keep Gumbel stochastic when a deterministic root starts cycling."""
    if ply < run.temp_threshold:
        return True
    return run.tree_state_mode == "latent" and run.search_mode == "gumbel" and board.is_repetition(2)


@dataclass(frozen=True, slots=True)
class SelfPlaySearchPlan:
    simulations: int
    train_policy: bool


def select_self_play_search_plan(run: TrainingRunConfig) -> SelfPlaySearchPlan:
    """Draw one shared Playout Cap Randomization cohort."""
    if run.playout_cap_full_probability <= 0.0:
        return SelfPlaySearchPlan(run.num_mcts_sims, True)
    full_search = bool(np.random.random() < run.playout_cap_full_probability)
    simulations = run.playout_cap_full_sims if full_search else run.playout_cap_fast_sims
    return SelfPlaySearchPlan(simulations, full_search)


def select_self_play_action(
    run: TrainingRunConfig,
    policy: np.ndarray | list[float],
    *,
    explore: bool,
    gumbel_proposal: int | None,
) -> int:
    """Select the action actually executed by a self-play actor."""
    if run.search_mode == "gumbel":
        if gumbel_proposal is None:
            raise RuntimeError("Gumbel search did not propose an action")
        return gumbel_proposal
    probabilities = np.asarray(policy, dtype=np.float64)
    if explore:
        return int(np.random.choice(len(probabilities), p=probabilities))
    return int(np.argmax(probabilities))


def enables_threefold_claim(
    game: ChessGame,
    board: chess.Board,
    player: int,
    action: int,
) -> bool:
    """Return whether an action gives the opponent an immediate threefold claim."""
    child, _next_player = game.get_next_state(board, player, action)
    outcome = child.outcome(claim_draw=True)
    return outcome is not None and outcome.termination == chess.Termination.THREEFOLD_REPETITION


def non_repetition_actions(
    game: ChessGame,
    board: chess.Board,
    player: int,
    legal_mask: np.ndarray,
) -> list[int]:
    """Return legal root actions that do not immediately enable a threefold claim."""
    return [
        int(action)
        for action in np.flatnonzero(legal_mask)
        if not enables_threefold_claim(game, board, player, int(action))
    ]


@dataclass
class _EpisodeState:
    board: chess.Board
    player: int = 1
    step: int = 0
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    valid_moves: list[np.ndarray] = field(default_factory=list)
    policy_train_mask: list[bool] = field(default_factory=list)
    guard_attempts: int = 0
    guard_interventions: int = 0
    guard_forced_fallbacks: int = 0
    guard_excluded_actions: int = 0
    search_contempt_opponent_selections: int = 0
    search_contempt_thompson_selections: int = 0
    search_contempt_frozen_nodes: int = 0


@dataclass(frozen=True)
class _RootDecision:
    canonical_board: chess.Board
    policy: list[float]
    value: float
    observation: np.ndarray
    valid_moves: np.ndarray
    action: int
    search_plan: SelfPlaySearchPlan
    search_contempt: SearchContemptStats


def execute_episode(coach: Coach) -> Trajectory:
    """Run one self-play game using latent MCTS, collecting a full trajectory."""
    mcts = MCTS(coach.game, coach.nnet, coach.run)
    state = _EpisodeState(board=coach.game.get_init_board())
    while True:
        state.step += 1
        decision = _search_root(coach, mcts, state)
        decision = _guard_repetition(coach, mcts, state, decision)
        outcome = _apply_decision(coach.game, state, decision)
        if outcome is not None:
            return _terminal_trajectory(coach.game, state, outcome)
        if coach.run.max_ply is not None and state.step >= coach.run.max_ply:
            bootstrap_value = evaluate_truncation_bootstrap(coach, state.board, state.player)
            return _trajectory_from_state(
                state,
                terminal_value=0.0,
                truncated=True,
                truncation_bootstrap_value=bootstrap_value,
            )


def _search_root(coach: Coach, mcts: MCTS, state: _EpisodeState) -> _RootDecision:
    canonical = coach.game.get_canonical_form(state.board, state.player)
    search_plan = select_self_play_search_plan(coach.run)
    explore = search_plan.train_policy and self_play_exploration_enabled(state.board, state.step, coach.run)
    policy, value = mcts.search_latent(
        canonical,
        num_sims=search_plan.simulations,
        temp=1.0,
        add_exploration_noise=explore,
    )
    return _RootDecision(
        canonical_board=canonical,
        policy=policy,
        value=value,
        observation=coach.game.to_array(canonical),
        valid_moves=coach.game.get_valid_moves(canonical, 1),
        action=select_self_play_action(
            coach.run,
            policy,
            explore=explore,
            gumbel_proposal=mcts.last_action,
        ),
        search_plan=search_plan,
        search_contempt=mcts.last_search_contempt_stats,
    )


def _guard_repetition(
    coach: Coach,
    mcts: MCTS,
    state: _EpisodeState,
    decision: _RootDecision,
) -> _RootDecision:
    guard_enabled = coach.run.self_play_repetition_guard and enables_threefold_claim(
        coach.game,
        state.board,
        state.player,
        decision.action,
    )
    if not guard_enabled:
        return decision
    state.guard_attempts += 1
    safe_actions = non_repetition_actions(coach.game, state.board, state.player, decision.valid_moves)
    if not safe_actions:
        state.guard_forced_fallbacks += 1
        return decision
    return _search_safe_root(coach, mcts, state, decision, safe_actions)


def _search_safe_root(
    coach: Coach,
    mcts: MCTS,
    state: _EpisodeState,
    decision: _RootDecision,
    safe_actions: list[int],
) -> _RootDecision:
    state.guard_interventions += 1
    state.guard_excluded_actions += int(np.count_nonzero(decision.valid_moves)) - len(safe_actions)
    policy, value = mcts.search_latent(
        decision.canonical_board,
        num_sims=decision.search_plan.simulations,
        temp=1.0,
        add_exploration_noise=decision.search_plan.train_policy,
        allowed_root_actions=safe_actions,
    )
    action = select_self_play_action(
        coach.run,
        policy,
        explore=decision.search_plan.train_policy,
        gumbel_proposal=mcts.last_action,
    )
    return _RootDecision(
        canonical_board=decision.canonical_board,
        policy=policy,
        value=value,
        observation=decision.observation,
        valid_moves=decision.valid_moves,
        action=action,
        search_plan=decision.search_plan,
        search_contempt=mcts.last_search_contempt_stats,
    )


def _apply_decision(game: ChessGame, state: _EpisodeState, decision: _RootDecision) -> float | None:
    state.observations.append(decision.observation)
    state.policies.append(np.asarray(decision.policy, dtype=np.float32))
    state.values.append(decision.value)
    state.valid_moves.append(decision.valid_moves)
    state.policy_train_mask.append(decision.search_plan.train_policy)
    state.player = game.push_action(state.board, state.player, decision.action)
    state.actions.append(decision.action)
    _record_search_contempt(state, decision.search_contempt)
    return game.get_game_outcome(state.board, state.player)


def _record_search_contempt(state: _EpisodeState, stats: SearchContemptStats) -> None:
    state.search_contempt_opponent_selections += stats.opponent_selections
    state.search_contempt_thompson_selections += stats.thompson_selections
    state.search_contempt_frozen_nodes += stats.frozen_nodes


def _terminal_trajectory(game: ChessGame, state: _EpisodeState, outcome: float) -> Trajectory:
    terminal_outcome = state.board.outcome(claim_draw=game.claim_draw)
    if terminal_outcome is None:
        raise RuntimeError("A terminal self-play state has no chess outcome")
    return _trajectory_from_state(state, terminal_value=outcome, termination=terminal_outcome.termination)


def _trajectory_from_state(
    state: _EpisodeState,
    *,
    terminal_value: float,
    truncated: bool = False,
    truncation_bootstrap_value: float | None = None,
    termination: chess.Termination | None = None,
) -> Trajectory:
    return trajectory_with_terminal_rewards(
        state.observations,
        state.actions,
        state.policies,
        state.values,
        state.valid_moves,
        terminal_value,
        policy_train_mask=state.policy_train_mask,
        truncated=truncated,
        truncation_bootstrap_value=truncation_bootstrap_value,
        termination=termination,
        repetition_guard_attempts=state.guard_attempts,
        repetition_guard_interventions=state.guard_interventions,
        repetition_guard_forced_fallbacks=state.guard_forced_fallbacks,
        repetition_guard_excluded_actions=state.guard_excluded_actions,
        search_contempt_opponent_selections=state.search_contempt_opponent_selections,
        search_contempt_thompson_selections=state.search_contempt_thompson_selections,
        search_contempt_frozen_nodes=state.search_contempt_frozen_nodes,
    )


def evaluate_truncation_bootstrap(coach: Coach, board: chess.Board, player: int) -> float:
    canonical = coach.game.get_canonical_form(board, player)
    observations = np.expand_dims(coach.game.to_array(canonical), axis=0)
    valid_moves = np.expand_dims(coach.game.get_valid_moves(canonical, 1), axis=0)
    _policies, values, _latents = coach.nnet.batched_initial_inference(observations, valid_moves)
    return float(values[0])


def trajectory_with_terminal_rewards(
    observations: list[np.ndarray],
    actions: list[int],
    root_policies: list[np.ndarray],
    root_values: list[float],
    valids_list: list[np.ndarray],
    terminal_value_for_next_player: float,
    *,
    truncated: bool = False,
    truncation_bootstrap_value: float | None = None,
    termination: chess.Termination | None = None,
    repetition_guard_attempts: int = 0,
    repetition_guard_interventions: int = 0,
    repetition_guard_forced_fallbacks: int = 0,
    repetition_guard_excluded_actions: int = 0,
    search_contempt_opponent_selections: int = 0,
    search_contempt_thompson_selections: int = 0,
    search_contempt_frozen_nodes: int = 0,
    policy_train_mask: list[bool] | None = None,
) -> Trajectory:
    rewards = [0.0] * len(actions)
    rewards[-1] = -float(terminal_value_for_next_player)
    return Trajectory(
        observations=observations,
        actions=actions,
        rewards=rewards,
        root_policies=root_policies,
        root_values=root_values,
        valids=valids_list,
        policy_train_mask=policy_train_mask,
        truncated=truncated,
        truncation_bootstrap_value=truncation_bootstrap_value,
        termination=termination,
        repetition_guard_attempts=repetition_guard_attempts,
        repetition_guard_interventions=repetition_guard_interventions,
        repetition_guard_forced_fallbacks=repetition_guard_forced_fallbacks,
        repetition_guard_excluded_actions=repetition_guard_excluded_actions,
        search_contempt_opponent_selections=search_contempt_opponent_selections,
        search_contempt_thompson_selections=search_contempt_thompson_selections,
        search_contempt_frozen_nodes=search_contempt_frozen_nodes,
    )
