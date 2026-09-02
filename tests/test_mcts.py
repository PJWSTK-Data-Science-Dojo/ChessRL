"""Tests for MCTS search (single-game and batched)."""

from typing import Literal, Never

import chess
import numpy as np
import pytest
import torch

from luna.config import EzV2LearnerConfig, MCTSParams
from luna.game.chess_game import ChessGame, move_to_action
from luna.mcts import (
    MCTS,
    _LatentNode,
)
from luna.network import LunaNetwork, RecurrentBatchResult


def _root_with_priors(priors: list[float], actions: list[int] | None = None) -> _LatentNode:
    root = _LatentNode(prior=0.0)
    root.expanded = True
    root.raw_value = 0.0
    if actions is None:
        actions = list(range(len(priors)))
    for action, prior in zip(actions, priors, strict=True):
        root.children[action] = _LatentNode(prior=prior)
    return root


class _MateInOneNetwork:
    """Minimal inference stub; recurrent inference is forbidden for exact terminals."""

    def __init__(self, action_size: int, mate_action: int) -> None:
        self.action_size = action_size
        self.mate_action = mate_action
        self.recurrent_calls = 0
        self.recurrent_batch_sizes: list[int] = []
        self.observation_batches: list[np.ndarray] = []

    def _policy(self, batch_size: int = 1) -> np.ndarray:
        policy = np.zeros((batch_size, self.action_size), dtype=np.float32)
        policy[:, self.mate_action] = 1.0
        return policy

    def predict_with_latent(
        self, observation: np.ndarray, _valid: np.ndarray
    ) -> tuple[np.ndarray, float, torch.Tensor]:
        self.observation_batches.append(observation[None].copy())
        return self._policy()[0], 0.0, torch.zeros((1, 1, 1, 1))

    def recurrent_predict(
        self,
        _latent: torch.Tensor,
        _action: int,
        _valid_mask: np.ndarray | None = None,
    ) -> Never:
        self.recurrent_calls += 1
        raise AssertionError("terminal transitions must not call recurrent inference")

    def batched_initial_inference(
        self, observations: np.ndarray, valids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        batch_size = observations.shape[0]
        policies = np.zeros((batch_size, self.action_size), dtype=np.float32)
        for i, valid in enumerate(valids):
            legal_actions = np.flatnonzero(valid)
            action = self.mate_action if valid[self.mate_action] > 0.0 else int(legal_actions[0])
            policies[i, action] = 1.0
        return (
            policies,
            np.zeros(batch_size, dtype=np.float32),
            torch.zeros((batch_size, 1, 1, 1)),
        )

    def batched_recurrent_inference(
        self,
        latents: torch.Tensor,
        _actions: list[int],
        *,
        valid_masks: list[np.ndarray | None],
        policy_topk: int | None,
    ) -> RecurrentBatchResult:
        del policy_topk
        self.recurrent_calls += 1
        batch_size = latents.shape[0]
        self.recurrent_batch_sizes.append(batch_size)
        policies = np.zeros((batch_size, self.action_size), dtype=np.float32)
        for i, valid in enumerate(valid_masks):
            legal_actions = np.arange(self.action_size) if valid is None else np.flatnonzero(valid)
            policies[i, legal_actions] = 1.0 / len(legal_actions)
        return RecurrentBatchResult(
            policy_full=policies,
            topk_indices=None,
            topk_probs=None,
            values=np.zeros(batch_size, dtype=np.float32),
            rewards=np.zeros(batch_size, dtype=np.float32),
            next_latent=torch.zeros((batch_size, 1, 1, 1)),
        )


class TestLatentSearch:
    def test_exact_state_expansion_preserves_temporal_observation(self, chess_game: ChessGame) -> None:
        board = chess_game.get_init_board()
        board.push_uci("e2e4")
        board = chess_game.get_canonical_form(board, -1)
        action = move_to_action(chess.Move.from_uci("e2e4"))
        network = _MateInOneNetwork(chess_game.get_action_size(), action)
        params = MCTSParams(
            num_mcts_sims=1,
            gumbel_max_considered_actions=1,
            tree_state_mode="exact",
        )

        MCTS(chess_game, network, params).search_latent(
            board,
            temp=0.0,
            add_exploration_noise=False,
        )

        child, child_player = chess_game.get_next_state(board, 1, action)
        canonical_child = chess_game.get_canonical_form(child, child_player)
        np.testing.assert_array_equal(network.observation_batches[-1][0], chess_game.to_array(canonical_child))
        assert len(network.observation_batches) == 2
        assert network.recurrent_calls == 0

    def test_returns_valid_policy(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        nnet = LunaNetwork(chess_game, small_learner_config)
        params = MCTSParams(num_mcts_sims=3, dir_noise=False, recurrent_policy_topk=None)
        mcts = MCTS(chess_game, nnet, params)

        board = chess_game.get_init_board()
        canonical = chess_game.get_canonical_form(board, 1)
        probs, root_v = mcts.search_latent(canonical, num_sims=3)

        assert len(probs) == chess_game.get_action_size()
        assert abs(sum(probs) - 1.0) < 1e-5
        assert isinstance(root_v, float)
        valid = chess_game.get_valid_moves(canonical, 1)
        assert np.count_nonzero(np.asarray(probs)[valid == 0]) == 0

    def test_allowed_root_actions_constrain_only_root_policy(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        network = LunaNetwork(chess_game, small_learner_config)
        search = MCTS(
            chess_game,
            network,
            MCTSParams(num_mcts_sims=3, dir_noise=False, recurrent_policy_topk=None),
        )
        allowed = {
            move_to_action(chess.Move.from_uci("e2e4")),
            move_to_action(chess.Move.from_uci("d2d4")),
        }

        policy, _value = search.search_latent(
            chess_game.get_init_board(),
            num_sims=3,
            allowed_root_actions=allowed,
        )

        assert search.last_action in allowed
        assert {int(action) for action in np.flatnonzero(policy)} <= allowed
        assert abs(sum(policy) - 1.0) < 1e-6

        with pytest.raises(ValueError, match="Root action must be"):
            search.search_latent(
                chess_game.get_init_board(),
                num_sims=1,
                allowed_root_actions={chess_game.get_action_size()},
            )

    def test_get_action_prob_uses_latent_search(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        nnet = LunaNetwork(chess_game, small_learner_config)
        params = MCTSParams(num_mcts_sims=3, dir_noise=False, recurrent_policy_topk=None)
        mcts = MCTS(chess_game, nnet, params)

        board = chess_game.get_init_board()
        canonical = chess_game.get_canonical_form(board, 1)
        p_latent, _ = mcts.search_latent(canonical, num_sims=3, temp=1.0)
        p_get = mcts.get_action_prob(canonical, temp=1.0)

        assert len(p_latent) == len(p_get)
        assert abs(sum(p_get) - 1.0) < 1e-5

    def test_gumbel_action_and_improved_target_are_separate(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        nnet = LunaNetwork(chess_game, small_learner_config)
        params = MCTSParams(num_mcts_sims=3, dir_noise=False, recurrent_policy_topk=None)
        mcts = MCTS(chess_game, nnet, params)

        board = chess_game.get_init_board()
        canonical = chess_game.get_canonical_form(board, 1)
        first = np.asarray(mcts.get_action_prob(canonical, temp=1, add_exploration_noise=False))
        first_action = mcts.last_action
        second = np.asarray(mcts.get_action_prob(canonical, temp=1, add_exploration_noise=False))
        second_action = mcts.last_action
        action_policy = np.asarray(mcts.get_action_prob(canonical, temp=0))

        np.testing.assert_allclose(first, second, atol=0.0)
        assert np.count_nonzero(first) > 1
        assert first_action == second_action == mcts.last_action
        assert np.count_nonzero(action_policy) == 1
        assert int(np.argmax(action_policy)) == first_action

    def test_puct_temp_zero_remains_one_hot(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        nnet = LunaNetwork(chess_game, small_learner_config)
        params = MCTSParams(
            num_mcts_sims=3,
            search_mode="puct",
            dir_noise=False,
            recurrent_policy_topk=None,
        )
        mcts = MCTS(chess_game, nnet, params)

        probs = mcts.get_action_prob(chess_game.get_init_board(), temp=0)

        assert sum(1 for probability in probs if probability > 0) == 1

    @pytest.mark.parametrize("temperature", [1e-8, np.nextafter(0.0, 1.0)])
    def test_puct_tiny_positive_temperature_returns_finite_policy(
        self,
        chess_game: ChessGame,
        temperature: float,
    ) -> None:
        board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        mate_action = move_to_action(chess.Move.from_uci("g6g7"))
        network = _MateInOneNetwork(chess_game.get_action_size(), mate_action)
        params = MCTSParams(num_mcts_sims=2, search_mode="puct", dir_noise=False)

        policy = np.asarray(
            MCTS(chess_game, network, params).get_action_prob(
                board,
                temp=temperature,
                add_exploration_noise=False,
            )
        )

        assert np.isfinite(policy).all()
        assert float(policy.sum()) == pytest.approx(1.0)
        assert int(np.argmax(policy)) == mate_action

    def test_terminal_root_short_circuits_initial_inference(self, chess_game: ChessGame) -> None:
        terminal = chess.Board("7K/6q1/6k1/8/8/8/8/8 w - - 0 1")

        class _NoInference:
            def predict_with_latent(
                self,
                _observation: np.ndarray,
                _valid: np.ndarray,
            ) -> Never:
                raise AssertionError("terminal roots must not call initial inference")

        policy, value = MCTS(chess_game, _NoInference(), MCTSParams()).search_latent(terminal)

        assert not any(policy)
        assert value == -1.0

    def test_claimable_draw_by_next_move_short_circuits_initial_inference(self, chess_game: ChessGame) -> None:
        board = chess.Board()
        for move in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
            board.push_uci(move)
        canonical = chess_game.get_canonical_form(board, -1)

        class _NoInference:
            def predict_with_latent(
                self,
                _observation: np.ndarray,
                _valid: np.ndarray,
            ) -> Never:
                raise AssertionError("claimable roots must not call initial inference")

        policy, value = MCTS(chess_game, _NoInference(), MCTSParams()).search_latent(canonical)

        assert not any(policy)
        assert value == 0.0

    def test_mate_in_one_uses_exact_terminal_value_without_recurrent_inference(
        self,
        chess_game: ChessGame,
    ) -> None:
        board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        mate_action = move_to_action(chess.Move.from_uci("g6g7"))
        network = _MateInOneNetwork(chess_game.get_action_size(), mate_action)
        params = MCTSParams(num_mcts_sims=1, gumbel_max_considered_actions=1)

        policy, value = MCTS(chess_game, network, params).search_latent(board, num_sims=1, temp=0)

        assert int(np.argmax(policy)) == mate_action
        assert value == 1.0
        assert network.recurrent_calls == 0

    @pytest.mark.parametrize("search_mode", ["gumbel", "puct"])
    def test_immediate_stop_uses_root_prediction(
        self,
        chess_game: ChessGame,
        search_mode: Literal["gumbel", "puct"],
    ) -> None:
        board = chess_game.get_init_board()
        preferred_action = move_to_action(chess.Move.from_uci("e2e4"))

        class _RootOnlyNetwork(_MateInOneNetwork):
            def predict_with_latent(
                self,
                _observation: np.ndarray,
                _valid: np.ndarray,
            ) -> tuple[np.ndarray, float, torch.Tensor]:
                return self._policy()[0], 0.625, torch.zeros((1, 1, 1, 1))

        network = _RootOnlyNetwork(chess_game.get_action_size(), preferred_action)
        search = MCTS(
            chess_game,
            network,
            MCTSParams(num_mcts_sims=8, search_mode=search_mode, dir_noise=False),
        )

        policy, root_value = search.search_latent(
            board,
            temp=0.0,
            add_exploration_noise=False,
            should_stop=lambda: True,
        )

        assert search.last_simulations == 0
        assert search.last_action == preferred_action
        assert int(np.argmax(policy)) == preferred_action
        assert root_value == pytest.approx(0.625)
