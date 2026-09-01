"""Tests for MCTS search (single-game and batched)."""

from typing import Never

import chess
import numpy as np
import pytest
import torch

from luna.config import EzV2LearnerConfig, MCTSParams
from luna.game.chess_game import ChessGame, move_to_action
from luna.mcts import (
    MCTS,
    BatchedMCTS,
)
from luna.network import LunaNetwork, RecurrentBatchResult


class _MateInOneNetwork:
    """Minimal inference stub; recurrent inference is forbidden for exact terminals."""

    def __init__(self, action_size: int, mate_action: int) -> None:
        self.action_size = action_size
        self.mate_action = mate_action
        self.recurrent_calls = 0
        self.recurrent_batch_sizes: list[int] = []

    def _policy(self, batch_size: int = 1) -> np.ndarray:
        policy = np.zeros((batch_size, self.action_size), dtype=np.float32)
        policy[:, self.mate_action] = 1.0
        return policy

    def predict_with_latent(
        self, _observation: np.ndarray, _valid: np.ndarray
    ) -> tuple[np.ndarray, float, torch.Tensor]:
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


class TestBatchedMCTS:
    def test_claimable_draw_root_returns_no_action_without_inference(self, chess_game: ChessGame) -> None:
        board = chess.Board()
        for move in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
            board.push_uci(move)
        canonical = chess_game.get_canonical_form(board, -1)
        assert canonical.can_claim_threefold_repetition()
        assert np.count_nonzero(chess_game.get_valid_moves(canonical, 1)) > 0

        class _NoInference:
            def batched_initial_inference(
                self,
                _observations: np.ndarray,
                _valids: np.ndarray,
            ) -> Never:
                raise AssertionError("terminal roots must not call initial inference")

        search = BatchedMCTS(chess_game, _NoInference(), MCTSParams())
        policy, value, _observation, valid = search.search_batch([canonical])[0]

        assert not np.any(policy)
        assert value == 0.0
        assert np.count_nonzero(valid) > 0
        assert search.last_actions == [None]

    def test_search_batch_returns_correct_count(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        nnet = LunaNetwork(chess_game, small_learner_config)
        params = MCTSParams(num_mcts_sims=3, dir_noise=False, recurrent_policy_topk=None)
        bmcts = BatchedMCTS(chess_game, nnet, params)

        boards = [chess_game.get_init_board() for _ in range(4)]
        canonicals = [chess_game.get_canonical_form(b, 1) for b in boards]
        results = bmcts.search_batch(canonicals, num_sims=3)

        assert len(results) == 4
        for probs, root_v, obs, valid in results:
            assert probs.shape == (chess_game.get_action_size(),)
            assert probs.dtype == np.float32
            assert abs(float(probs.sum()) - 1.0) < 1e-5
            assert isinstance(root_v, float)
            assert obs.shape == chess_game.get_board_size()
            assert valid.shape == (chess_game.get_action_size(),)
            assert np.count_nonzero(probs[valid == 0]) == 0

    def test_allowed_root_actions_restrict_search_but_preserve_full_legal_masks(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        network = LunaNetwork(chess_game, small_learner_config)
        search = BatchedMCTS(
            chess_game,
            network,
            MCTSParams(num_mcts_sims=3, dir_noise=False, recurrent_policy_topk=None),
        )
        boards = [chess_game.get_init_board(), chess_game.get_init_board()]
        allowed_action = move_to_action(chess.Move.from_uci("e2e4"))
        other_legal_action = move_to_action(chess.Move.from_uci("g1f3"))
        expected_legal = chess_game.get_valid_moves(boards[0], 1)

        results = search.search_batch(
            boards,
            num_sims=3,
            add_exploration_noise=False,
            allowed_root_actions=[{allowed_action}, None],
        )

        restricted_policy, _restricted_value, _restricted_obs, restricted_valid = results[0]
        unrestricted_policy, _unrestricted_value, _unrestricted_obs, unrestricted_valid = results[1]
        np.testing.assert_array_equal(restricted_valid, expected_legal)
        np.testing.assert_array_equal(unrestricted_valid, expected_legal)
        assert restricted_valid[other_legal_action] == 1.0
        assert restricted_policy[other_legal_action] == 0.0
        assert set(np.flatnonzero(restricted_policy)) == {allowed_action}
        assert search.last_actions[0] == allowed_action
        assert np.count_nonzero(unrestricted_policy[expected_legal == 0]) == 0

        with pytest.raises(ValueError, match="one entry per batched root"):
            search.search_batch(boards, allowed_root_actions=[{allowed_action}])

    @pytest.mark.parametrize("temperature", [1e-8, np.nextafter(0.0, 1.0)])
    def test_batch_puct_tiny_positive_temperature_returns_finite_policy(
        self,
        chess_game: ChessGame,
        temperature: float,
    ) -> None:
        board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        mate_action = move_to_action(chess.Move.from_uci("g6g7"))
        network = _MateInOneNetwork(chess_game.get_action_size(), mate_action)
        params = MCTSParams(num_mcts_sims=2, search_mode="puct", dir_noise=False)

        policy = BatchedMCTS(chess_game, network, params).search_batch(
            [board],
            temp=temperature,
            add_exploration_noise=False,
        )[0][0]

        assert np.isfinite(policy).all()
        assert float(policy.sum()) == pytest.approx(1.0)
        assert int(np.argmax(policy)) == mate_action

    def test_batch_gumbel_actions_and_improved_targets_are_separate(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        nnet = LunaNetwork(chess_game, small_learner_config)
        params = MCTSParams(num_mcts_sims=3, dir_noise=False, recurrent_policy_topk=None)
        bmcts = BatchedMCTS(chess_game, nnet, params)

        boards = [chess_game.get_init_board(), chess_game.get_init_board()]
        canonicals = [chess_game.get_canonical_form(b, 1) for b in boards]
        first = bmcts.search_batch(
            canonicals,
            num_sims=3,
            temp=1,
            add_exploration_noise=False,
        )
        first_actions = bmcts.last_actions.copy()
        second = bmcts.search_batch(
            canonicals,
            num_sims=3,
            temp=1,
            add_exploration_noise=False,
        )
        second_actions = bmcts.last_actions.copy()
        action_results = bmcts.search_batch(canonicals, num_sims=3, temp=0)

        assert first_actions == second_actions == bmcts.last_actions
        for index, (first_result, second_result) in enumerate(zip(first, second, strict=True)):
            first_policy = first_result[0]
            second_policy = second_result[0]
            np.testing.assert_allclose(first_policy, second_policy, atol=0.0)
            assert int((first_policy > 0).sum()) > 1
            assert int((action_results[index][0] > 0).sum()) == 1
            assert int(np.argmax(action_results[index][0])) == first_actions[index]

    def test_single_and_batch_gumbel_search_match_at_evaluation(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        np.random.seed(7)
        torch.manual_seed(7)
        nnet = LunaNetwork(chess_game, small_learner_config)
        params = MCTSParams(num_mcts_sims=8, dir_noise=False, recurrent_policy_topk=None)
        board = chess_game.get_init_board()

        single_action = MCTS(chess_game, nnet, params)
        single_policy, single_value = single_action.search_latent(board, temp=1, add_exploration_noise=False)
        batched = BatchedMCTS(chess_game, nnet, params)
        batch_policy, batch_value, _obs, _valid = batched.search_batch([board], temp=1, add_exploration_noise=False)[0]

        np.testing.assert_allclose(batch_policy, single_policy, rtol=0.0, atol=1e-7)
        assert abs(batch_value - single_value) < 1e-7
        assert batched.last_actions == [single_action.last_action]
