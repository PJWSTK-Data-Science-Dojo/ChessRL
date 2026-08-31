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
    _backup_latent_path,
    _completed_qvalues,
    _get_sequence_of_considered_visits,
    _gumbel_improved_policy,
    _GumbelRootState,
    _LatentNode,
)
from luna.network import LunaNetwork, RecurrentBatchResult
from luna.profiling import SelfPlayMCTSTimings


def _root_with_priors(priors: list[float], actions: list[int] | None = None) -> _LatentNode:
    root = _LatentNode(prior=0.0)
    root.expanded = True
    root.raw_value = 0.0
    if actions is None:
        actions = list(range(len(priors)))
    for action, prior in zip(actions, priors):
        root.children[action] = _LatentNode(prior=prior)
    return root


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


class TestLatentSearch:
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
        search_mode: str,
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
        for index, (first_result, second_result) in enumerate(zip(first, second)):
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

    def test_batch_mate_in_one_skips_recurrent_inference(self, chess_game: ChessGame) -> None:
        board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        mate_action = move_to_action(chess.Move.from_uci("g6g7"))
        network = _MateInOneNetwork(chess_game.get_action_size(), mate_action)
        params = MCTSParams(num_mcts_sims=1, gumbel_max_considered_actions=1)

        policy, value, _obs, _valid = BatchedMCTS(chess_game, network, params).search_batch(
            [board], num_sims=1, temp=0
        )[0]

        assert int(np.argmax(policy)) == mate_action
        assert value == 1.0
        assert network.recurrent_calls == 0

    def test_mixed_terminal_batch_only_infers_nonterminal_leaves(self, chess_game: ChessGame) -> None:
        mate_board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        mate_action = move_to_action(chess.Move.from_uci("g6g7"))
        network = _MateInOneNetwork(chess_game.get_action_size(), mate_action)
        params = MCTSParams(
            num_mcts_sims=1,
            gumbel_max_considered_actions=1,
            recurrent_policy_topk=None,
        )

        results = BatchedMCTS(chess_game, network, params).search_batch(
            [mate_board, chess_game.get_init_board()], num_sims=1, temp=0
        )

        assert results[0][1] == 1.0
        assert int(np.argmax(results[0][0])) == mate_action
        assert abs(float(results[1][0].sum()) - 1.0) < 1e-6
        assert network.recurrent_calls == 1
        assert network.recurrent_batch_sizes == [1]

    def test_batch_tree_keeps_native_turns_after_root(
        self,
        chess_game: ChessGame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        network = _MateInOneNetwork(chess_game.get_action_size(), mate_action=0)
        params = MCTSParams(
            num_mcts_sims=2,
            gumbel_max_considered_actions=1,
            recurrent_policy_topk=None,
        )

        def reject_recanonicalization(_board: chess.Board, _player: int) -> Never:
            raise AssertionError("MCTS descendants must retain their native board turn")

        monkeypatch.setattr(chess_game, "get_canonical_form", reject_recanonicalization)

        results = BatchedMCTS(chess_game, network, params).search_batch(
            [chess_game.get_init_board()],
            num_sims=2,
            temp=0,
            add_exploration_noise=False,
        )

        assert len(results) == 1
        assert network.recurrent_calls == 2

    def test_batch_exploration_noise_accepts_per_root_flags(
        self,
        chess_game: ChessGame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        mate_action = move_to_action(chess.Move.from_uci("g6g7"))
        network = _MateInOneNetwork(chess_game.get_action_size(), mate_action)
        params = MCTSParams(num_mcts_sims=1, gumbel_max_considered_actions=1)
        batched = BatchedMCTS(chess_game, network, params)
        legal_count = int(np.count_nonzero(chess_game.get_valid_moves(board, 1)))
        calls: list[int] = []

        def fake_gumbel(*, size: int) -> np.ndarray:
            calls.append(size)
            return np.zeros(size, dtype=np.float64)

        monkeypatch.setattr(np.random, "gumbel", fake_gumbel)

        batched.search_batch(
            [board, board],
            num_sims=1,
            temp=1,
            add_exploration_noise=[True, False],
        )
        assert calls == [legal_count]

        calls.clear()
        batched.search_batch(
            [board, board],
            num_sims=1,
            temp=1,
            add_exploration_noise=True,
        )
        assert calls == [legal_count, legal_count]

        with pytest.raises(ValueError, match="one flag per batched root"):
            batched.search_batch(
                [board, board],
                num_sims=1,
                temp=1,
                add_exploration_noise=[True],
            )

    def test_profile_separates_rule_expansion_from_recurrent_inference(
        self,
        chess_game: ChessGame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        preferred_action = move_to_action(chess.Move.from_uci("e2e4"))
        network = _MateInOneNetwork(chess_game.get_action_size(), preferred_action)
        timings = SelfPlayMCTSTimings()
        clock = 0.0
        original_next_state = chess_game.get_next_search_state
        original_recurrent = network.batched_recurrent_inference

        def timed_next_state(board: chess.Board, player: int, action: int) -> tuple[chess.Board, int]:
            nonlocal clock
            clock += 3.0
            return original_next_state(board, player, action)

        def timed_recurrent(
            latents: torch.Tensor,
            actions: list[int],
            *,
            valid_masks: list[np.ndarray | None],
            policy_topk: int | None,
        ) -> RecurrentBatchResult:
            nonlocal clock
            clock += 5.0
            return original_recurrent(
                latents,
                actions,
                valid_masks=valid_masks,
                policy_topk=policy_topk,
            )

        monkeypatch.setattr("luna.mcts.time.perf_counter", lambda: clock)
        monkeypatch.setattr(chess_game, "get_next_search_state", timed_next_state)
        monkeypatch.setattr(network, "batched_recurrent_inference", timed_recurrent)

        BatchedMCTS(
            chess_game,
            network,
            MCTSParams(num_mcts_sims=1, gumbel_max_considered_actions=1),
            timings=timings,
        ).search_batch([chess_game.get_init_board()], num_sims=1)

        assert timings.expand_backup_s == pytest.approx(3.0)
        assert timings.recurrent_inf_s == pytest.approx(5.0)


class TestGumbelMuZero:
    def test_default_mode_is_gumbel(self) -> None:
        assert MCTSParams().search_mode == "gumbel"

    def test_sequential_halving_respects_budget_and_top_m(self) -> None:
        params = MCTSParams(
            search_mode="gumbel",
            gumbel_max_considered_actions=4,
            gumbel_scale=1.0,
        )
        root = _root_with_priors([0.40, 0.30, 0.20, 0.09, 0.005, 0.005])
        state = _GumbelRootState(root, 12, params, add_noise=False)
        quality = [1.0, 0.5, 0.0, -0.5, -1.0, -1.0]

        assert state.schedule == _get_sequence_of_considered_visits(4, 12)
        assert state.schedule == (0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4)
        assert not np.any(state.gumbel)

        for _ in range(12):
            action = state.select_action(root)
            child = root.children[action]
            child.visit_count += 1
            child.value_sum += quality[action]
            root.total_child_visits += 1

        visits = np.array([child.visit_count for child in root.children.values()])
        assert int(visits.sum()) == 12
        np.testing.assert_array_equal(visits, [5, 5, 1, 1, 0, 0])
        assert state.proposed_action(root) == 0

    def test_completed_q_policy_is_soft_legal_and_improved(self) -> None:
        params = MCTSParams(
            gumbel_value_scale=0.1,
            gumbel_maxvisit_init=50.0,
        )
        root = _root_with_priors([0.5, 0.3, 0.2], actions=[2, 5, 7])
        root.raw_value = 0.1
        root.children[2].visit_count = 2
        root.children[2].value_sum = 1.6
        root.children[5].visit_count = 1
        root.children[5].value_sum = -0.2

        _actions, _priors, _visits, transformed_q = _completed_qvalues(root, params)
        policy = _gumbel_improved_policy(root, action_size=10, params=params)

        np.testing.assert_allclose(transformed_q, [5.2, 0.0, 2.8275], atol=1e-10)
        assert abs(float(policy.sum()) - 1.0) < 1e-6
        assert np.all(policy[[2, 5, 7]] > 0)
        assert np.count_nonzero(np.delete(policy, [2, 5, 7])) == 0
        assert policy[2] > 0.5
        assert np.count_nonzero(policy) == 3

    def test_backup_alternates_player_perspective_at_every_depth(self) -> None:
        root = _LatentNode(0.0)
        middle = _LatentNode(1.0)
        parent = _LatentNode(1.0)
        leaf = _LatentNode(1.0)
        middle.reward = 0.1
        parent.reward = 0.2

        _backup_latent_path([root, middle, parent], leaf, q_leaf=0.4, discount=0.5)

        assert leaf.value() == 0.4
        assert parent.value() == 0.0
        assert middle.value() == 0.1
        assert root.total_child_visits == 1
