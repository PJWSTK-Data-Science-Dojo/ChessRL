"""Tests for MCTS search (single-game and batched)."""

from typing import Never

import chess
import numpy as np
import pytest
import torch

from luna.config import MCTSParams
from luna.game.chess_game import ChessGame, move_to_action
from luna.mcts import (
    BatchedMCTS,
)
from luna.network import RecurrentBatchResult
from luna.profiling import SelfPlayMCTSTimings


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
