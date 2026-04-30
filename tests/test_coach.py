"""Tests for Coach self-play (e.g. max ply truncation, batched self-play)."""

from __future__ import annotations

import numpy as np

from luna.coach import Coach
from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.game.arena import Arena
from luna.game.chess_game import DRAW_VALUE, ChessGame
from luna.network import LunaNetwork


def _small_learner() -> EzV2LearnerConfig:
    return EzV2LearnerConfig(num_channels=32, repr_blocks=2, dyn_blocks=1, proj_dim=64)


class TestMaxPlyTruncation:
    def test_execute_episode_stops_at_max_ply_with_draw_reward(self, monkeypatch) -> None:
        game = ChessGame()
        monkeypatch.setattr(game, "get_game_ended", lambda board, player: 0.0)

        nnet = LunaNetwork(game, _small_learner())
        run = TrainingRunConfig(
            num_mcts_sims=2,
            max_ply=5,
            dir_noise=False,
            temp_threshold=1,
            recurrent_policy_topk=None,
        )
        coach = Coach(game, nnet, run)
        traj = coach.execute_episode()

        assert len(traj.actions) == 5
        assert len(traj.rewards) == 5
        assert all(r == 0.0 for r in traj.rewards[:-1])
        assert np.isclose(traj.rewards[-1], -DRAW_VALUE)


class TestBatchedSelfPlay:
    def test_execute_episodes_batched_returns_trajectories(self) -> None:
        game = ChessGame()
        nnet = LunaNetwork(game, _small_learner())
        run = TrainingRunConfig(
            num_mcts_sims=2,
            max_ply=5,
            dir_noise=False,
            temp_threshold=1,
            parallel_games=2,
            recurrent_policy_topk=None,
        )
        coach = Coach(game, nnet, run)
        trajs = coach.execute_episodes_batched(num_episodes=3)

        assert len(trajs) == 3
        for t in trajs:
            assert t.game_length > 0
            assert t.game_length <= 5
            assert t.observations.shape[0] == t.game_length


class TestArenaBatched:
    def test_play_arena_games_batched_returns_expected_count(self) -> None:
        game = ChessGame()
        nnet = LunaNetwork(game, _small_learner())
        run = TrainingRunConfig(
            num_mcts_sims=2,
            max_ply=6,
            dir_noise=False,
            arena_num_mcts_sims=2,
        )
        coach = Coach(game, nnet, run)
        params = Coach._arena_mcts_params(run)
        assert params.num_mcts_sims == 2
        assert params.dir_noise is False
        results = coach._play_arena_games_batched(nnet, nnet, params, num_games=3)
        assert len(results) == 3
        for r in results:
            assert -1.0 <= r <= 1.0


class TestArenaMaxPly:
    def test_play_game_returns_draw_when_max_ply_reached(self) -> None:
        game = ChessGame()

        def pick_first(canonical_board):
            valids = game.get_valid_moves(canonical_board, 1)
            return int(np.argmax(valids))

        arena = Arena(pick_first, pick_first, game)
        result = arena.play_game(verbose=False, max_ply=3)
        assert result == 0.0
