"""Tests for Coach self-play (e.g. max ply truncation, batched self-play)."""

from collections.abc import Sequence
from pathlib import Path

import chess
import numpy as np
import pytest

from luna.coach import Coach
from luna.config import EzV2LearnerConfig, MCTSParams, TrainingRunConfig
from luna.game.arena import Arena
from luna.game.chess_game import ChessGame, move_to_action
from luna.game.stockfish_eval import StockfishEvalScores, StockfishEvalSkipped
from luna.network import LunaNetwork
from luna.profiling import SelfPlayMCTSTimings


class TestMaxPlyTruncation:
    def test_execute_episode_stops_at_max_ply_with_zero_draw_reward(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        nnet = LunaNetwork(chess_game, small_learner_config)
        run = TrainingRunConfig(
            num_mcts_sims=2,
            max_ply=5,
            dir_noise=False,
            temp_threshold=1,
            recurrent_policy_topk=None,
        )
        coach = Coach(chess_game, nnet, run)
        traj = coach.execute_episode()

        assert len(traj.actions) == 5
        assert len(traj.rewards) == 5
        assert all(r == 0.0 for r in traj.rewards[:-1])
        assert np.isclose(traj.rewards[-1], 0.0)


class TestBatchedSelfPlay:
    def test_execute_episodes_batched_returns_trajectories(
        self,
        chess_game: ChessGame,
        small_learner_config: EzV2LearnerConfig,
    ) -> None:
        nnet = LunaNetwork(chess_game, small_learner_config)
        run = TrainingRunConfig(
            num_mcts_sims=2,
            max_ply=5,
            dir_noise=False,
            temp_threshold=1,
            parallel_games=2,
            recurrent_policy_topk=None,
        )
        coach = Coach(chess_game, nnet, run)
        trajs = coach.execute_episodes_batched(num_episodes=3)

        assert len(trajs) == 3
        for t in trajs:
            assert t.game_length > 0
            assert t.game_length <= 5
            assert t.observations.shape[0] == t.game_length


def test_gumbel_selfplay_executes_proposal_but_stores_improved_target(
    monkeypatch: pytest.MonkeyPatch,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    target_action = move_to_action(chess.Move.from_uci("d2d4"))
    proposed_action = move_to_action(chess.Move.from_uci("e2e4"))

    class _Search:
        def __init__(self, _game: ChessGame, _network: LunaNetwork, _params: MCTSParams) -> None:
            self.last_action = proposed_action

        def search_latent(
            self,
            _board: chess.Board,
            temp: float,
            *,
            add_exploration_noise: bool | None,
        ) -> tuple[np.ndarray, float]:
            assert temp == 1.0
            assert add_exploration_noise is True
            policy = np.zeros(chess_game.get_action_size(), dtype=np.float32)
            policy[target_action] = 1.0
            return policy, 0.0

    monkeypatch.setattr("luna.coach.MCTS", _Search)
    network = LunaNetwork(chess_game, small_learner_config)
    coach = Coach(
        chess_game,
        network,
        TrainingRunConfig(num_mcts_sims=1, max_ply=1, temp_threshold=2),
    )

    trajectory = coach.execute_episode()

    assert trajectory.actions.tolist() == [proposed_action]
    assert int(np.argmax(trajectory.root_policies[0])) == target_action


def test_batched_gumbel_selfplay_routes_per_root_exploration_and_proposals(
    monkeypatch: pytest.MonkeyPatch,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    target_action = move_to_action(chess.Move.from_uci("d2d4"))
    proposed_action = move_to_action(chess.Move.from_uci("e2e4"))

    class _BatchedSearch:
        def __init__(
            self,
            game: ChessGame,
            _network: LunaNetwork,
            _params: MCTSParams,
            timings: SelfPlayMCTSTimings | None = None,
        ) -> None:
            del timings
            self.game = game
            self.last_actions: list[int | None] = []

        def search_batch(
            self,
            boards: list[chess.Board],
            temp: float,
            *,
            add_exploration_noise: bool | Sequence[bool] | None,
        ) -> list[tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
            assert temp == 1.0
            assert add_exploration_noise == [True]
            self.last_actions = [proposed_action]
            outputs = []
            for board in boards:
                policy = np.zeros(self.game.get_action_size(), dtype=np.float32)
                policy[target_action] = 1.0
                outputs.append(
                    (
                        policy,
                        0.0,
                        self.game.to_array(board),
                        self.game.get_valid_moves(board, 1),
                    )
                )
            return outputs

    monkeypatch.setattr("luna.coach.BatchedMCTS", _BatchedSearch)
    network = LunaNetwork(chess_game, small_learner_config)
    coach = Coach(
        chess_game,
        network,
        TrainingRunConfig(
            num_mcts_sims=1,
            max_ply=1,
            temp_threshold=2,
            parallel_games=1,
        ),
    )

    trajectory = coach.execute_episodes_batched(1)[0]

    assert trajectory.actions.tolist() == [proposed_action]
    assert int(np.argmax(trajectory.root_policies[0])) == target_action


class TestArenaMaxPly:
    def test_play_game_returns_draw_when_max_ply_reached(self, chess_game: ChessGame) -> None:
        def pick_first(canonical_board: chess.Board) -> int:
            valids = chess_game.get_valid_moves(canonical_board, 1)
            return int(np.argmax(valids))

        arena = Arena(pick_first, pick_first, chess_game)
        result = arena.play_game(verbose=False, max_ply=3)
        assert result == 0.0


def test_checkpoint_retention_keeps_top_k(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    tmp_path: Path,
) -> None:
    run = TrainingRunConfig(
        num_mcts_sims=2,
        dir_noise=False,
        checkpoint=str(tmp_path.resolve()),
        checkpoint_top_k=2,
        recurrent_policy_topk=None,
    )
    nnet = LunaNetwork(chess_game, small_learner_config)
    coach = Coach(chess_game, nnet, run)
    coach._publish_checkpoint(1)
    coach._publish_checkpoint(2)
    coach._publish_checkpoint(3)

    assert not (tmp_path / "checkpoint_1.pth.tar").is_file()
    assert (tmp_path / "checkpoint_2.pth.tar").is_file()
    assert (tmp_path / "checkpoint_3.pth.tar").is_file()


def test_corrupt_best_evaluation_metadata_fails_loudly(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    run = TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0)
    coach = Coach(chess_game, LunaNetwork(chess_game, small_learner_config), run)
    coach._publish_checkpoint(1)
    (tmp_path / "best_eval.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="external-evaluation metadata"):
        coach._update_best_from_stockfish(1, StockfishEvalScores(model_wins=1, draws=1, stockfish_wins=0))
    assert (tmp_path / "latest.pth.tar").is_file()
    assert not (tmp_path / "best.pth.tar").exists()


def test_configured_external_evaluation_failure_stops_promotion(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    run = TrainingRunConfig(checkpoint=str(tmp_path))
    coach = Coach(chess_game, LunaNetwork(chess_game, small_learner_config), run)

    with pytest.raises(RuntimeError, match=r"External evaluation did not complete.*no_engine"):
        coach._update_best_from_stockfish(1, StockfishEvalSkipped("no_engine", "binary not found"))

    assert not (tmp_path / "best.pth.tar").exists()
