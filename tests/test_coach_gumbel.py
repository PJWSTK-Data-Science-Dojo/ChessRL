"""Tests for Coach self-play (e.g. max ply truncation, batched self-play)."""

from collections.abc import Sequence

import chess
import numpy as np
import pytest

from luna.coach import (
    Coach,
)
from luna.config import EzV2LearnerConfig, MCTSParams, TrainingRunConfig
from luna.game.arena import Arena
from luna.game.chess_game import ChessGame, move_to_action
from luna.mcts_search_contempt import SearchContemptStats
from luna.network import LunaNetwork
from luna.profiling import SelfPlayMCTSTimings


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
            self.last_search_contempt_stats = SearchContemptStats()

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

    monkeypatch.setattr("luna.coach_self_play.MCTS", _Search)
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
            self.last_search_contempt_stats: list[SearchContemptStats] = []

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
            self.last_search_contempt_stats = [SearchContemptStats()]
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

    monkeypatch.setattr("luna.coach_batched_self_play.BatchedMCTS", _BatchedSearch)
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

    def test_initial_board_selects_its_side_to_move_without_mutating_caller(self, chess_game: ChessGame) -> None:
        initial_board = chess.Board()
        initial_board.push_uci("e2e4")
        initial_fen = initial_board.fen()
        called_players: list[int] = []

        def player_one(canonical_board: chess.Board) -> int:
            called_players.append(1)
            return int(np.argmax(chess_game.get_valid_moves(canonical_board, 1)))

        def player_two(canonical_board: chess.Board) -> int:
            called_players.append(-1)
            assert canonical_board.turn == chess.WHITE
            assert len(canonical_board.move_stack) == 1
            return int(np.argmax(chess_game.get_valid_moves(canonical_board, 1)))

        result = Arena(player_one, player_two, chess_game).play_game(
            max_ply=1,
            initial_board=initial_board,
        )

        assert result == 0.0
        assert called_players == [-1]
        assert initial_board.fen() == initial_fen
        assert len(initial_board.move_stack) == 1
