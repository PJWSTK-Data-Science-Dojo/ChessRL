"""Tests for Coach self-play (e.g. max ply truncation, batched self-play)."""

import json
from collections.abc import Sequence
from pathlib import Path

import chess
import numpy as np
import pytest

from luna.coach import Coach, validate_fresh_checkpoint_target, validate_resume_checkpoint_target
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


def test_best_evaluation_metadata_is_bound_to_its_protocol(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    run = TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0)
    coach = Coach(chess_game, LunaNetwork(chess_game, small_learner_config), run)
    coach._publish_checkpoint(1)
    score = StockfishEvalScores(model_wins=1, draws=1, stockfish_wins=0)
    coach._update_best_from_stockfish(1, score)
    metadata_path = tmp_path / "best_eval.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["protocol"]["opening_suite_version"] == 1
    metadata["protocol"]["opening_suite_version"] = 2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="protocol differs"):
        coach._update_best_from_stockfish(1, score)


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


def test_fresh_training_refuses_managed_checkpoint_without_clobbering_it(tmp_path: Path) -> None:
    latest_path = tmp_path / "latest.pth.tar"
    original = b"existing checkpoint"
    latest_path.write_bytes(original)
    run = TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0)

    with pytest.raises(FileExistsError, match="Fresh training would overwrite managed files"):
        validate_fresh_checkpoint_target(run)

    assert latest_path.read_bytes() == original


@pytest.mark.parametrize("managed_name", ["checkpoint_2.pth.tar", "latest.pth.tar", "best.pth.tar", "best_eval.json"])
def test_resume_refuses_managed_state_from_another_directory(tmp_path: Path, managed_name: str) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    managed_path = target / managed_name
    managed_path.write_bytes(b"another run")
    run = TrainingRunConfig(checkpoint=str(target), stockfish_eval_every=0)

    with pytest.raises(FileExistsError, match="another checkpoint lineage"):
        validate_resume_checkpoint_target(run, source / "latest.pth.tar")

    assert managed_path.read_bytes() == b"another run"


def test_resume_allows_source_directory_or_empty_new_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    empty_target = tmp_path / "empty-target"
    source.mkdir()
    (source / "latest.pth.tar").write_bytes(b"resume checkpoint")

    validate_resume_checkpoint_target(
        TrainingRunConfig(checkpoint=str(source), stockfish_eval_every=0),
        source / "latest.pth.tar",
    )
    validate_resume_checkpoint_target(
        TrainingRunConfig(checkpoint=str(empty_target), stockfish_eval_every=0),
        source / "latest.pth.tar",
    )


def test_resume_resolves_traversal_before_comparing_lineages(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "latest.pth.tar").write_bytes(b"source run")
    target_latest = target / "latest.pth.tar"
    target_latest.write_bytes(b"target run")
    traversing_source = target / ".." / "source" / "latest.pth.tar"

    with pytest.raises(FileExistsError, match="another checkpoint lineage"):
        validate_resume_checkpoint_target(
            TrainingRunConfig(checkpoint=str(target), stockfish_eval_every=0),
            traversing_source,
        )

    assert target_latest.read_bytes() == b"target run"


def test_zero_counter_checkpoint_is_recognized_as_resume(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    initial = LunaNetwork(chess_game, small_learner_config)
    initial.save_checkpoint(str(tmp_path), "latest.pth.tar")
    resumed = LunaNetwork(chess_game, small_learner_config)
    resumed.load_checkpoint(str(tmp_path), "latest.pth.tar")
    coach = Coach(
        chess_game,
        resumed,
        TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0),
    )

    coach._assert_checkpoint_target()
    coach._assert_checkpoint_target()

    assert resumed._global_step == 0
    assert resumed._trainer_iteration == 0
    assert (tmp_path / "latest.pth.tar").is_file()


def test_zero_counter_resume_rejects_newer_numbered_lineage(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    initial = LunaNetwork(chess_game, small_learner_config)
    initial.save_checkpoint(str(tmp_path), "latest.pth.tar")
    newer = LunaNetwork(chess_game, small_learner_config)
    newer._trainer_iteration = 5
    newer.save_checkpoint(str(tmp_path), "checkpoint_5.pth.tar")
    resumed = LunaNetwork(chess_game, small_learner_config)
    resumed.load_checkpoint(str(tmp_path), "latest.pth.tar")
    coach = Coach(
        chess_game,
        resumed,
        TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0),
    )

    coach._assert_checkpoint_target()
    with pytest.raises(RuntimeError, match="newer training state"):
        coach._assert_checkpoint_lineage()


def test_publish_checkpoint_refuses_to_replace_numbered_snapshot(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    checkpoint_path = tmp_path / "checkpoint_1.pth.tar"
    original = b"existing checkpoint"
    checkpoint_path.write_bytes(original)
    network = LunaNetwork(chess_game, small_learner_config)
    coach = Coach(
        chess_game,
        network,
        TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0),
    )

    with pytest.raises(FileExistsError, match="immutable numbered checkpoint"):
        coach._publish_checkpoint(1)

    assert checkpoint_path.read_bytes() == original
    assert network._trainer_iteration == 0


def test_publish_checkpoint_restores_iteration_when_numbered_save_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    network._trainer_iteration = 4
    coach = Coach(
        chess_game,
        network,
        TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0),
    )

    def fail_save(folder: str, filename: str) -> None:
        del folder, filename
        raise OSError("storage unavailable")

    monkeypatch.setattr(network, "save_checkpoint", fail_save)

    with pytest.raises(OSError, match="storage unavailable"):
        coach._publish_checkpoint(5)

    assert network._trainer_iteration == 4
    assert not (tmp_path / "checkpoint_5.pth.tar").exists()


def test_publish_checkpoint_refuses_to_roll_back_a_newer_lineage(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    newer_path = tmp_path / "checkpoint_5.pth.tar"
    newer_path.write_bytes(b"newer checkpoint")
    network = LunaNetwork(chess_game, small_learner_config)
    network._trainer_iteration = 1
    coach = Coach(
        chess_game,
        network,
        TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0),
    )

    with pytest.raises(FileExistsError, match="Refusing non-monotonic checkpoint"):
        coach._publish_checkpoint(2)

    assert newer_path.read_bytes() == b"newer checkpoint"
    assert not (tmp_path / "latest.pth.tar").exists()


def test_publish_checkpoint_refuses_to_roll_back_latest_only_lineage(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    newer = LunaNetwork(chess_game, small_learner_config)
    newer._trainer_iteration = 5
    newer.save_checkpoint(str(tmp_path), "latest.pth.tar")
    resumed = LunaNetwork(chess_game, small_learner_config)
    resumed._trainer_iteration = 1
    coach = Coach(
        chess_game,
        resumed,
        TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0),
    )

    with pytest.raises(FileExistsError, match="Refusing non-monotonic checkpoint"):
        coach._publish_checkpoint(2)

    assert LunaNetwork.checkpoint_trainer_iteration(tmp_path / "latest.pth.tar") == 5


def test_coach_rejects_run_and_learner_discount_mismatch(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.discount = 0.9
    network = LunaNetwork(chess_game, small_learner_config)

    with pytest.raises(ValueError, match=r"run\.discount and learner\.discount must match"):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(discount=1.0, stockfish_eval_every=0),
        )


def test_coach_rejects_invalid_training_schedule(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with pytest.raises(ValueError, match="parallel_games must be a positive integer"):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(parallel_games=0, stockfish_eval_every=0),
        )


def test_coach_rejects_evaluation_larger_than_opening_suite(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with pytest.raises(ValueError, match="stockfish_eval_games cannot exceed 20"):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(stockfish_eval_games=22),
        )
