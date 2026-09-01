"""Integration tests for full training pipeline."""

import tempfile
import time

from luna.coach import Coach
from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.game.chess_game import ChessGame
from luna.network import LunaNetwork


def _minimal_config() -> tuple[TrainingRunConfig, EzV2LearnerConfig]:
    run = TrainingRunConfig(
        num_iters=1,
        num_episodes=2,
        parallel_games=2,
        num_mcts_sims=3,
        max_ply=20,  # Short games
        train_steps_per_iter=5,
        checkpoint="",  # Don't save checkpoints in tests
        temp_threshold=1,
        recurrent_policy_topk=None,  # Full policy for accuracy
        stockfish_eval_every=0,  # Avoid Stockfish during integration runs if loop expands
    )

    learner = EzV2LearnerConfig(
        device="cpu",
        num_channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        proj_dim=32,
        batch_size=4,
        compile_inference=False,
        dataloader_workers=0,  # Avoid multiprocessing in tests
    )

    return run, learner


class TestFullTrainingIteration:
    def test_one_iteration_with_legal_masking(self) -> None:
        game = ChessGame()
        run_cfg, learner_cfg = _minimal_config()

        nnet = LunaNetwork(game, learner_cfg)
        coach = Coach(game, nnet, run_cfg)

        initial_step = coach.nnet._global_step
        start_time = time.time()

        train_examples = coach.execute_episodes_batched(run_cfg.num_episodes)
        assert len(train_examples) == run_cfg.num_episodes
        assert all(traj.game_length > 0 for traj in train_examples)

        for traj in train_examples:
            coach.replay.save_trajectory(traj)

        if coach.replay.size >= learner_cfg.batch_size:
            coach.nnet.train_ezv2(
                coach.replay,
                steps=run_cfg.train_steps_per_iter,
                mcts_for_reanalyze=None,
            )

        elapsed = time.time() - start_time

        assert coach.nnet._global_step > initial_step
        assert coach.replay.size > 0

        assert elapsed < 300, f"Integration test took {elapsed:.1f}s (should be <300s)"

        for traj in train_examples:
            # The trajectory remains internally consistent even when the
            # configured ply cap adjudicates the position as a draw.
            assert len(traj.rewards) > 0

    def test_training_with_reanalysis(self) -> None:
        game = ChessGame()
        run_cfg, learner_cfg = _minimal_config()

        learner_cfg.reanalyze_mcts_sims = 4
        learner_cfg.reanalyze_prob = 1.0
        learner_cfg.reanalyze_start_step = 0

        nnet = LunaNetwork(game, learner_cfg)
        coach = Coach(game, nnet, run_cfg)

        train_examples = coach.execute_episodes_batched(2)
        for traj in train_examples:
            coach.replay.save_trajectory(traj)

        initial_step = coach.nnet._global_step
        coach.nnet.train_ezv2(
            coach.replay,
            steps=3,
            mcts_for_reanalyze=run_cfg,
        )

        assert coach.nnet._global_step > initial_step

    def test_batched_self_play_board_tracking(self) -> None:
        game = ChessGame()
        run_cfg, learner_cfg = _minimal_config()

        nnet = LunaNetwork(game, learner_cfg)
        coach = Coach(game, nnet, run_cfg)

        trajectories = coach.execute_episodes_batched(num_episodes=3)

        assert len(trajectories) == 3

        for traj in trajectories:
            assert traj.game_length > 0
            assert len(traj.observations) == traj.game_length
            assert len(traj.actions) == traj.game_length
            assert len(traj.rewards) == traj.game_length
            assert len(traj.root_policies) == traj.game_length
            assert len(traj.root_values) == traj.game_length
            assert len(traj.valids) == traj.game_length

            assert traj.observations.shape[0] > 0


class TestDeviceSupport:
    def test_cpu_training(self) -> None:
        game = ChessGame()
        run_cfg, learner_cfg = _minimal_config()
        learner_cfg.device = "cpu"

        nnet = LunaNetwork(game, learner_cfg)
        assert nnet.device.type == "cpu"

        coach = Coach(game, nnet, run_cfg)
        train_examples = coach.execute_episodes_batched(1)
        assert len(train_examples) == 1

    def test_checkpoint_save_load(self) -> None:
        game = ChessGame()
        run_cfg, learner_cfg = _minimal_config()

        nnet1 = LunaNetwork(game, learner_cfg)

        coach = Coach(game, nnet1, run_cfg)
        train_examples = coach.execute_episodes_batched(1)
        for traj in train_examples:
            coach.replay.save_trajectory(traj)
        nnet1.train_ezv2(coach.replay, steps=2, mcts_for_reanalyze=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            nnet1.save_checkpoint(tmpdir, "test.pth.tar")

            nnet2 = LunaNetwork(game, learner_cfg)
            nnet2.load_checkpoint(tmpdir, "test.pth.tar")

            assert nnet2._global_step == nnet1._global_step

            for p1, p2 in zip(nnet1.nnet.parameters(), nnet2.nnet.parameters(), strict=True):
                assert (p1 == p2).all()
