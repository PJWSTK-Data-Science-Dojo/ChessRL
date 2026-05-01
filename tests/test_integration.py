"""Integration tests for full training pipeline."""

import tempfile

from luna.coach import Coach
from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.game.chess_game import ChessGame
from luna.network import LunaNetwork


def _minimal_config() -> tuple[TrainingRunConfig, EzV2LearnerConfig]:
    """Minimal config for fast integration tests."""
    run = TrainingRunConfig(
        num_iters=1,
        num_episodes=2,
        parallel_games=2,
        num_mcts_sims=3,
        max_ply=20,  # Short games
        train_steps_per_iter=5,
        batch_size=4,
        arena_compare=2,
        checkpoint="",  # Don't save checkpoints in tests
        temp_threshold=1,
        recurrent_policy_topk=None,  # Full policy for accuracy
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
    """Test that full training iterations complete successfully."""

    def test_one_iteration_with_legal_masking(self) -> None:
        """Verify one full training iteration with legal move masking."""
        game = ChessGame()
        run_cfg, learner_cfg = _minimal_config()

        nnet = LunaNetwork(game, learner_cfg)
        coach = Coach(game, nnet, run_cfg)

        # Run one iteration (self-play + training + arena)
        # This should complete without errors and with legal masking active
        initial_step = coach.nnet._global_step

        # Execute one iteration manually (since we don't want to save checkpoints)
        import time

        start_time = time.time()

        # Self-play
        train_examples = coach.execute_episodes_batched(run_cfg.num_episodes)
        assert len(train_examples) == run_cfg.num_episodes
        assert all(traj.game_length > 0 for traj in train_examples)

        # Add to replay
        for traj in train_examples:
            coach.replay.save(traj)

        # Training
        if coach.replay.size() >= run_cfg.batch_size:
            coach.nnet.train(
                coach.replay,
                run_cfg.train_steps_per_iter,
                mcts_for_reanalyze=None,  # No reanalysis in test
            )

        # Arena
        coach.pnet.nnet.load_state_dict(coach.nnet.nnet.state_dict())
        arena_results = coach._play_arena_games_batched(
            coach.nnet,
            coach.pnet,
            Coach._arena_mcts_params(run_cfg),
            num_games=run_cfg.arena_compare,
        )
        assert len(arena_results) == run_cfg.arena_compare

        elapsed = time.time() - start_time

        # Verify training happened
        assert coach.nnet._global_step > initial_step
        assert coach.replay.size() > 0

        # Should complete reasonably fast on CPU
        assert elapsed < 300, f"Integration test took {elapsed:.1f}s (should be <300s)"

        # Check that all games finished (no crashes from illegal moves)
        for traj in train_examples:
            # Last reward should be terminal reward (non-zero)
            # Note: get_next_state defensive handling might pick random moves,
            # but it should still complete without crashing
            assert len(traj.rewards) > 0

    def test_training_with_reanalysis(self) -> None:
        """Test training with reanalysis enabled."""
        game = ChessGame()
        run_cfg, learner_cfg = _minimal_config()

        # Enable reanalysis
        learner_cfg.reanalyze_mcts_sims = 4
        learner_cfg.reanalyze_prob = 1.0  # Always reanalyze for testing
        learner_cfg.mixed_value_td_until_step = 0  # Start immediately

        nnet = LunaNetwork(game, learner_cfg)
        coach = Coach(game, nnet, run_cfg)

        # Generate some data
        train_examples = coach.execute_episodes_batched(2)
        for traj in train_examples:
            coach.replay.save(traj)

        # Train with reanalysis
        initial_step = coach.nnet._global_step
        coach.nnet.train(
            coach.replay,
            steps=3,
            mcts_for_reanalyze=coach.nnet,  # Use current network for reanalysis
        )

        # Should complete without errors
        assert coach.nnet._global_step > initial_step

    def test_batched_self_play_board_tracking(self) -> None:
        """Verify that batched self-play properly tracks board states in MCTS."""
        game = ChessGame()
        run_cfg, learner_cfg = _minimal_config()

        nnet = LunaNetwork(game, learner_cfg)
        coach = Coach(game, nnet, run_cfg)

        # Run batched self-play
        trajectories = coach.execute_episodes_batched(num_episodes=3)

        assert len(trajectories) == 3

        for traj in trajectories:
            # Each trajectory should have valid game data
            assert traj.game_length > 0
            assert len(traj.observations) == traj.game_length
            assert len(traj.actions) == traj.game_length
            assert len(traj.rewards) == traj.game_length
            assert len(traj.root_policies) == traj.game_length
            assert len(traj.root_values) == traj.game_length
            assert len(traj.valids) == traj.game_length

            # All moves should have been legal (with legal masking)
            # We can't verify this directly, but the game should complete
            assert traj.observations.shape[0] > 0


class TestDeviceSupport:
    """Test training on different devices."""

    def test_cpu_training(self) -> None:
        """Verify training works on CPU."""
        game = ChessGame()
        run_cfg, learner_cfg = _minimal_config()
        learner_cfg.device = "cpu"

        nnet = LunaNetwork(game, learner_cfg)
        assert nnet.device.type == "cpu"

        # Run minimal training
        coach = Coach(game, nnet, run_cfg)
        train_examples = coach.execute_episodes_batched(1)
        assert len(train_examples) == 1

    def test_checkpoint_save_load(self) -> None:
        """Test saving and loading checkpoints."""
        game = ChessGame()
        run_cfg, learner_cfg = _minimal_config()

        nnet1 = LunaNetwork(game, learner_cfg)

        # Train for a few steps
        coach = Coach(game, nnet1, run_cfg)
        train_examples = coach.execute_episodes_batched(1)
        for traj in train_examples:
            coach.replay.save(traj)
        nnet1.train(coach.replay, steps=2, mcts_for_reanalyze=None)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            nnet1.save_checkpoint(tmpdir, "test.pth.tar")

            # Load into new network
            nnet2 = LunaNetwork(game, learner_cfg)
            nnet2.load_checkpoint(tmpdir, "test.pth.tar")

            # Global step should match
            assert nnet2._global_step == nnet1._global_step

            # Weights should match
            for p1, p2 in zip(nnet1.nnet.parameters(), nnet2.nnet.parameters()):
                assert (p1 == p2).all()
