"""Integration tests for crash-safe external-evaluation recovery."""

from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

import luna.online_checkpoints as online_checkpoints
import main as training_entry
from luna.coach import Coach
from luna.config import EzV2LearnerConfig, TrainCliConfig, TrainingRunConfig
from luna.game.benchmark_state import (
    BENCHMARK_STATE_NAME,
    load_benchmark_state,
    record_benchmark_result,
    write_benchmark_state,
)
from luna.game.chess_game import ChessGame
from luna.game.stockfish_eval import StockfishEvalScores, stockfish_evaluation_protocol
from luna.game.stockfish_ladder import LADDER_STATE_NAME, load_fairy_ladder_state, write_fairy_ladder_state
from luna.network import LunaNetwork


def test_cross_directory_resume_recovers_newest_numbered_checkpoint(tmp_path: Path) -> None:
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    latest, newest = source / "latest.pth.tar", source / "checkpoint_12.pth.tar"
    latest.write_bytes(b"old")
    newest.write_bytes(b"new")
    config = TrainCliConfig(
        load_model=True,
        load_checkpoint_dir=str(source),
        run=TrainingRunConfig(checkpoint=str(target), stockfish_eval_every=0),
    )
    iteration = {latest.name: 11, newest.name: 12}
    with patch.object(
        online_checkpoints,
        "_validated_checkpoint_identity",
        side_effect=lambda path: online_checkpoints._CheckpointIdentity(iteration[Path(path).name], None),
    ):
        plan = training_entry._checkpoint_plan(config, config.to_training_run())
    assert plan.cross_directory
    assert plan.source == newest


@pytest.fixture
def engine_paths(tmp_path: Path) -> tuple[Path, Path]:
    official = tmp_path / "stockfish"
    fairy = tmp_path / "fairy-stockfish"
    official.write_bytes(b"official-stockfish-test-binary")
    fairy.write_bytes(b"fairy-stockfish-test-binary")
    return official, fairy


def _loaded_checkpoint(
    folder: Path,
    iteration: int,
    game: ChessGame,
    learner: EzV2LearnerConfig,
) -> tuple[LunaNetwork, Path]:
    source = LunaNetwork(game, learner)
    source._trainer_iteration = iteration
    source.save_checkpoint(str(folder), "latest.pth.tar")
    checkpoint_path = folder / "latest.pth.tar"
    loaded = LunaNetwork(game, learner)
    loaded.load_checkpoint(str(folder), checkpoint_path.name)
    return loaded, checkpoint_path


def _write_initial_sidecars(folder: Path, run: TrainingRunConfig) -> None:
    benchmark_path = folder / BENCHMARK_STATE_NAME
    benchmark_protocol = asdict(stockfish_evaluation_protocol(run))
    write_benchmark_state(
        benchmark_path,
        load_benchmark_state(benchmark_path, benchmark_protocol),
    )
    ladder_path = folder / LADDER_STATE_NAME
    write_fairy_ladder_state(
        ladder_path,
        load_fairy_ladder_state(ladder_path, run),
    )


def test_loaded_due_checkpoint_reconciles_fixed_and_ladder_before_early_return(
    tmp_path: Path,
    engine_paths: tuple[Path, Path],
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    official, fairy = engine_paths
    checkpoint_dir = tmp_path / "run"
    network, checkpoint_path = _loaded_checkpoint(checkpoint_dir, 10, chess_game, small_learner_config)
    run = TrainingRunConfig(
        num_iters=10,
        checkpoint=str(checkpoint_dir),
        stockfish_eval_every=5,
        stockfish_eval_games=2,
        stockfish_path=str(official),
        ladder_eval_every=5,
        ladder_eval_games=2,
        ladder_path=str(fairy),
    )
    _write_initial_sidecars(checkpoint_dir, run)
    coach = Coach(chess_game, network, run)
    scores = StockfishEvalScores(model_wins=1, draws=1, stockfish_wins=0)
    evaluation_order: list[str] = []

    def fixed_eval(*_args: object, **_kwargs: object) -> StockfishEvalScores:
        evaluation_order.append("fixed")
        return scores

    def ladder_eval(*_args: object, **_kwargs: object) -> object:
        evaluation_order.append("ladder")
        return object()

    with (
        patch("luna.coach_training.validate_stockfish_configuration"),
        patch("luna.coach_training.validate_ladder_configuration"),
        patch("luna.coach_evaluation.run_stockfish_eval", side_effect=fixed_eval) as fixed_match,
        patch("luna.coach_evaluation.run_fairy_ladder_eval", side_effect=ladder_eval) as ladder_match,
        patch.object(network, "warmup_mcts_inference") as warmup,
        patch.object(coach, "_learn_iterations") as train,
        patch("luna.coach_evaluation.wandb.run", None),
    ):
        coach.learn()

    assert evaluation_order == ["fixed", "ladder"]
    fixed_match.assert_called_once()
    assert fixed_match.call_args.kwargs == {"iteration": 10, "metric_prefix": None}
    ladder_match.assert_called_once()
    assert ladder_match.call_args.kwargs["iteration"] == 10
    assert ladder_match.call_args.kwargs["checkpoint_sha256"] == Coach._checkpoint_sha256(checkpoint_path)
    assert ladder_match.call_args.kwargs["state_required"] is True
    warmup.assert_called_once_with(chess_game)
    train.assert_not_called()
    benchmark_state = load_benchmark_state(
        checkpoint_dir / BENCHMARK_STATE_NAME,
        asdict(stockfish_evaluation_protocol(run)),
        required=True,
    )
    assert benchmark_state.last_iteration == 10
    assert benchmark_state.last_scores == scores


def test_durable_benchmark_result_promotes_missing_best_without_replaying_match(
    tmp_path: Path,
    engine_paths: tuple[Path, Path],
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    official, _fairy = engine_paths
    checkpoint_dir = tmp_path / "run"
    network, checkpoint_path = _loaded_checkpoint(checkpoint_dir, 5, chess_game, small_learner_config)
    run = TrainingRunConfig(
        num_iters=5,
        checkpoint=str(checkpoint_dir),
        stockfish_eval_every=5,
        stockfish_eval_games=2,
        stockfish_path=str(official),
        ladder_eval_every=0,
    )
    scores = StockfishEvalScores(model_wins=1, draws=1, stockfish_wins=0)
    record_benchmark_result(
        checkpoint_dir / BENCHMARK_STATE_NAME,
        asdict(stockfish_evaluation_protocol(run)),
        iteration=5,
        checkpoint_sha256=Coach._checkpoint_sha256(checkpoint_path),
        scores=scores,
    )
    coach = Coach(chess_game, network, run)

    with (
        patch("luna.coach_training.validate_stockfish_configuration"),
        patch("luna.coach_evaluation.run_stockfish_eval") as fixed_match,
        patch.object(network, "warmup_mcts_inference"),
        patch.object(coach, "_learn_iterations") as train,
        patch("luna.coach_evaluation.wandb.run", None),
    ):
        coach.learn()

    fixed_match.assert_not_called()
    train.assert_not_called()
    best_path = checkpoint_dir / "best.pth.tar"
    assert best_path.is_file()
    assert (checkpoint_dir / "best_eval.json").is_file()
    assert LunaNetwork.checkpoint_trainer_iteration(best_path) == 5
    best_checkpoint = LunaNetwork._read_checkpoint(best_path)
    best_record = best_checkpoint["best_evaluation"]
    assert isinstance(best_record, dict)
    assert best_record["iteration"] == 5
    assert best_record["score"] == 0.75
    assert best_record["source_checkpoint_sha256"] == Coach._checkpoint_sha256(checkpoint_path)


@pytest.mark.parametrize(
    ("fixed_interval", "ladder_interval", "message"),
    [
        (5, 0, "Required benchmark state is missing"),
        (0, 5, "Required Fairy ladder state is missing"),
    ],
)
def test_resume_after_first_interval_requires_existing_evaluation_sidecar(
    tmp_path: Path,
    engine_paths: tuple[Path, Path],
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    fixed_interval: int,
    ladder_interval: int,
    message: str,
) -> None:
    official, fairy = engine_paths
    checkpoint_dir = tmp_path / f"run-{fixed_interval}-{ladder_interval}"
    network, _checkpoint_path = _loaded_checkpoint(checkpoint_dir, 6, chess_game, small_learner_config)
    run = TrainingRunConfig(
        num_iters=6,
        checkpoint=str(checkpoint_dir),
        stockfish_eval_every=fixed_interval,
        stockfish_eval_games=2,
        stockfish_path=str(official),
        ladder_eval_every=ladder_interval,
        ladder_eval_games=2,
        ladder_path=str(fairy),
    )
    coach = Coach(chess_game, network, run)

    with (
        patch("luna.coach_training.validate_stockfish_configuration") as validate_fixed,
        patch("luna.coach_training.validate_ladder_configuration") as validate_ladder,
        patch("luna.coach_evaluation.run_stockfish_eval") as fixed_match,
        patch("luna.coach_evaluation.run_fairy_ladder_eval") as ladder_match,
        patch.object(network, "warmup_mcts_inference") as warmup,
        pytest.raises((FileNotFoundError, RuntimeError), match=message),
    ):
        coach.learn()

    validate_fixed.assert_not_called()
    validate_ladder.assert_not_called()
    fixed_match.assert_not_called()
    ladder_match.assert_not_called()
    warmup.assert_not_called()


def test_cross_directory_migration_initializes_empty_evaluation_sidecars(
    tmp_path: Path,
    engine_paths: tuple[Path, Path],
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    official, fairy = engine_paths
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    network, source_checkpoint = _loaded_checkpoint(source_dir, 6, chess_game, small_learner_config)
    run = TrainingRunConfig(
        num_iters=6,
        checkpoint=str(target_dir),
        stockfish_eval_every=5,
        stockfish_eval_games=2,
        stockfish_path=str(official),
        ladder_eval_every=5,
        ladder_eval_games=2,
        ladder_path=str(fairy),
    )
    coach = Coach(
        chess_game,
        network,
        run,
        initialize_evaluation_state=True,
    )

    with (
        patch("luna.coach_training.validate_stockfish_configuration") as validate_fixed,
        patch("luna.coach_training.validate_ladder_configuration") as validate_ladder,
        patch.object(network, "warmup_mcts_inference") as warmup,
        patch.object(coach, "_learn_iterations") as train,
    ):
        coach.learn()

    assert source_checkpoint.is_file()
    assert not (target_dir / "latest.pth.tar").exists()
    benchmark_state = load_benchmark_state(
        target_dir / BENCHMARK_STATE_NAME,
        asdict(stockfish_evaluation_protocol(run)),
        required=True,
    )
    ladder_state = load_fairy_ladder_state(target_dir / LADDER_STATE_NAME, run, required=True)
    assert benchmark_state.last_iteration is None
    assert benchmark_state.evaluation_step == 0
    assert ladder_state.last_iteration is None
    assert ladder_state.evaluation_step == 0
    validate_fixed.assert_not_called()
    validate_ladder.assert_not_called()
    warmup.assert_not_called()
    train.assert_not_called()
