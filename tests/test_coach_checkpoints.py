"""Tests for Coach self-play (e.g. max ply truncation, batched self-play)."""

import json
from pathlib import Path
from typing import cast

import pytest

from luna.coach import (
    Coach,
    validate_fresh_checkpoint_target,
    validate_resume_checkpoint_target,
)
from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.game.chess_game import ChessGame
from luna.game.stockfish_eval import StockfishEvalScores, StockfishEvalSkipped
from luna.network import LunaNetwork


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


def test_orphaned_best_evaluation_metadata_fails_loudly(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    run = TrainingRunConfig(checkpoint=str(tmp_path), stockfish_eval_every=0)
    coach = Coach(chess_game, LunaNetwork(chess_game, small_learner_config), run)
    coach._publish_checkpoint(1)
    (tmp_path / "best_eval.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata exists without its best checkpoint"):
        coach._update_best_from_stockfish(1, StockfishEvalScores(model_wins=1, draws=1, stockfish_wins=0))
    assert (tmp_path / "latest.pth.tar").is_file()
    assert not (tmp_path / "best.pth.tar").exists()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_best_checkpoint_record_rejects_invalid_score(
    tmp_path: Path,
    score: float,
) -> None:
    with pytest.raises(RuntimeError, match="finite and between zero and one"):
        Coach._validate_best_record(
            {
                "schema_version": 1,
                "iteration": 1,
                "score": score,
                "protocol": {},
                "source_checkpoint_sha256": "a" * 64,
            },
            protocol={},
            best_path=tmp_path / "best.pth.tar",
            trainer_iteration=1,
        )


def test_best_checkpoint_record_repairs_metadata_and_is_bound_to_protocol(
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
    metadata: dict[str, object] = json.loads(metadata_path.read_text(encoding="utf-8"))

    protocol = cast(dict[str, object], metadata["protocol"])
    assert protocol["opening_suite_version"] == 1
    metadata_path.write_text("not-json", encoding="utf-8")
    assert Coach._previous_best_score(tmp_path, protocol) == 0.75
    repaired = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert repaired["protocol"] == protocol

    changed_protocol = dict(protocol)
    changed_protocol["opening_suite_version"] = 2
    with pytest.raises(RuntimeError, match="protocol differs"):
        Coach._previous_best_score(tmp_path, changed_protocol)


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


@pytest.mark.parametrize(
    "managed_name",
    [
        "checkpoint_2.pth.tar",
        "latest.pth.tar",
        "best.pth.tar",
        "best_eval.json",
        "benchmark_state.json",
        "fairy_ladder.json",
    ],
)
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


def test_explicit_evaluation_migration_allows_only_sidecars_in_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    for name in ("benchmark_state.json", "fairy_ladder.json"):
        (target / name).write_text("{}", encoding="utf-8")
    run = TrainingRunConfig(checkpoint=str(target), stockfish_eval_every=0)

    validate_resume_checkpoint_target(
        run,
        source / "latest.pth.tar",
        allow_evaluation_artifacts_only=True,
    )
    (target / "latest.pth.tar").write_bytes(b"different lineage")

    with pytest.raises(FileExistsError, match="another checkpoint lineage"):
        validate_resume_checkpoint_target(
            run,
            source / "latest.pth.tar",
            allow_evaluation_artifacts_only=True,
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
