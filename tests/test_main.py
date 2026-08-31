"""Training entry-point safety tests."""

import subprocess
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

import main as training_entry
from luna.config import EzV2LearnerConfig, TrainCliConfig, TrainingRunConfig, WandbResumeMode


def test_new_training_phase_target_must_be_dedicated_and_empty(tmp_path: Path) -> None:
    target = tmp_path / "phase"
    training_entry.validate_new_training_phase_target(str(target))
    target.mkdir()
    training_entry.validate_new_training_phase_target(str(target))
    (target / "notes.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError, match="requires an empty checkpoint directory"):
        training_entry.validate_new_training_phase_target(str(target))

    with pytest.raises(ValueError, match="requires a non-empty"):
        training_entry.validate_new_training_phase_target("")


def test_main_rejects_resume_and_new_phase_together() -> None:
    config = TrainCliConfig(load_model=True, new_training_phase=True)

    with patch.object(training_entry.tyro, "cli", return_value=config):
        assert training_entry.main() == 2


def test_main_requires_explicit_evaluation_state_initialization_for_cross_directory_resume(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "latest.pth.tar").write_bytes(b"checkpoint placeholder")
    config = TrainCliConfig(
        load_model=True,
        load_checkpoint_dir=str(source),
        run=TrainingRunConfig(
            checkpoint=str(tmp_path / "target"),
            stockfish_eval_every=0,
            ladder_eval_every=5,
        ),
        learner=EzV2LearnerConfig(device="cpu"),
    )

    with patch.object(training_entry.tyro, "cli", return_value=config):
        assert training_entry.main() == 2


def test_main_rejects_evaluation_state_initialization_without_migration(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "latest.pth.tar"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint placeholder")
    config = TrainCliConfig(
        load_model=True,
        initialize_evaluation_state=True,
        load_checkpoint_dir=str(checkpoint.parent),
        run=TrainingRunConfig(checkpoint=str(checkpoint.parent), stockfish_eval_every=0),
        learner=EzV2LearnerConfig(device="cpu"),
    )

    with patch.object(training_entry.tyro, "cli", return_value=config):
        assert training_entry.main() == 2


def test_main_rejects_invalid_log_level() -> None:
    config = TrainCliConfig(log_level="verbose")

    with patch.object(training_entry.tyro, "cli", return_value=config):
        assert training_entry.main() == 2


@pytest.mark.parametrize("resume_mode", ["allow", "never", "must"])
def test_cli_parses_stable_wandb_run_id_and_resume_mode(resume_mode: str) -> None:
    config = training_entry.tyro.cli(
        TrainCliConfig,
        args=[
            "--wandb-project",
            "ChessRL",
            "--wandb-run-id",
            "luna-strength-1500-v1",
            "--wandb-run-name",
            "Luna Strength 1500 v1",
            "--wandb-resume",
            resume_mode,
        ],
    )

    assert config.wandb_project == "ChessRL"
    assert config.wandb_run_id == "luna-strength-1500-v1"
    assert config.wandb_run_name == "Luna Strength 1500 v1"
    assert config.wandb_resume == resume_mode


def test_main_rejects_invalid_wandb_resume_mode() -> None:
    config = TrainCliConfig(wandb_resume=cast(WandbResumeMode, "sometimes"))

    with patch.object(training_entry.tyro, "cli", return_value=config):
        assert training_entry.main() == 2


def test_cli_rejects_invalid_wandb_resume_mode() -> None:
    with pytest.raises(SystemExit):
        training_entry.tyro.cli(TrainCliConfig, args=["--wandb-resume", "sometimes"])


@pytest.mark.parametrize("run_id", ["", "   ", " leading", "trailing ", "bad/id", "bad:id", "bad?id"])
def test_main_rejects_invalid_wandb_run_id(run_id: str) -> None:
    config = TrainCliConfig(wandb_run_id=run_id)

    with patch.object(training_entry.tyro, "cli", return_value=config):
        assert training_entry.main() == 2


@pytest.mark.parametrize("run_name", ["", "   ", " leading", "trailing "])
def test_main_rejects_invalid_wandb_run_name(run_name: str) -> None:
    config = TrainCliConfig(wandb_run_name=run_name)

    with patch.object(training_entry.tyro, "cli", return_value=config):
        assert training_entry.main() == 2


def test_main_routes_new_phase_to_weights_only_initializer(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "latest.pth.tar").write_bytes(b"checkpoint placeholder")
    target = tmp_path / "new-phase"
    config = TrainCliConfig(
        new_training_phase=True,
        load_checkpoint_dir=str(source),
        wandb_project="ChessRL",
        wandb_run_id="luna-strength-1500-v1",
        wandb_run_name="Luna Strength 1500 v1",
        wandb_resume="never",
        run=TrainingRunConfig(checkpoint=str(target), stockfish_eval_every=0),
        learner=EzV2LearnerConfig(device="cpu"),
    )

    with (
        patch.object(training_entry.tyro, "cli", return_value=config),
        patch.object(training_entry, "LunaNetwork") as network_type,
        patch.object(training_entry, "Coach") as coach_type,
    ):
        network_type.__name__ = "LunaNetwork"
        result = training_entry.main()

    assert result == 0
    network_type.return_value.initialize_training_phase.assert_called_once_with(str(source), "latest.pth.tar")
    network_type.return_value.load_checkpoint.assert_not_called()
    assert coach_type.call_args.kwargs["wandb_project"] == "ChessRL"
    assert coach_type.call_args.kwargs["wandb_run_id"] == "luna-strength-1500-v1"
    assert coach_type.call_args.kwargs["wandb_run_name"] == "Luna Strength 1500 v1"
    assert coach_type.call_args.kwargs["wandb_resume"] == "never"
    coach_type.return_value.learn.assert_called_once_with()


@pytest.mark.parametrize(
    ("target", "resume_mode"),
    [("train-phase", "never"), ("resume-phase", "must"), ("migrate-ladder-phase", "never")],
)
def test_phase_make_target_sets_explicit_wandb_resume_policy(target: str, resume_mode: str) -> None:
    repository = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        ["make", "-n", target],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '--wandb-run-id "luna-fairy-ladder-v1"' in result.stdout
    assert '--wandb-run-name "Luna Fairy Ladder 500+ · Benchmark 1500 v1"' in result.stdout
    assert f"--wandb-resume {resume_mode}" in result.stdout
    assert "--run.self-play-workers 4" in result.stdout
    assert "--run.stockfish-elo 1500" in result.stdout
    assert "--run.ladder-start-elo 500" in result.stdout
    assert "--run.ladder-step-elo 100" in result.stdout
    assert "--learner.reanalyze-prob 0.10" in result.stdout
    if target == "migrate-ladder-phase":
        assert "--initialize-evaluation-state" in result.stdout


def test_train_phase_make_target_keeps_pinned_source_preflight() -> None:
    repository = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        ["make", "-n", "train-phase"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "best.pth.tar" in result.stdout
    assert "b6ec9f2e5455f592a3833a285fe478dfba9bb9bdddba9207a2d66572277c7b8d" in result.stdout
