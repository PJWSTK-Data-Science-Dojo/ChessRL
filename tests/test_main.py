"""Training entry-point safety tests."""

from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

import luna.online_checkpoints as online_checkpoints
import main as training_entry
from luna.config import EzV2LearnerConfig, TrainCliConfig, TrainingRunConfig, WandbResumeMode


def test_new_training_phase_target_must_be_dedicated_and_empty(tmp_path: Path) -> None:
    target = tmp_path / "phase"
    online_checkpoints.validate_new_training_phase_target(str(target))
    target.mkdir()
    online_checkpoints.validate_new_training_phase_target(str(target))
    (target / "notes.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError, match="requires an empty checkpoint directory"):
        online_checkpoints.validate_new_training_phase_target(str(target))

    with pytest.raises(ValueError, match="requires a non-empty"):
        online_checkpoints.validate_new_training_phase_target("")


def test_resume_selects_newest_numbered_checkpoint_when_latest_lags(tmp_path: Path) -> None:
    target = tmp_path / "run"
    target.mkdir()
    latest = target / "latest.pth.tar"
    newest = target / "checkpoint_12.pth.tar"
    latest.write_bytes(b"old")
    newest.write_bytes(b"new")

    with patch.object(
        online_checkpoints,
        "_validated_checkpoint_identity",
        side_effect=lambda path: online_checkpoints._CheckpointIdentity(
            11 if Path(path).name == latest.name else 12,
            None,
        ),
    ):
        selected = online_checkpoints.resolve_resume_checkpoint(latest, target)

    assert selected == newest


def test_resume_prefers_immutable_checkpoint_when_latest_has_same_iteration(tmp_path: Path) -> None:
    target = tmp_path / "run"
    target.mkdir()
    latest = target / "latest.pth.tar"
    numbered = target / "checkpoint_12.pth.tar"
    latest.write_bytes(b"latest")
    numbered.write_bytes(b"numbered")

    identity = online_checkpoints._CheckpointIdentity(12, None)
    with patch.object(online_checkpoints, "_validated_checkpoint_identity", return_value=identity):
        selected = online_checkpoints.resolve_resume_checkpoint(latest, target)

    assert selected == numbered
    assert latest.read_bytes() == numbered.read_bytes()


def test_resume_recovers_numbered_checkpoint_when_latest_is_missing(tmp_path: Path) -> None:
    target = tmp_path / "run"
    target.mkdir()
    numbered = target / "checkpoint_3.pth.tar"
    numbered.write_bytes(b"checkpoint")

    identity = online_checkpoints._CheckpointIdentity(3, None)
    with patch.object(online_checkpoints, "_validated_checkpoint_identity", return_value=identity):
        selected = online_checkpoints.resolve_resume_checkpoint(target / "latest.pth.tar", target)

    assert selected == numbered


def test_resume_rejects_numbered_checkpoint_with_mismatched_iteration(tmp_path: Path) -> None:
    target = tmp_path / "run"
    target.mkdir()
    numbered = target / "checkpoint_4.pth.tar"
    numbered.write_bytes(b"checkpoint")

    with (
        patch.object(
            online_checkpoints,
            "_validated_checkpoint_identity",
            return_value=online_checkpoints._CheckpointIdentity(3, None),
        ),
        pytest.raises(RuntimeError, match="differs from its filename"),
    ):
        online_checkpoints.resolve_resume_checkpoint(target / "latest.pth.tar", target)


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


def test_main_handles_training_interrupt_without_traceback(tmp_path: Path) -> None:
    config = TrainCliConfig(
        run=TrainingRunConfig(checkpoint=str(tmp_path / "run"), stockfish_eval_every=0),
        learner=EzV2LearnerConfig(device="cpu"),
    )

    with (
        patch.object(training_entry.tyro, "cli", return_value=config),
        patch.object(training_entry, "LunaNetwork") as network_type,
        patch.object(training_entry, "Coach") as coach_type,
    ):
        network_type.__name__ = "LunaNetwork"
        coach_type.return_value.learn.side_effect = KeyboardInterrupt

        assert training_entry.main() == 130


def test_main_returns_non_restarting_exit_for_representation_collapse(tmp_path: Path) -> None:
    config = TrainCliConfig(
        run=TrainingRunConfig(checkpoint=str(tmp_path / "run"), stockfish_eval_every=0),
        learner=EzV2LearnerConfig(device="cpu"),
    )

    with (
        patch.object(training_entry.tyro, "cli", return_value=config),
        patch.object(training_entry, "LunaNetwork") as network_type,
        patch.object(training_entry, "Coach") as coach_type,
    ):
        network_type.__name__ = "LunaNetwork"
        coach_type.return_value.learn.side_effect = training_entry.RepresentationCollapseError("collapsed")

        assert training_entry.main() == 78


def test_training_finishes_wandb_with_process_exit_code() -> None:
    coach = cast(training_entry.Coach, object())
    with (
        patch.object(training_entry, "_learn", return_value=78),
        patch.object(training_entry.wandb, "run", object()),
        patch.object(training_entry.wandb, "finish") as finish,
    ):
        result = training_entry._learn_and_finish_wandb(coach)

    assert result == 78
    finish.assert_called_once_with(exit_code=78)


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
        patch.object(training_entry, "publish_bootstrap_checkpoint") as publish_bootstrap,
        patch.object(training_entry, "Coach") as coach_type,
    ):
        network_type.__name__ = "LunaNetwork"
        result = training_entry.main()

    assert result == 0
    network_type.return_value.initialize_training_phase.assert_called_once_with(str(source), "latest.pth.tar")
    publish_bootstrap.assert_called_once_with(network_type.return_value, str(target))
    network_type.return_value.load_checkpoint.assert_not_called()
    assert coach_type.call_args.kwargs["wandb_project"] == "ChessRL"
    assert coach_type.call_args.kwargs["wandb_run_id"] == "luna-strength-1500-v1"
    assert coach_type.call_args.kwargs["wandb_run_name"] == "Luna Strength 1500 v1"
    assert coach_type.call_args.kwargs["wandb_resume"] == "never"
    assert coach_type.call_args.kwargs["restore_replay"] is False
    coach_type.return_value.learn.assert_called_once_with()
