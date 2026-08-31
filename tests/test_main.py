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


def test_resume_selects_newest_numbered_checkpoint_when_latest_lags(tmp_path: Path) -> None:
    target = tmp_path / "run"
    target.mkdir()
    latest = target / "latest.pth.tar"
    newest = target / "checkpoint_12.pth.tar"
    latest.write_bytes(b"old")
    newest.write_bytes(b"new")

    with patch.object(
        training_entry.LunaNetwork,
        "checkpoint_trainer_iteration",
        side_effect=lambda path: 11 if Path(path).name == latest.name else 12,
    ):
        selected = training_entry.resolve_resume_checkpoint(latest, target)

    assert selected == newest


def test_resume_prefers_latest_when_its_iteration_matches_numbered_checkpoint(tmp_path: Path) -> None:
    target = tmp_path / "run"
    target.mkdir()
    latest = target / "latest.pth.tar"
    numbered = target / "checkpoint_12.pth.tar"
    latest.write_bytes(b"latest")
    numbered.write_bytes(b"numbered")

    with patch.object(training_entry.LunaNetwork, "checkpoint_trainer_iteration", return_value=12):
        selected = training_entry.resolve_resume_checkpoint(latest, target)

    assert selected == latest


def test_resume_recovers_numbered_checkpoint_when_latest_is_missing(tmp_path: Path) -> None:
    target = tmp_path / "run"
    target.mkdir()
    numbered = target / "checkpoint_3.pth.tar"
    numbered.write_bytes(b"checkpoint")

    with patch.object(training_entry.LunaNetwork, "checkpoint_trainer_iteration", return_value=3):
        selected = training_entry.resolve_resume_checkpoint(target / "latest.pth.tar", target)

    assert selected == numbered


def test_resume_rejects_numbered_checkpoint_with_mismatched_iteration(tmp_path: Path) -> None:
    target = tmp_path / "run"
    target.mkdir()
    numbered = target / "checkpoint_4.pth.tar"
    numbered.write_bytes(b"checkpoint")

    with (
        patch.object(training_entry.LunaNetwork, "checkpoint_trainer_iteration", return_value=3),
        pytest.raises(RuntimeError, match="differs from its filename"),
    ):
        training_entry.resolve_resume_checkpoint(target / "latest.pth.tar", target)


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
    ("target", "resume_mode", "run_id", "run_name"),
    [
        (
            "train-phase",
            "never",
            "luna-balanced-ezv2-anti-collapse-v2",
            "Luna Balanced EZ-V2 · Anti-Collapse v2",
        ),
        (
            "resume-phase",
            "must",
            "luna-balanced-ezv2-anti-collapse-v2",
            "Luna Balanced EZ-V2 · Anti-Collapse v2",
        ),
        (
            "migrate-ladder-phase",
            "never",
            "luna-fairy-ladder-v1",
            "Luna Fairy Ladder 500+ · Benchmark 1500 v1",
        ),
        (
            "resume-migrated-phase",
            "must",
            "luna-fairy-ladder-v1",
            "Luna Fairy Ladder 500+ · Benchmark 1500 v1",
        ),
    ],
)
def test_phase_make_target_sets_explicit_wandb_resume_policy(
    target: str,
    resume_mode: str,
    run_id: str,
    run_name: str,
) -> None:
    repository = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        ["make", "-n", target],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f'--wandb-run-id "{run_id}"' in result.stdout
    assert f'--wandb-run-name "{run_name}"' in result.stdout
    assert f"--wandb-resume {resume_mode}" in result.stdout
    assert "--run.self-play-workers 4" in result.stdout
    assert "--run.stockfish-elo 1500" in result.stdout
    assert "--run.ladder-start-elo 500" in result.stdout
    assert "--run.ladder-step-elo 100" in result.stdout
    assert "--learner.model-name balanced" in result.stdout
    assert "--learner.repr-blocks 10" in result.stdout
    assert "--learner.dyn-blocks 1" in result.stdout
    assert "--learner.unroll-steps 5" in result.stdout
    assert "--learner.td-steps 5" in result.stdout
    assert "--run.self-play-repetition-guard" in result.stdout
    assert "--run.target-replay-ratio 2.0" in result.stdout
    assert "--run.lr-schedule-total-steps 60000" in result.stdout
    assert "--run.replay-warmup-positions 50000" in result.stdout
    assert "--learner.reanalyze-mcts-sims 8" in result.stdout
    assert "--learner.reanalyze-prob 0.02" in result.stdout
    assert "--learner.no-reanalyze-policy" in result.stdout
    assert "--learner.reanalyze-start-step 10000" in result.stdout
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

    assert "luna-balanced-precollapse-iter40.pth.tar" in result.stdout
    assert "dd07d8ddf2aa652719b405b4e3b6f7381bb652873a34d139fe37b95327ba99dd" in result.stdout


def test_maintained_train_target_uses_balanced_ezv2_contract() -> None:
    repository = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        ["make", "-n", "train"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    expected_flags = (
        "--run.search-mode gumbel",
        "--run.gumbel-max-considered-actions 16",
        "--run.num-mcts-sims 32",
        "--run.self-play-workers 4",
        "--learner.model-name balanced",
        "--learner.batch-size 256",
        "--learner.repr-blocks 10",
        "--learner.dyn-blocks 1",
        "--learner.unroll-steps 5",
        "--learner.td-steps 5",
        "--learner.compile-inference",
        "--learner.compile-training",
        "--run.self-play-repetition-guard",
        "--run.target-replay-ratio 2.0",
        "--run.lr-schedule-total-steps 60000",
        "--run.replay-warmup-positions 50000",
        "--learner.reanalyze-mcts-sims 8",
        "--learner.reanalyze-prob 0.02",
        "--learner.no-reanalyze-policy",
        "--learner.reanalyze-start-step 10000",
    )
    for flag in expected_flags:
        assert flag in result.stdout
