"""Focused recovery tests for online training checkpoints."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import luna.online_checkpoints as online_checkpoints
from luna.coach import Coach
from luna.coach_checkpoints import numbered_checkpoints, publish_bootstrap_checkpoint
from luna.config import EzV2LearnerConfig, TrainingRunConfig
from luna.game.chess_game import ChessGame
from luna.network import LunaNetwork
from luna.network_types import TrainingPhaseProvenance


def test_corrupt_latest_falls_back_to_numbered_and_heals_alias(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    latest = tmp_path / "latest.pth.tar"
    numbered = tmp_path / "checkpoint_12.pth.tar"
    network = LunaNetwork(chess_game, small_learner_config)
    network._trainer_iteration = 12
    network.save_checkpoint(str(tmp_path), numbered.name)
    latest.write_bytes(b"")

    selected = online_checkpoints.resolve_resume_checkpoint(latest, tmp_path)

    assert selected == numbered
    assert latest.read_bytes() == numbered.read_bytes()
    assert LunaNetwork.checkpoint_trainer_iteration(latest) == 12


def test_same_iteration_immutable_checkpoint_heals_valid_but_divergent_latest(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    network._trainer_iteration = 12
    numbered = tmp_path / "checkpoint_12.pth.tar"
    latest = tmp_path / "latest.pth.tar"
    network.save_checkpoint(str(tmp_path), numbered.name)
    with torch.no_grad():
        next(network.nnet.parameters()).add_(1.0)
    network.save_checkpoint(str(tmp_path), latest.name)
    assert latest.read_bytes() != numbered.read_bytes()

    selected = online_checkpoints.resolve_resume_checkpoint(latest, tmp_path)

    assert selected == numbered
    assert latest.read_bytes() == numbered.read_bytes()


def test_numbered_lineage_replaces_newer_latest_from_another_phase(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    numbered_provenance = TrainingPhaseProvenance("a" * 64, 4, 40)
    network._training_phase_provenance = numbered_provenance
    network._trainer_iteration = 11
    numbered = tmp_path / "checkpoint_11.pth.tar"
    network.save_checkpoint(str(tmp_path), numbered.name)
    network._training_phase_provenance = TrainingPhaseProvenance("b" * 64, 5, 50)
    network._trainer_iteration = 12
    latest = tmp_path / "latest.pth.tar"
    network.save_checkpoint(str(tmp_path), latest.name)

    selected = online_checkpoints.resolve_resume_checkpoint(latest, tmp_path)

    healed = LunaNetwork.from_checkpoint(chess_game, latest, device="cpu", load_optimizer=True)
    assert selected == numbered
    assert healed.trainer_iteration == 11
    assert healed.training_phase_provenance == numbered_provenance


def test_mixed_healthy_numbered_lineages_fail_without_mutation(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    first = tmp_path / "checkpoint_10.pth.tar"
    second = tmp_path / "checkpoint_11.pth.tar"
    network._training_phase_provenance = TrainingPhaseProvenance("a" * 64, 4, 40)
    network._trainer_iteration = 10
    network.save_checkpoint(str(tmp_path), first.name)
    network._training_phase_provenance = TrainingPhaseProvenance("b" * 64, 5, 50)
    network._trainer_iteration = 11
    network.save_checkpoint(str(tmp_path), second.name)
    before = first.read_bytes(), second.read_bytes()

    with pytest.raises(RuntimeError, match="mixed training-phase lineage"):
        online_checkpoints.resolve_resume_checkpoint(tmp_path / "latest.pth.tar", tmp_path)

    assert (first.read_bytes(), second.read_bytes()) == before
    assert not (tmp_path / "latest.pth.tar").exists()


def test_recovery_quarantines_corrupt_numbered_checkpoint_without_losing_diagnostics(tmp_path: Path) -> None:
    latest = tmp_path / "latest.pth.tar"
    healthy = tmp_path / "checkpoint_11.pth.tar"
    corrupt = tmp_path / "checkpoint_12.pth.tar"
    latest.write_bytes(b"stale")
    healthy.write_bytes(b"healthy")
    corrupt.write_bytes(b"corrupt-numbered")
    previous_quarantine = tmp_path / "checkpoint_12.pth.tar.invalid"
    previous_quarantine.write_bytes(b"previous-diagnostic")

    def checkpoint_iteration(path: Path) -> online_checkpoints._CheckpointIdentity:
        iterations = {latest.name: 10, healthy.name: 11}
        if Path(path) == corrupt:
            raise EOFError("truncated checkpoint")
        return online_checkpoints._CheckpointIdentity(iterations[Path(path).name], None)

    with patch.object(
        online_checkpoints,
        "_validated_checkpoint_identity",
        side_effect=checkpoint_iteration,
    ):
        selected = online_checkpoints.resolve_resume_checkpoint(latest, tmp_path)

    assert selected == healthy
    assert latest.read_bytes() == b"healthy"
    assert not corrupt.exists()
    assert previous_quarantine.read_bytes() == b"previous-diagnostic"
    assert (tmp_path / "checkpoint_12.pth.tar.invalid-1").read_bytes() == b"corrupt-numbered"


def test_recovery_fails_without_mutating_when_every_candidate_is_corrupt(tmp_path: Path) -> None:
    latest = tmp_path / "latest.pth.tar"
    numbered = tmp_path / "checkpoint_3.pth.tar"
    latest.write_bytes(b"bad-latest")
    numbered.write_bytes(b"bad-numbered")

    with (
        patch.object(
            online_checkpoints,
            "_validated_checkpoint_identity",
            side_effect=RuntimeError("unreadable"),
        ),
        pytest.raises(RuntimeError, match="No healthy resumable checkpoint"),
    ):
        online_checkpoints.resolve_resume_checkpoint(latest, tmp_path)

    assert latest.read_bytes() == b"bad-latest"
    assert numbered.read_bytes() == b"bad-numbered"


def test_recovery_quarantines_invalid_optimizer_then_accepts_lineage_and_next_checkpoint(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    latest = tmp_path / "latest.pth.tar"
    healthy = tmp_path / "checkpoint_11.pth.tar"
    invalid = tmp_path / "checkpoint_12.pth.tar"
    for iteration, path in ((11, latest), (11, healthy), (12, invalid)):
        network._trainer_iteration = iteration
        network.save_checkpoint(str(tmp_path), path.name)
    payload = torch.load(invalid, map_location="cpu", weights_only=True)
    payload["optimizer"] = {"invalid": True}
    torch.save(payload, invalid)
    invalid_bytes = invalid.read_bytes()

    selected = online_checkpoints.resolve_resume_checkpoint(latest, tmp_path)

    assert selected == healthy
    assert LunaNetwork.checkpoint_trainer_iteration(latest) == 11
    quarantine = tmp_path / "checkpoint_12.pth.tar.invalid"
    assert not invalid.exists()
    assert quarantine.read_bytes() == invalid_bytes

    resumed = LunaNetwork.from_checkpoint(chess_game, selected, device="cpu", load_optimizer=True)
    coach = Coach(
        chess_game,
        resumed,
        TrainingRunConfig(num_iters=12, checkpoint=str(tmp_path), stockfish_eval_every=0),
    )
    coach._assert_checkpoint_target()
    coach._assert_checkpoint_lineage()
    coach._publish_checkpoint(12)

    assert LunaNetwork.checkpoint_trainer_iteration(invalid) == 12
    assert [iteration for iteration, _path in numbered_checkpoints(tmp_path)] == [11, 12]
    assert quarantine.read_bytes() == invalid_bytes


def test_stale_bootstrap_temporary_does_not_trap_a_fresh_phase(tmp_path: Path) -> None:
    target = tmp_path / "online"
    target.mkdir()
    temporary = target / "checkpoint_0.pth.tar.tmp-1234"
    temporary.write_bytes(b"interrupted-write")

    online_checkpoints.validate_new_training_phase_target(str(target))

    assert temporary.read_bytes() == b"interrupted-write"
    (target / "notes.txt").write_text("unrelated", encoding="utf-8")
    with pytest.raises(FileExistsError, match="requires an empty checkpoint directory"):
        online_checkpoints.validate_new_training_phase_target(str(target))


def test_new_phase_bootstrap_is_immediately_resumable_and_accepted_by_coach(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    source_dir = tmp_path / "source"
    source = LunaNetwork(chess_game, small_learner_config)
    source._global_step = 17
    source._trainer_iteration = 4
    source.save_checkpoint(str(source_dir), "source.pth.tar")

    phase = LunaNetwork(chess_game, small_learner_config)
    phase.initialize_training_phase(str(source_dir), "source.pth.tar")
    target = tmp_path / "online"
    checkpoint = publish_bootstrap_checkpoint(phase, str(target))

    resumed = LunaNetwork(chess_game, small_learner_config)
    resumed.load_checkpoint(str(target), "latest.pth.tar")
    assert checkpoint == target / "checkpoint_0.pth.tar"
    assert checkpoint.read_bytes() == (target / "latest.pth.tar").read_bytes()
    assert resumed.global_step == 0
    assert resumed.trainer_iteration == 0
    assert resumed.training_phase_provenance == phase.training_phase_provenance
    assert phase._loaded_checkpoint_path == checkpoint

    coach = Coach(
        chess_game,
        phase,
        TrainingRunConfig(num_iters=1, checkpoint=str(target), stockfish_eval_every=0),
    )
    with patch("luna.coach_training._run_with_self_play_actors") as run_self_play:
        coach.learn()

    run_self_play.assert_called_once_with(coach, 1)
