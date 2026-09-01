"""Tests for Coach self-play (e.g. max ply truncation, batched self-play)."""

import json
from hashlib import file_digest
from pathlib import Path
from typing import cast
from unittest.mock import call, patch

import pytest

from luna.coach import (
    Coach,
)
from luna.config import EzV2LearnerConfig, TrainingRunConfig, WandbResumeMode
from luna.game.chess_game import ChessGame
from luna.network import LunaNetwork


def test_wandb_metrics_use_domain_specific_step_axes(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    stockfish_binary: Path,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with (
        patch("luna.coach.wandb.init") as wandb_init,
        patch("luna.coach.wandb.define_metric") as define_metric,
    ):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(stockfish_path=str(stockfish_binary)),
            wandb_project="ChessRL",
        )

    init_kwargs = wandb_init.call_args.kwargs
    assert init_kwargs["project"] == "ChessRL"
    assert init_kwargs["name"] is None
    assert "id" not in init_kwargs
    assert "resume" not in init_kwargs
    assert init_kwargs["config"]["training_phase_provenance"] is None
    assert define_metric.call_args_list == [
        call("global_step"),
        call("train/*", step_metric="global_step"),
        call("iteration"),
        call("replay_buffer_size", step_metric="iteration"),
        call("selfplay/*", step_metric="iteration"),
        call("performance/*", step_metric="iteration"),
        call("replay/*", step_metric="iteration"),
        call("benchmark/*", step_metric="iteration"),
        call("ladder/evaluation_step"),
        call("ladder/*", step_metric="ladder/evaluation_step"),
    ]


def test_wandb_config_records_training_phase_source_without_private_path(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    stockfish_binary: Path,
) -> None:
    source = LunaNetwork(chess_game, small_learner_config)
    source._global_step = 123
    source._trainer_iteration = 17
    source.save_checkpoint(str(tmp_path), "source.pth.tar")
    source_path = tmp_path / "source.pth.tar"
    with source_path.open("rb") as source_file:
        expected_sha256 = file_digest(source_file, "sha256").hexdigest()
    phase = LunaNetwork(chess_game, small_learner_config)
    phase.initialize_training_phase(str(tmp_path), source_path.name)

    with (
        patch("luna.coach.wandb.init") as wandb_init,
        patch("luna.coach.wandb.define_metric"),
    ):
        Coach(
            chess_game,
            phase,
            TrainingRunConfig(stockfish_path=str(stockfish_binary)),
            wandb_project="ChessRL",
        )

    provenance_config = wandb_init.call_args.kwargs["config"]["training_phase_provenance"]
    assert provenance_config == {
        "source_checkpoint_sha256": expected_sha256,
        "source_trainer_iteration": 17,
        "source_global_step": 123,
    }
    serialized_config = json.dumps(provenance_config)
    assert str(tmp_path) not in serialized_config
    assert source_path.name not in serialized_config


@pytest.mark.parametrize("resume_mode", ["allow", "never", "must"])
def test_wandb_run_id_uses_requested_resume_policy(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    stockfish_binary: Path,
    resume_mode: WandbResumeMode,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with (
        patch("luna.coach.wandb.init") as wandb_init,
        patch("luna.coach.wandb.define_metric"),
    ):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(stockfish_path=str(stockfish_binary)),
            wandb_project="ChessRL",
            wandb_run_id="luna-throughput-phase-v1",
            wandb_resume=resume_mode,
        )

    init_kwargs = wandb_init.call_args.kwargs
    assert init_kwargs["project"] == "ChessRL"
    assert init_kwargs["id"] == "luna-throughput-phase-v1"
    assert init_kwargs["resume"] == resume_mode


def test_wandb_display_name_is_independent_of_stable_run_id(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    stockfish_binary: Path,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with (
        patch("luna.coach.wandb.init") as wandb_init,
        patch("luna.coach.wandb.define_metric"),
    ):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(stockfish_path=str(stockfish_binary)),
            wandb_project="ChessRL",
            wandb_run_id="luna-strength-1500-v1",
            wandb_run_name="Luna Strength 1500 v1",
            wandb_resume="never",
        )

    init_kwargs = wandb_init.call_args.kwargs
    assert init_kwargs["id"] == "luna-strength-1500-v1"
    assert init_kwargs["name"] == "Luna Strength 1500 v1"


@pytest.mark.parametrize("resume_mode", ["never", "must"])
def test_wandb_resume_policy_is_not_forwarded_without_run_id(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    stockfish_binary: Path,
    resume_mode: WandbResumeMode,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with (
        patch("luna.coach.wandb.init") as wandb_init,
        patch("luna.coach.wandb.define_metric"),
    ):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(stockfish_path=str(stockfish_binary)),
            wandb_project="ChessRL",
            wandb_resume=resume_mode,
        )

    init_kwargs = wandb_init.call_args.kwargs
    assert "id" not in init_kwargs
    assert "resume" not in init_kwargs


def test_coach_rejects_invalid_wandb_resume_policy(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with pytest.raises(ValueError, match="wandb_resume"):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(),
            wandb_resume=cast(WandbResumeMode, "sometimes"),
        )


@pytest.mark.parametrize("run_name", ["", "   ", " leading", "trailing "])
def test_coach_rejects_invalid_wandb_display_name(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    run_name: str,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)

    with pytest.raises(ValueError, match="wandb_run_name"):
        Coach(
            chess_game,
            network,
            TrainingRunConfig(),
            wandb_run_name=run_name,
        )
