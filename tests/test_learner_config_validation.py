"""Validation tests for EfficientZeroV2 learner configuration."""

import pytest

from luna.config import EzV2LearnerConfig
from luna.game.chess_game import ChessGame
from luna.network import LunaNetwork


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("model_name", "unknown", "model_name must be one of"),
        ("grad_accum_steps", 0, "grad_accum_steps must be a positive integer"),
        ("dataloader_workers", -1, "dataloader_workers must be a non-negative integer"),
        ("support_size", 0, "support_size must be a positive integer"),
        ("grad_clip_norm", 0.0, "grad_clip_norm must be positive and finite"),
        ("grad_clip_norm", float("inf"), "grad_clip_norm must be positive and finite"),
        ("lr", float("nan"), "lr must be finite"),
        ("reanalyze_prob", 1.1, "reanalyze_prob must be between 0 and 1"),
        ("consistency_loss_weight", float("inf"), "consistency_loss_weight must be finite"),
        ("reconstruction_loss_weight", float("inf"), "reconstruction_loss_weight must be finite"),
    ],
)
def test_invalid_execution_settings_fail_loudly(
    small_learner_config: EzV2LearnerConfig,
    field_name: str,
    value: object,
    message: str,
) -> None:
    setattr(small_learner_config, field_name, value)

    with pytest.raises(ValueError, match=message):
        LunaNetwork(ChessGame(), small_learner_config)


def test_learner_accepts_root_only_objective(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.unroll_steps = 0
    small_learner_config.reward_loss_weight = 0.0
    small_learner_config.consistency_loss_weight = 0.0

    network = LunaNetwork(ChessGame(), small_learner_config)

    assert network._learner.unroll_steps == 0


@pytest.mark.parametrize("loss_name", ["reward_loss_weight", "consistency_loss_weight"])
def test_root_only_objective_rejects_recurrent_losses(
    small_learner_config: EzV2LearnerConfig,
    loss_name: str,
) -> None:
    small_learner_config.unroll_steps = 0
    small_learner_config.reward_loss_weight = 0.0
    small_learner_config.consistency_loss_weight = 0.0
    setattr(small_learner_config, loss_name, 0.1)

    with pytest.raises(
        ValueError,
        match="unroll_steps=0 requires reward_loss_weight=0 and consistency_loss_weight=0",
    ):
        LunaNetwork(ChessGame(), small_learner_config)


def test_reconstruction_loss_requires_matching_model(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.reconstruction_loss_weight = 0.5

    with pytest.raises(ValueError, match="requires model_name='balanced_reconstruction'"):
        LunaNetwork(ChessGame(), small_learner_config)


def test_learner_requires_equal_accumulation_microbatches(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.batch_size = 3
    small_learner_config.grad_accum_steps = 2

    with pytest.raises(ValueError, match="batch_size must be divisible by grad_accum_steps"):
        LunaNetwork(ChessGame(), small_learner_config)
