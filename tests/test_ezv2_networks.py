"""Tests for EfficientZeroV2 model components."""

from dataclasses import replace

import chess
import pytest
import torch

from luna.balanced_networks import (
    BalancedDynamicsNetwork,
    BalancedNetworks,
    BalancedReconstructionNetworks,
    LayerNorm2d,
    PieceReconstructionHead,
    SEResBlock,
)
from luna.config import EzV2LearnerConfig
from luna.ezv2_networks import (
    EZV2Networks,
    PredictionNetwork,
    _flatten_spatial_policy,
    _scale_latent,
    _support_to_scalar,
    action_index_to_planes,
    scalar_to_support,
)
from luna.game.chess_game import OBS_PLANES, ChessGame
from luna.model_factory import available_models, build_model


@pytest.mark.parametrize("network_type", ["initial", "recurrent"])
def test_network_output_shapes(
    network_type: str,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    """All networks produce correct output tensor shapes."""
    nnet = EZV2Networks(chess_game, small_learner_config)
    action_size = chess_game.get_action_size()

    if network_type == "initial":
        obs = torch.randn(2, 8, 8, OBS_PLANES)
        valid = torch.ones(2, action_size)
        log_pi, v = nnet.initial_inference(obs, valid)
        assert log_pi.shape == (2, action_size)
        assert v.shape == (2,)
        assert (log_pi <= 0).all()
    else:
        obs = torch.randn(2, 8, 8, OBS_PLANES)
        latent, _, _ = nnet.initial_inference_with_latent(obs)
        act_planes = action_index_to_planes(torch.tensor([0, 100]), latent.device)
        next_latent, reward, log_pi, v = nnet.recurrent_inference(latent, act_planes)
        assert next_latent.shape == latent.shape
        assert reward.shape == (2,)
        assert log_pi.shape == (2, action_size)
        assert v.shape == (2,)


def test_support_transform_roundtrip() -> None:
    support_size = 5
    values = torch.tensor([0.0, 1.0, -1.0, 3.5, -4.9])
    encoded = scalar_to_support(values, support_size)
    assert encoded.shape == (5, 2 * support_size + 1)
    assert torch.allclose(encoded.sum(dim=1), torch.ones(5), atol=1e-5)

    logits = encoded.log().clamp(min=-30)
    recovered = _support_to_scalar(logits * 100, support_size)
    assert torch.allclose(values.clamp(-support_size, support_size), recovered, atol=0.1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_latent_scaling_uses_stable_statistics_and_preserves_dtype(dtype: torch.dtype) -> None:
    latent = torch.linspace(-3.0, 5.0, 2 * 8 * 8 * 8, dtype=dtype).reshape(2, 8, 8, 8)

    scaled = _scale_latent(latent)
    flat = scaled.float().flatten(1)

    assert scaled.dtype == dtype
    torch.testing.assert_close(flat.mean(dim=1), torch.zeros(2), atol=2e-2, rtol=0.0)
    torch.testing.assert_close(flat.std(dim=1, correction=0), torch.ones(2), atol=2e-2, rtol=0.0)


def test_action_spatial_encoding() -> None:
    actions = torch.tensor([0, 4095, 100])
    planes = action_index_to_planes(actions, torch.device("cpu"))
    assert planes.shape == (3, 5, 8, 8)
    assert planes[0, 0].sum().item() == 1.0
    assert planes[0, 1].sum().item() == 1.0


def test_policy_mask_is_finite_for_float16_logits() -> None:
    prediction = PredictionNetwork(channels=8, action_size=4288, support_size=1).half()
    latent = torch.randn(1, 8, 8, 8, dtype=torch.float16)
    valid = torch.zeros(1, 4288, dtype=torch.float16)
    valid[:, :2] = 1

    policy_logits, _value_logits = prediction(latent, valid)
    log_policy = torch.log_softmax(policy_logits, dim=1)

    assert torch.isfinite(log_policy).all()
    assert torch.count_nonzero(torch.exp(log_policy[:, 2:])) == 0


def test_spatial_policy_head_preserves_action_layout() -> None:
    raw_logits = torch.zeros(1, 88, 8, 8)

    normal_action = chess.E2 * 64 + chess.E4
    raw_logits[0, chess.E4, chess.square_rank(chess.E2), chess.square_file(chess.E2)] = 3.0

    knight_promotion_action = 4096 + chess.square_file(chess.B7) * 8 + chess.square_file(chess.B8)
    promotion_channel = 64 + chess.square_file(chess.B8)
    raw_logits[0, promotion_channel, 6, chess.square_file(chess.B7)] = 5.0

    policy_logits = _flatten_spatial_policy(raw_logits)

    assert policy_logits.shape == (1, 4288)
    assert policy_logits[0, normal_action].item() == 3.0
    assert policy_logits[0, knight_promotion_action].item() == 5.0


def test_model_factory_builds_all_registered_architectures(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    assert available_models() == ("baseline", "balanced", "balanced_reconstruction")

    baseline = build_model(chess_game, small_learner_config)
    balanced = build_model(chess_game, replace(small_learner_config, model_name="balanced"))
    reconstructed = build_model(
        chess_game,
        replace(small_learner_config, model_name="balanced_reconstruction"),
    )

    assert type(baseline) is EZV2Networks
    assert type(balanced) is BalancedNetworks
    assert isinstance(reconstructed, BalancedReconstructionNetworks)
    assert isinstance(reconstructed.piece_reconstruction, PieceReconstructionHead)


def test_state_anchored_model_decodes_piece_classes_only_during_training(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    config = replace(small_learner_config, model_name="balanced_reconstruction")
    model = build_model(chess_game, config)
    assert model.piece_reconstruction is not None

    observation = torch.randn(2, 8, 8, OBS_PLANES)
    latent, log_policy, value = model.initial_inference_with_latent(observation)
    piece_logits = model.piece_reconstruction(latent)

    assert piece_logits.shape == (2, 13, 8, 8)
    assert log_policy.shape == (2, chess_game.get_action_size())
    assert value.shape == (2,)


def test_state_anchor_decoder_is_not_called_by_mcts_inference(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    class ForbiddenDecoder(torch.nn.Module):
        def forward(self, _latent: torch.Tensor) -> torch.Tensor:
            raise AssertionError("training-only decoder was called by inference")

    config = replace(small_learner_config, model_name="balanced_reconstruction")
    model = build_model(chess_game, config)
    model.piece_reconstruction = ForbiddenDecoder()
    observation = torch.randn(2, 8, 8, OBS_PLANES)
    valid = torch.ones(2, chess_game.get_action_size())

    latent, _log_policy, _value = model.initial_inference_with_latent(observation, valid)
    actions = action_index_to_planes(torch.tensor([0, 100]), latent.device)
    _ = model.recurrent_inference(latent, actions, valid)


def test_balanced_model_uses_dense_asymmetric_se_trunks(
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    config = replace(small_learner_config, model_name="balanced", repr_blocks=3, dyn_blocks=1)
    model = build_model(chess_game, config)

    assert len(model.representation.blocks) == 3
    assert len(model.dynamics.blocks) == 1
    assert isinstance(model.representation.norm_in, LayerNorm2d)
    assert isinstance(model.dynamics.blocks[0], SEResBlock)
    assert isinstance(model.dynamics, BalancedDynamicsNetwork)
    assert model.dynamics.conv_in.groups == 1
    assert model.simsiam.pool_spatial

    observation = torch.randn(2, 8, 8, OBS_PLANES)
    valid = torch.ones(2, chess_game.get_action_size())
    latent, log_policy, value = model.initial_inference_with_latent(observation, valid)
    actions = action_index_to_planes(torch.tensor([0, 4096]), latent.device)
    next_latent, reward, recurrent_policy, recurrent_value = model.recurrent_inference(
        latent,
        actions,
        valid,
    )

    assert latent.shape == next_latent.shape == (2, config.num_channels, 8, 8)
    assert log_policy.shape == recurrent_policy.shape == (2, chess_game.get_action_size())
    assert value.shape == reward.shape == recurrent_value.shape == (2,)


def test_layer_norm_2d_preserves_channels_last_layout() -> None:
    normalized = LayerNorm2d(16)(torch.randn(2, 16, 8, 8))

    assert normalized.is_contiguous(memory_format=torch.channels_last)
