"""Tests for EfficientZeroV2 model components."""

import chess
import pytest
import torch

from luna.ezv2_networks import (
    EZV2Networks,
    _flatten_spatial_policy,
    _support_to_scalar,
    action_index_to_planes,
    scalar_to_support,
)
from luna.game.chess_game import OBS_PLANES


@pytest.mark.parametrize("network_type", ["initial", "recurrent"])
def test_network_output_shapes(network_type, chess_game, small_learner_config):
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


def test_support_transform_roundtrip():
    support_size = 5
    values = torch.tensor([0.0, 1.0, -1.0, 3.5, -4.9])
    encoded = scalar_to_support(values, support_size)
    assert encoded.shape == (5, 2 * support_size + 1)
    assert torch.allclose(encoded.sum(dim=1), torch.ones(5), atol=1e-5)

    logits = encoded.log().clamp(min=-30)
    recovered = _support_to_scalar(logits * 100, support_size)
    assert torch.allclose(values.clamp(-support_size, support_size), recovered, atol=0.1)


def test_action_spatial_encoding():
    actions = torch.tensor([0, 4095, 100])
    planes = action_index_to_planes(actions, torch.device("cpu"))
    assert planes.shape == (3, 5, 8, 8)
    assert planes[0, 0].sum().item() == 1.0
    assert planes[0, 1].sum().item() == 1.0


def test_spatial_policy_head_preserves_action_layout():
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
