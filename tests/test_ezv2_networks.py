"""Tests for EfficientZeroV2 model components."""

from __future__ import annotations

import torch

from luna.config import EzV2LearnerConfig
from luna.ezv2_networks import (
    EZV2Networks,
    _support_to_scalar,
    action_index_to_planes,
    scalar_to_support,
)
from luna.game.chess_game import OBS_PLANES, ChessGame

ACTION_SIZE = ChessGame().get_action_size()


def _make_nnet() -> EZV2Networks:
    game = ChessGame()
    cfg = EzV2LearnerConfig(num_channels=32, support_size=5, repr_blocks=2, dyn_blocks=1, proj_dim=64)
    return EZV2Networks(game, cfg)


class TestInitialInference:
    def test_output_shapes(self) -> None:
        nnet = _make_nnet()
        obs = torch.randn(2, 8, 8, OBS_PLANES)
        valid = torch.ones(2, ACTION_SIZE)
        log_pi, v = nnet.initial_inference(obs, valid)
        assert log_pi.shape == (2, ACTION_SIZE)
        assert v.shape == (2,)
        assert (log_pi <= 0).all()

    def test_with_latent(self) -> None:
        nnet = _make_nnet()
        obs = torch.randn(1, 8, 8, OBS_PLANES)
        latent, _log_pi, _v = nnet.initial_inference_with_latent(obs)
        assert latent.shape[0] == 1
        assert latent.shape[2] == 8
        assert latent.shape[3] == 8


class TestRecurrentInference:
    def test_output_shapes(self) -> None:
        nnet = _make_nnet()
        obs = torch.randn(2, 8, 8, OBS_PLANES)
        latent, _, _ = nnet.initial_inference_with_latent(obs)
        act_planes = action_index_to_planes(torch.tensor([0, 100]), latent.device)
        next_latent, reward, log_pi, v = nnet.recurrent_inference(latent, act_planes)
        assert next_latent.shape == latent.shape
        assert reward.shape == (2,)
        assert log_pi.shape == (2, ACTION_SIZE)
        assert v.shape == (2,)


class TestSupportTransform:
    def test_roundtrip(self) -> None:
        support_size = 5
        values = torch.tensor([0.0, 1.0, -1.0, 3.5, -4.9])
        encoded = scalar_to_support(values, support_size)
        assert encoded.shape == (5, 2 * support_size + 1)
        assert torch.allclose(encoded.sum(dim=1), torch.ones(5), atol=1e-5)

        logits = encoded.log().clamp(min=-30)
        recovered = _support_to_scalar(logits * 100, support_size)
        assert torch.allclose(values.clamp(-support_size, support_size), recovered, atol=0.1)


class TestSimSiamProjector:
    def test_output_shapes(self) -> None:
        nnet = _make_nnet()
        latent = torch.randn(4, 32, 8, 8)
        z = nnet.simsiam.project(latent)
        assert z.shape == (4, 64)
        p = nnet.simsiam.predict(z)
        assert p.shape == (4, 64)


class TestActionPlanes:
    def test_spatial_encoding(self) -> None:
        actions = torch.tensor([0, 4095, 100])
        planes = action_index_to_planes(actions, torch.device("cpu"))
        assert planes.shape == (3, 2, 8, 8)
        assert planes[0, 0].sum().item() == 1.0
        assert planes[0, 1].sum().item() == 1.0
