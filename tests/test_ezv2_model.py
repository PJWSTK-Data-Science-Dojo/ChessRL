"""Tests for EfficientZeroV2 model components."""

from __future__ import annotations

import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from luna.ezv2_model import EZV2Networks, scalar_to_support, _support_to_scalar
from luna.game.luna_game import ChessGame
from luna.utils import dotdict


def _make_nnet() -> EZV2Networks:
    game = ChessGame()
    args = dotdict({"num_channels": 32, "support_size": 5, "repr_blocks": 2, "dyn_blocks": 1})
    return EZV2Networks(game, args)


class TestInitialInference:
    def test_output_shapes(self) -> None:
        nnet = _make_nnet()
        obs = torch.randn(2, 8, 8, 6)
        valid = torch.ones(2, 4096)
        log_pi, v = nnet.initial_inference(obs, valid)
        assert log_pi.shape == (2, 4096)
        assert v.shape == (2,)
        assert (log_pi <= 0).all()

    def test_with_latent(self) -> None:
        nnet = _make_nnet()
        obs = torch.randn(1, 8, 8, 6)
        latent, log_pi, v = nnet.initial_inference_with_latent(obs)
        assert latent.shape[0] == 1
        assert latent.shape[2] == 8
        assert latent.shape[3] == 8


class TestRecurrentInference:
    def test_output_shapes(self) -> None:
        nnet = _make_nnet()
        obs = torch.randn(2, 8, 8, 6)
        latent, _, _ = nnet.initial_inference_with_latent(obs)
        action_oh = torch.zeros(2, 4096)
        action_oh[:, 0] = 1.0
        next_latent, reward, log_pi, v = nnet.recurrent_inference(latent, action_oh)
        assert next_latent.shape == latent.shape
        assert reward.shape == (2,)
        assert log_pi.shape == (2, 4096)
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
