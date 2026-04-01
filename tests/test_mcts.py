"""Tests for MCTS search."""

from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from luna.game.luna_game import ChessGame
from luna.NNet import Luna_Network
from luna.mcts import MCTS
from luna.utils import dotdict


def _make_mcts(num_sims: int = 5) -> tuple[MCTS, ChessGame]:
    game = ChessGame()
    nnet = Luna_Network(game)
    args = dotdict({"numMCTSSims": num_sims, "cpuct": 1.25, "dir_noise": False, "dir_alpha": 0.3})
    return MCTS(game, nnet, args), game


class TestRealBoardSearch:
    def test_returns_valid_policy(self) -> None:
        mcts, game = _make_mcts(num_sims=3)
        board = game.getInitBoard()
        canonical = game.getCanonicalForm(board, 1)
        probs = mcts.getActionProb(canonical, temp=1)
        assert len(probs) == game.getActionSize()
        assert abs(sum(probs) - 1.0) < 1e-5

    def test_deterministic_at_temp_zero(self) -> None:
        mcts, game = _make_mcts(num_sims=3)
        board = game.getInitBoard()
        canonical = game.getCanonicalForm(board, 1)
        probs = mcts.getActionProb(canonical, temp=0)
        assert sum(1 for p in probs if p > 0) == 1


class TestLatentSearch:
    def test_returns_valid_policy(self) -> None:
        mcts, game = _make_mcts(num_sims=3)
        board = game.getInitBoard()
        canonical = game.getCanonicalForm(board, 1)
        probs, root_v = mcts.search_latent(canonical, num_sims=3)
        assert len(probs) == game.getActionSize()
        assert abs(sum(probs) - 1.0) < 1e-5
        assert isinstance(root_v, float)
