from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import chess
import numpy as np

if TYPE_CHECKING:
    from luna.mcts import MCTS

MAX_PENDING_SEARCHES = 2
SEARCH_ADMISSION_TIMEOUT_SECONDS = 1.0
SEARCH_DEADLINE_SECONDS = 75.0

ColorName = Literal["white", "black"]
GameMode = Literal["human", "selfplay"]


@dataclass(frozen=True)
class StrengthProfile:
    name: str
    simulations: int
    description: str


@dataclass(frozen=True)
class EngineDecision:
    move: chess.Move
    san: str
    confidence: float
    evaluation_white: float
    think_time_ms: int
    simulations: int


class EngineBusyError(RuntimeError):
    pass


class LunaEngineService:
    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        device: str,
        search_simulations: int,
        compile_inference: bool,
    ) -> None:
        if search_simulations < 8:
            raise ValueError("search_simulations must be at least 8")

        from luna.game.chess_game import ChessGame
        from luna.network import LunaNetwork

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        self.game = ChessGame()
        self.network = LunaNetwork.from_checkpoint(
            self.game,
            checkpoint,
            device=device,
            compile_inference=compile_inference,
            load_optimizer=False,
        )
        self.network.nnet.eval()
        self.checkpoint_name = checkpoint.name
        self.device = device
        self._inference_lock = threading.Lock()
        self._search_slots = threading.BoundedSemaphore(MAX_PENDING_SEARCHES)
        self.strengths = _strength_profiles(search_simulations)
        if compile_inference:
            self.network.warmup_mcts_inference(self.game)

    def analyze(self, board: chess.Board, strength: str) -> EngineDecision:
        from luna.config import MCTSParams
        from luna.mcts import MCTS

        profile = self._profile(strength)
        if board.is_game_over(claim_draw=True):
            raise ValueError("The game is already over")

        player = 1 if board.turn == chess.WHITE else -1
        canonical_board = self.game.get_canonical_form(board, player)
        params = MCTSParams(
            num_mcts_sims=profile.simulations,
            cpuct=1.25,
            dir_noise=False,
            discount=1.0,
            recurrent_policy_topk=256,
        )
        started = time.monotonic()
        search = MCTS(self.game, self.network, params)
        probabilities, root_value = self._run_search(search, canonical_board, profile, started)
        action = self._selected_action(search.last_action, probabilities)
        next_board, _ = self.game.get_next_state(board, player, action)
        move = next_board.peek()
        evaluation_white = float(root_value if board.turn == chess.WHITE else -root_value)
        return EngineDecision(
            move=move,
            san=board.san(move),
            confidence=float(probabilities[action]),
            evaluation_white=float(np.clip(evaluation_white, -1.0, 1.0)),
            think_time_ms=round((time.monotonic() - started) * 1000),
            simulations=search.last_simulations,
        )

    def _profile(self, strength: str) -> StrengthProfile:
        profile = self.strengths.get(strength)
        if profile is None:
            raise ValueError(f"Unknown strength profile: {strength}")
        return profile

    def _run_search(
        self,
        search: MCTS,
        canonical_board: chess.Board,
        profile: StrengthProfile,
        started: float,
    ) -> tuple[list[float], float]:
        admitted = self._search_slots.acquire(timeout=SEARCH_ADMISSION_TIMEOUT_SECONDS)
        if not admitted:
            raise EngineBusyError("The inference queue is full")
        try:
            with self._inference_lock:
                return search.search_latent(
                    canonical_board,
                    num_sims=profile.simulations,
                    temp=1.0,
                    add_exploration_noise=False,
                    should_stop=lambda: time.monotonic() >= started + SEARCH_DEADLINE_SECONDS,
                )
        finally:
            self._search_slots.release()

    @staticmethod
    def _selected_action(last_action: int | None, probabilities: list[float]) -> int:
        if last_action is None or probabilities[last_action] <= 0:
            raise RuntimeError("Search returned no legal continuation")
        return last_action


def _strength_profiles(search_simulations: int) -> dict[str, StrengthProfile]:
    return {
        "quick": StrengthProfile(
            "Quick scan", max(8, search_simulations // 2), "Responsive play with a compact search."
        ),
        "strong": StrengthProfile(
            "Deep orbit",
            search_simulations,
            "The recommended balance of speed and calculation.",
        ),
        "maximum": StrengthProfile(
            "Event horizon",
            search_simulations * 2,
            "The deepest search and the longest think time.",
        ),
    }
