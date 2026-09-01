from __future__ import annotations

import io
import threading
import time
from argparse import Namespace
from collections.abc import Callable, Collection
from os import PathLike

import chess
import numpy as np
import pytest

from luna.config import MCTSParams
from luna.game.chess_game import ACTION_SIZE, ChessGame, move_to_action
from luna.network import LunaNetwork
from luna.uci import LunaUciEngine, UciOptions


class _PolicyMCTS:
    policy: np.ndarray
    proposed_action: int | None = None
    allowed_root_actions: set[int] | None = None

    def __init__(self, game: ChessGame, network: LunaNetwork, params: MCTSParams) -> None:
        del game, network, params
        self.last_action: int | None = None

    def search_latent(
        self,
        board: chess.Board,
        temp: float,
        *,
        add_exploration_noise: bool,
        should_stop: Callable[[], bool] | None = None,
        allowed_root_actions: Collection[int] | None = None,
    ) -> tuple[np.ndarray, float]:
        del board, temp, should_stop
        assert add_exploration_noise is False
        type(self).allowed_root_actions = None if allowed_root_actions is None else set(allowed_root_actions)
        eligible = range(ACTION_SIZE) if allowed_root_actions is None else allowed_root_actions
        proposal_is_allowed = self.proposed_action is not None and self.proposed_action in eligible
        self.last_action = (
            self.proposed_action
            if proposal_is_allowed
            else max(eligible, key=lambda action: float(self.policy[action]))
        )
        self.last_simulations = 8
        return self.policy.copy(), 0.25


def _network_stub() -> LunaNetwork:
    network = object.__new__(LunaNetwork)
    network._mcts_inference_compiled = False
    return network


def _engine() -> LunaUciEngine:
    return LunaUciEngine(network=_network_stub(), options=UciOptions(mcts_simulations=64))


def test_uci_warms_once_after_announcing_protocol_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def record_warmup(self: LunaNetwork, game: ChessGame) -> None:
        del self, game
        events.append("warmup")

    engine = _engine()
    monkeypatch.setattr(LunaNetwork, "warmup_mcts_inference", record_warmup)
    monkeypatch.setattr(engine, "send", events.append)
    monkeypatch.setattr("sys.stdin", io.StringIO("uci\nisready\nisready\nquit\n"))

    engine.run()

    assert events.count("warmup") == 1
    assert events.count("readyok") == 2
    assert events.index("uciok") < events.index("warmup")


def test_stop_interrupts_background_search_and_emits_one_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luna import uci

    action = move_to_action(chess.Move.from_uci("e2e4"))

    class _InterruptibleMCTS:
        def __init__(self, game: ChessGame, network: LunaNetwork, params: MCTSParams) -> None:
            del game, network, params
            self.last_action = action
            self.last_simulations = 0

        def search_latent(
            self,
            board: chess.Board,
            temp: float,
            *,
            add_exploration_noise: bool,
            should_stop: Callable[[], bool],
            allowed_root_actions: Collection[int] | None = None,
        ) -> tuple[np.ndarray, float]:
            del board, temp, add_exploration_noise, allowed_root_actions
            while not should_stop():
                self.last_simulations += 1
                time.sleep(0.001)
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy[action] = 1.0
            return policy, 0.0

    monkeypatch.setattr(uci, "MCTS", _InterruptibleMCTS)
    engine = _engine()
    output: list[str] = []
    monkeypatch.setattr(engine, "send", output.append)
    monkeypatch.setattr("sys.stdin", io.StringIO("go infinite\nstop\nquit\n"))

    engine.run()

    assert output.count("bestmove e2e4") == 1
    assert engine._search_thread is None


def test_infinite_search_waits_for_stop_after_finite_search_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luna import uci

    action = move_to_action(chess.Move.from_uci("e2e4"))
    search_completed = threading.Event()

    class _FiniteMCTS:
        def __init__(self, game: ChessGame, network: LunaNetwork, params: MCTSParams) -> None:
            del game, network, params
            self.last_action = action
            self.last_simulations = 8

        def search_latent(
            self,
            board: chess.Board,
            temp: float,
            *,
            add_exploration_noise: bool,
            should_stop: Callable[[], bool],
            allowed_root_actions: Collection[int] | None = None,
        ) -> tuple[np.ndarray, float]:
            del board, temp, add_exploration_noise, should_stop, allowed_root_actions
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy[action] = 1.0
            search_completed.set()
            return policy, 0.0

    monkeypatch.setattr(uci, "MCTS", _FiniteMCTS)
    engine = _engine()
    output: list[str] = []
    monkeypatch.setattr(engine, "send", output.append)

    engine._start_search(["infinite"])
    assert search_completed.wait(timeout=1.0)
    assert not any(message.startswith("bestmove") for message in output)
    assert engine._search_thread is not None and engine._search_thread.is_alive()

    engine._stop_active_search()

    assert output.count("bestmove e2e4") == 1
    assert engine._search_thread is None


def test_timed_search_stops_at_monotonic_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from luna import uci

    action = move_to_action(chess.Move.from_uci("e2e4"))

    class _DeadlineMCTS:
        deadline_reached = False

        def __init__(self, game: ChessGame, network: LunaNetwork, params: MCTSParams) -> None:
            del game, network, params
            self.last_action = action
            self.last_simulations = 0

        def search_latent(
            self,
            board: chess.Board,
            temp: float,
            *,
            add_exploration_noise: bool,
            should_stop: Callable[[], bool],
            allowed_root_actions: Collection[int] | None,
        ) -> tuple[np.ndarray, float]:
            del board, temp, add_exploration_noise, allowed_root_actions
            type(self).deadline_reached = should_stop()
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy[action] = 1.0
            return policy, 0.0

    ticks = iter((10.0, 10.0, 10.002, 10.003))

    def monotonic() -> float:
        return next(ticks)

    monkeypatch.setattr(uci.time, "monotonic", monotonic)
    monkeypatch.setattr(uci, "MCTS", _DeadlineMCTS)
    engine = _engine()
    output: list[str] = []
    monkeypatch.setattr(engine, "send", output.append)

    engine.search(["movetime", "1"])

    assert _DeadlineMCTS.deadline_reached
    assert output[-1] == "bestmove e2e4"


def test_search_worker_emits_fallback_before_unexpected_error_escapes(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine()
    output: list[str] = []

    def fail_search(go_tokens: list[str], stop_event: threading.Event | None = None) -> None:
        del go_tokens, stop_event
        raise TypeError("unexpected search failure")

    monkeypatch.setattr(engine, "send", output.append)
    monkeypatch.setattr(engine, "search", fail_search)

    with pytest.raises(TypeError, match="unexpected search failure"):
        engine._search_worker([], threading.Event())

    assert output == ["bestmove 0000"]


def test_main_starts_protocol_without_eager_inference_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    from luna import uci

    network = _network_stub()
    warmed_games: list[ChessGame] = []
    protocol_runs: list[LunaUciEngine] = []
    args = Namespace(
        checkpoint="checkpoint.pth.tar",
        device="cpu",
        cuda_device=None,
        mcts_sims=8,
        minimum_sims=2,
        estimated_sim_ms=3.0,
        compile_inference=True,
    )

    def parse_args() -> Namespace:
        return args

    def load_checkpoint(
        cls: type[LunaNetwork],
        game: ChessGame,
        checkpoint_path: str | PathLike[str],
        *,
        device: str = "cuda",
        cuda_device: int | None = None,
        compile_inference: bool = False,
        load_optimizer: bool = False,
    ) -> LunaNetwork:
        del cls, game, checkpoint_path, device, cuda_device, compile_inference, load_optimizer
        return network

    def record_warmup(self: LunaNetwork, game: ChessGame) -> None:
        del self
        warmed_games.append(game)

    def record_protocol_run(self: LunaUciEngine) -> None:
        protocol_runs.append(self)

    monkeypatch.setattr(uci, "_parse_args", parse_args)
    monkeypatch.setattr(
        uci.LunaNetwork,
        "from_checkpoint",
        classmethod(load_checkpoint),
    )
    monkeypatch.setattr(
        uci.LunaNetwork,
        "warmup_mcts_inference",
        record_warmup,
    )
    monkeypatch.setattr(uci.LunaUciEngine, "run", record_protocol_run)

    result = uci.main()

    assert result == 0
    assert warmed_games == []
    assert len(protocol_runs) == 1
