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
from luna.game.chess_game import ACTION_SIZE, ChessGame, mirror_move, move_to_action
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
    return object.__new__(LunaNetwork)


def _engine() -> LunaUciEngine:
    return LunaUciEngine(network=_network_stub(), options=UciOptions(mcts_simulations=64))


def test_position_accepts_startpos_and_fen_move_lists() -> None:
    engine = _engine()
    engine.set_position(["startpos", "moves", "e2e4", "c7c5"])
    assert engine.board.peek() == chess.Move.from_uci("c7c5")
    assert engine.board.turn == chess.WHITE

    engine.set_position(
        [
            "fen",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
            "w",
            "KQkq",
            "-",
            "0",
            "1",
            "moves",
            "d2d4",
        ]
    )
    assert engine.board.peek() == chess.Move.from_uci("d2d4")


def test_time_management_caps_and_floors_simulation_budget() -> None:
    engine = LunaUciEngine(
        network=_network_stub(),
        options=UciOptions(mcts_simulations=100, minimum_simulations=8, estimated_simulation_ms=5.0),
    )
    assert engine._simulation_budget(["movetime", "2025"]) == 100
    assert engine._simulation_budget(["movetime", "1"]) == 8
    assert engine._simulation_budget(["wtime", "3000", "winc", "0"]) == 20
    assert engine._simulation_budget(["nodes", "12"]) == 12
    assert engine._simulation_budget(["nodes", "1"]) == 1
    assert engine._search_limits(["nodes", "12", "movetime", "50"]).deadline is not None
    with pytest.raises(ValueError, match="requires an integer"):
        engine._simulation_budget(["movetime", "invalid"])
    with pytest.raises(ValueError, match="requires an integer"):
        engine._simulation_budget(["movetime"])

    engine.options.mcts_simulations = 4
    assert engine._simulation_budget(["movetime", "1"]) == 4


def test_unrestricted_search_executes_gumbel_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    from luna import uci

    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[move_to_action(chess.Move.from_uci("e2e4"))] = 0.9
    proposed = move_to_action(chess.Move.from_uci("d2d4"))
    _PolicyMCTS.policy = policy
    _PolicyMCTS.proposed_action = proposed
    monkeypatch.setattr(uci, "MCTS", _PolicyMCTS)
    engine = _engine()
    output: list[str] = []
    monkeypatch.setattr(engine, "send", output.append)

    engine.search(["nodes", "8"])

    assert output[-1] == "bestmove d2d4"
    _PolicyMCTS.proposed_action = None


def test_searchmoves_restricts_white_move(monkeypatch: pytest.MonkeyPatch) -> None:
    from luna import uci

    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[move_to_action(chess.Move.from_uci("e2e4"))] = 0.9
    policy[move_to_action(chess.Move.from_uci("d2d4"))] = 0.1
    _PolicyMCTS.policy = policy
    monkeypatch.setattr(uci, "MCTS", _PolicyMCTS)
    engine = _engine()
    output: list[str] = []
    monkeypatch.setattr(engine, "send", output.append)

    engine.search(["searchmoves", "d2d4", "wtime", "60000", "btime", "60000"])

    assert output[-1] == "bestmove d2d4"
    assert _PolicyMCTS.allowed_root_actions == {move_to_action(chess.Move.from_uci("d2d4"))}


def test_searchmoves_uses_canonical_action_for_black(monkeypatch: pytest.MonkeyPatch) -> None:
    from luna import uci

    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[move_to_action(mirror_move(chess.Move.from_uci("e7e5")))] = 0.9
    policy[move_to_action(mirror_move(chess.Move.from_uci("d7d5")))] = 0.1
    _PolicyMCTS.policy = policy
    monkeypatch.setattr(uci, "MCTS", _PolicyMCTS)
    engine = _engine()
    engine.set_position(["startpos", "moves", "e2e4"])
    output: list[str] = []
    monkeypatch.setattr(engine, "send", output.append)

    engine.search(["searchmoves", "d7d5", "wtime", "60000", "btime", "60000"])

    assert output[-1] == "bestmove d7d5"


def test_searchmoves_with_no_legal_candidate_returns_null_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine()
    output: list[str] = []
    monkeypatch.setattr(engine, "send", output.append)

    engine.search(["searchmoves", "a1a8"])

    assert output == ["info string searchmoves contains no legal moves", "bestmove 0000"]


def test_claimable_draw_still_returns_a_legal_move(monkeypatch: pytest.MonkeyPatch) -> None:
    from luna import uci

    engine = _engine()
    engine.board = chess.Board("8/8/8/8/8/8/4k3/R5K1 w - - 100 75")
    assert engine.board.is_game_over(claim_draw=True)
    assert not engine.board.is_game_over(claim_draw=False)

    move = chess.Move.from_uci("a1a2")
    assert move in engine.board.legal_moves
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[move_to_action(move)] = 1.0
    _PolicyMCTS.policy = policy
    _PolicyMCTS.proposed_action = move_to_action(move)
    monkeypatch.setattr(uci, "MCTS", _PolicyMCTS)
    output: list[str] = []
    monkeypatch.setattr(engine, "send", output.append)

    engine.search(["nodes", "8"])

    assert output[-1] == "bestmove a1a2"
    _PolicyMCTS.proposed_action = None


def test_uci_handshake_and_runtime_options(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine()
    output: list[str] = []
    monkeypatch.setattr(engine, "send", output.append)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            "uci\n"
            "setoption name MCTS Simulations value 42\n"
            "setoption name Minimum Simulations value 6\n"
            "setoption name Estimated Simulation ms value 2.5\n"
            "isready\n"
            "quit\n"
        ),
    )

    engine.run()

    assert output[0] == "id name Luna ChessRL"
    assert "uciok" in output
    assert output[-1] == "readyok"
    assert engine.options.mcts_simulations == 42
    assert engine.options.minimum_simulations == 6
    assert engine.options.estimated_simulation_ms == 2.5


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


def test_main_warms_inference_before_running_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert len(warmed_games) == 1
    assert len(protocol_runs) == 1
