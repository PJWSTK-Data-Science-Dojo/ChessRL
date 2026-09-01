from __future__ import annotations

import io
from collections.abc import Callable, Collection

import chess
import numpy as np
import pytest

from luna.config import MCTSParams
from luna.game.chess_game import ACTION_SIZE, ChessGame, mirror_move, move_to_action
from luna.network import LunaNetwork
from luna.uci import MAX_MCTS_SIMULATIONS, MAX_MINIMUM_SIMULATIONS, LunaUciEngine, UciOptions


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
    assert engine._simulation_budget(["wtime", "3000", "winc", "0", "movestogo", "10"]) == 60
    assert engine._simulation_budget(["nodes", "12"]) == 12
    assert engine._simulation_budget(["nodes", "1"]) == 1
    assert engine._search_limits(["nodes", "12", "movetime", "50"]).deadline is not None
    with pytest.raises(ValueError, match="requires an integer"):
        engine._simulation_budget(["movetime", "invalid"])
    with pytest.raises(ValueError, match="requires an integer"):
        engine._simulation_budget(["movetime"])
    with pytest.raises(ValueError, match="64-bit integer range"):
        engine._simulation_budget(["movetime", str(1 << 63)])

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


def test_uci_options_enforce_advertised_bounds_and_finite_timing() -> None:
    options = UciOptions(
        mcts_simulations=MAX_MCTS_SIMULATIONS + 1,
        minimum_simulations=MAX_MINIMUM_SIMULATIONS + 1,
    )
    engine = LunaUciEngine(network=_network_stub(), options=options)

    assert options.mcts_simulations == MAX_MCTS_SIMULATIONS
    assert options.minimum_simulations == MAX_MINIMUM_SIMULATIONS

    engine.set_option(["name", "MCTS", "Simulations", "value", str(MAX_MCTS_SIMULATIONS + 100)])
    engine.set_option(["name", "Minimum", "Simulations", "value", str(MAX_MINIMUM_SIMULATIONS + 100)])
    assert options.mcts_simulations == MAX_MCTS_SIMULATIONS
    assert options.minimum_simulations == MAX_MINIMUM_SIMULATIONS

    with pytest.raises(ValueError, match="finite and positive"):
        engine.set_option(["name", "Estimated", "Simulation", "ms", "value", "inf"])
    with pytest.raises(ValueError, match="finite and positive"):
        UciOptions(mcts_simulations=8, estimated_simulation_ms=float("nan"))
