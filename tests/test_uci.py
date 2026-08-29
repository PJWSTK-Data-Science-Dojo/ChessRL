from __future__ import annotations

import io
import time
from typing import Any

import chess
import numpy as np

from luna.game.chess_game import ACTION_SIZE, mirror_move, move_to_action
from luna.uci import LunaUciEngine, UciOptions


class _PolicyMCTS:
    policy: np.ndarray
    proposed_action: int | None = None

    def __init__(self, game: Any, network: Any, params: Any) -> None:
        del game, network, params
        self.last_action: int | None = None

    def search_latent(
        self,
        board: chess.Board,
        temp: float,
        *,
        add_exploration_noise: bool,
        should_stop: Any = None,
    ) -> tuple[np.ndarray, float]:
        del board, temp, should_stop
        assert add_exploration_noise is False
        self.last_action = self.proposed_action if self.proposed_action is not None else int(np.argmax(self.policy))
        self.last_simulations = 8
        return self.policy.copy(), 0.25


def _engine() -> LunaUciEngine:
    return LunaUciEngine(network=object(), options=UciOptions(mcts_simulations=64))  # type: ignore[arg-type]


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
        network=object(),  # type: ignore[arg-type]
        options=UciOptions(mcts_simulations=100, minimum_simulations=8, estimated_simulation_ms=5.0),
    )
    assert engine._simulation_budget(["movetime", "2025"]) == 100
    assert engine._simulation_budget(["movetime", "1"]) == 8
    assert engine._simulation_budget(["wtime", "3000", "winc", "0"]) == 20
    assert engine._simulation_budget(["nodes", "12"]) == 12

    engine.options.mcts_simulations = 4
    assert engine._simulation_budget(["movetime", "1"]) == 4


def test_unrestricted_search_executes_gumbel_proposal(monkeypatch: Any) -> None:
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


def test_searchmoves_restricts_white_move(monkeypatch: Any) -> None:
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


def test_searchmoves_uses_canonical_action_for_black(monkeypatch: Any) -> None:
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


def test_searchmoves_with_no_legal_candidate_returns_null_move(monkeypatch: Any) -> None:
    engine = _engine()
    output: list[str] = []
    monkeypatch.setattr(engine, "send", output.append)

    engine.search(["searchmoves", "a1a8"])

    assert output == ["info string searchmoves contains no legal moves", "bestmove 0000"]


def test_claimable_draw_still_returns_a_legal_move(monkeypatch: Any) -> None:
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


def test_uci_handshake_and_runtime_options(monkeypatch: Any) -> None:
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


def test_stop_interrupts_background_search_and_emits_one_move(monkeypatch: Any) -> None:
    from luna import uci

    action = move_to_action(chess.Move.from_uci("e2e4"))

    class _InterruptibleMCTS:
        def __init__(self, game: Any, network: Any, params: Any) -> None:
            del game, network, params
            self.last_action = action
            self.last_simulations = 0

        def search_latent(
            self,
            board: chess.Board,
            temp: float,
            *,
            add_exploration_noise: bool,
            should_stop: Any,
        ) -> tuple[np.ndarray, float]:
            del board, temp, add_exploration_noise
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
