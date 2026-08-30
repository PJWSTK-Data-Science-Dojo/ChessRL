"""Universal Chess Interface adapter for Luna.

The adapter keeps protocol output on stdout and diagnostics on stderr, making it
compatible with desktop chess GUIs and the community ``lichess-bot`` bridge.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import chess
import numpy as np
import torch
from loguru import logger

from luna.config import MCTSParams
from luna.game.chess_game import ChessGame, mirror_move, move_to_action, player_from_turn
from luna.mcts import MCTS
from luna.network import LunaNetwork

_TIMED_GO_KEYS = frozenset({"movetime", "wtime", "btime", "winc", "binc", "movestogo"})
_CLOCK_SAFETY_MS = 25


@dataclass
class UciOptions:
    """Runtime options that can be changed with ``setoption``."""

    mcts_simulations: int
    minimum_simulations: int = 8
    estimated_simulation_ms: float = 4.0


@dataclass(frozen=True)
class SearchLimits:
    """A simulation cap and optional hard wall-clock boundary."""

    simulations: int
    deadline: float | None


class LunaUciEngine:
    """UCI state machine with interruptible background search."""

    def __init__(self, network: LunaNetwork, options: UciOptions) -> None:
        self.game = ChessGame(claim_draw=False)
        self.network = network
        self.options = options
        self.board = chess.Board()
        self._search_thread: threading.Thread | None = None
        self._search_stop: threading.Event | None = None
        self._ready = False

    @staticmethod
    def send(message: str) -> None:
        print(message, flush=True)

    def set_position(self, tokens: list[str]) -> None:
        if not tokens:
            raise ValueError("position requires 'startpos' or 'fen'")
        if tokens[0] == "startpos":
            board = chess.Board()
            move_marker = 1
        elif tokens[0] == "fen":
            try:
                move_marker = tokens.index("moves")
            except ValueError:
                move_marker = len(tokens)
            fen = " ".join(tokens[1:move_marker])
            board = chess.Board(fen)
        else:
            raise ValueError(f"unsupported position command: {tokens[0]}")

        if move_marker < len(tokens) and tokens[move_marker] == "moves":
            for move_text in tokens[move_marker + 1 :]:
                move = chess.Move.from_uci(move_text)
                if move not in board.legal_moves:
                    raise ValueError(f"illegal position move: {move_text}")
                board.push(move)
        self.board = board

    @staticmethod
    def _go_values(go_tokens: list[str]) -> dict[str, int]:
        values: dict[str, int] = {}
        for index, token in enumerate(go_tokens):
            if token in _TIMED_GO_KEYS or token == "nodes":
                if index + 1 >= len(go_tokens):
                    raise ValueError(f"go {token} requires an integer value")
                try:
                    values[token] = int(go_tokens[index + 1])
                except ValueError as exc:
                    raise ValueError(f"go {token} requires an integer value") from exc
        return values

    def _allocated_time_ms(self, values: dict[str, int]) -> int | None:
        if "movetime" in values:
            return max(1, values["movetime"] - _CLOCK_SAFETY_MS)

        side = "w" if self.board.turn == chess.WHITE else "b"
        remaining = values.get(f"{side}time")
        if remaining is None:
            return None
        increment = max(0, values.get(f"{side}inc", 0))
        moves_to_go = max(1, values.get("movestogo", 30))
        normal_allocation = max(_CLOCK_SAFETY_MS, remaining // moves_to_go + int(increment * 0.75))
        clock_safe_limit = max(1, remaining - _CLOCK_SAFETY_MS)
        return min(normal_allocation, clock_safe_limit)

    def _search_limits(self, go_tokens: list[str]) -> SearchLimits:
        values = self._go_values(go_tokens)
        maximum = max(1, self.options.mcts_simulations)
        minimum = min(maximum, max(1, self.options.minimum_simulations))
        budget_ms = self._allocated_time_ms(values)
        deadline = None if budget_ms is None else time.monotonic() + budget_ms / 1_000
        if "nodes" in values:
            simulations = min(maximum, max(1, values["nodes"]))
            return SearchLimits(simulations=simulations, deadline=deadline)

        if budget_ms is None:
            return SearchLimits(simulations=maximum, deadline=None)
        estimated = max(self.options.estimated_simulation_ms, 0.1)
        affordable = int(budget_ms / estimated)
        simulations = min(maximum, max(minimum, affordable))
        return SearchLimits(simulations=simulations, deadline=deadline)

    def _simulation_budget(self, go_tokens: list[str]) -> int:
        return self._search_limits(go_tokens).simulations

    @staticmethod
    def _stop_callback(
        stop_event: threading.Event | None,
        deadline: float | None,
    ) -> Callable[[], bool] | None:
        if stop_event is None and deadline is None:
            return None

        def should_stop() -> bool:
            externally_stopped = stop_event is not None and stop_event.is_set()
            timed_out = deadline is not None and time.monotonic() >= deadline
            return externally_stopped or timed_out

        return should_stop

    def _searchmove_actions(self, go_tokens: list[str]) -> set[int] | None:
        """Return canonical action IDs requested by UCI ``searchmoves``.

        ``None`` means unrestricted search. An empty set means the caller supplied
        the option but none of its moves is legal in the current position.
        """
        keywords = {
            "searchmoves",
            "ponder",
            "wtime",
            "btime",
            "winc",
            "binc",
            "movestogo",
            "depth",
            "nodes",
            "mate",
            "movetime",
            "infinite",
        }
        try:
            start = go_tokens.index("searchmoves") + 1
        except ValueError:
            return None

        actions: set[int] = set()
        for token in go_tokens[start:]:
            if token.lower() in keywords:
                break
            try:
                move = chess.Move.from_uci(token)
            except ValueError:
                continue
            if move not in self.board.legal_moves:
                continue
            canonical_move = move if self.board.turn == chess.WHITE else mirror_move(move)
            actions.add(move_to_action(canonical_move))
        return actions

    def search(self, go_tokens: list[str], stop_event: threading.Event | None = None) -> None:
        if self.board.is_game_over(claim_draw=False):
            self.send("bestmove 0000")
            return
        infinite = any(token.lower() == "infinite" for token in go_tokens)
        if infinite and stop_event is None:
            raise ValueError("go infinite requires an interruptible search context")

        restricted_actions = self._searchmove_actions(go_tokens)
        if restricted_actions is not None and not restricted_actions:
            self.send("info string searchmoves contains no legal moves")
            self.send("bestmove 0000")
            return

        limits = self._search_limits(go_tokens)
        current_player = player_from_turn(self.board.turn)
        canonical = self.game.get_canonical_form(self.board, current_player)
        params = MCTSParams(
            num_mcts_sims=limits.simulations,
            dir_noise=False,
            recurrent_policy_topk=256,
        )
        started = time.monotonic()
        search = MCTS(self.game, self.network, params)
        _policy, value = search.search_latent(
            canonical,
            temp=1.0,
            add_exploration_noise=False,
            should_stop=self._stop_callback(stop_event, limits.deadline),
            allowed_root_actions=restricted_actions,
        )
        search_elapsed_ms = max(1, int((time.monotonic() - started) * 1_000))
        if infinite:
            if stop_event is None:
                raise RuntimeError("Infinite search lost its stop event")
            stop_event.wait()
        if search.last_action is None:
            raise RuntimeError("Search returned no legal continuation")
        action = search.last_action
        if restricted_actions is not None and action not in restricted_actions:
            raise RuntimeError("Search returned an action outside the requested root moves")
        next_board, _ = self.game.get_next_state(self.board, current_player, action)
        move = next_board.peek()
        elapsed_ms = max(1, int((time.monotonic() - started) * 1_000)) if infinite else search_elapsed_ms
        completed_simulations = int(getattr(search, "last_simulations", limits.simulations))
        if completed_simulations > 0:
            observed_simulation_ms = search_elapsed_ms / completed_simulations
            self.options.estimated_simulation_ms = max(
                0.1,
                0.8 * self.options.estimated_simulation_ms + 0.2 * observed_simulation_ms,
            )
        score_cp = int(np.clip(value, -1.0, 1.0) * 1_000)
        self.send(
            f"info depth 1 seldepth {completed_simulations} score cp {score_cp} "
            f"nodes {completed_simulations} time {elapsed_ms} pv {move.uci()}"
        )
        self.send(f"bestmove {move.uci()}")

    def _search_worker(self, go_tokens: list[str], stop_event: threading.Event) -> None:
        search_finished = False
        try:
            self.search(go_tokens, stop_event)
            search_finished = True
        except (AssertionError, RuntimeError, ValueError) as exc:
            logger.exception("UCI search failed")
            self.send(f"info string error {type(exc).__name__}: {exc}")
        finally:
            if not search_finished:
                self.send("bestmove 0000")

    def _stop_active_search(self) -> None:
        thread = self._search_thread
        if thread is None:
            return
        if self._search_stop is not None:
            self._search_stop.set()
        thread.join()
        self._search_thread = None
        self._search_stop = None

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        self.network.warmup_mcts_inference(self.game)
        self._ready = True

    def _start_search(self, go_tokens: list[str]) -> None:
        self._stop_active_search()
        self._ensure_ready()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._search_worker,
            args=(go_tokens, stop_event),
            name="luna-uci-search",
            daemon=True,
        )
        self._search_stop = stop_event
        self._search_thread = thread
        thread.start()

    def set_option(self, tokens: list[str]) -> None:
        lowered = [token.lower() for token in tokens]
        if "name" not in lowered or "value" not in lowered:
            return
        name_at = lowered.index("name") + 1
        value_at = lowered.index("value")
        name = " ".join(tokens[name_at:value_at]).strip().lower()
        value = " ".join(tokens[value_at + 1 :]).strip()
        if name == "mcts simulations":
            self.options.mcts_simulations = max(1, int(value))
            self.options.minimum_simulations = min(
                self.options.minimum_simulations,
                self.options.mcts_simulations,
            )
        elif name == "minimum simulations":
            self.options.minimum_simulations = min(
                self.options.mcts_simulations,
                max(1, int(value)),
            )
        elif name == "estimated simulation ms":
            self.options.estimated_simulation_ms = max(0.1, float(value))

    def run(self) -> None:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                tokens = shlex.split(line)
                command, arguments = tokens[0].lower(), tokens[1:]
                if command == "uci":
                    self.send("id name Luna ChessRL")
                    self.send("id author ChessRL contributors")
                    self.send(
                        f"option name MCTS Simulations type spin default {self.options.mcts_simulations} min 1 max 4096"
                    )
                    self.send(
                        f"option name Minimum Simulations type spin default {self.options.minimum_simulations} min 1 max 512"
                    )
                    self.send(
                        "option name Estimated Simulation ms type string "
                        f"default {self.options.estimated_simulation_ms:g}"
                    )
                    self.send("uciok")
                elif command == "isready":
                    self._ensure_ready()
                    self.send("readyok")
                elif command == "ucinewgame":
                    self._stop_active_search()
                    self.board.reset()
                elif command == "position":
                    self._stop_active_search()
                    self.set_position(arguments)
                elif command == "setoption":
                    self.set_option(arguments)
                elif command == "go":
                    self._start_search(arguments)
                elif command == "stop":
                    self._stop_active_search()
                elif command == "quit":
                    self._stop_active_search()
                    return
            except (RuntimeError, ValueError) as exc:
                logger.exception("UCI command failed: {}", line)
                self.send(f"info string error {type(exc).__name__}: {exc}")
                if line.split(maxsplit=1)[0].lower() == "go":
                    self.send("bestmove 0000")
        self._stop_active_search()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Luna checkpoint as a UCI chess engine.")
    parser.add_argument("--checkpoint", default="./runs/luna-main/latest.pth.tar")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--mcts-sims", type=int, default=100)
    parser.add_argument("--minimum-sims", type=int, default=8)
    parser.add_argument("--estimated-sim-ms", type=float, default=4.0)
    parser.add_argument("--compile-inference", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    torch.set_float32_matmul_precision("medium")
    game = ChessGame(claim_draw=False)
    try:
        network = LunaNetwork.from_checkpoint(
            game,
            args.checkpoint,
            device=args.device,
            cuda_device=args.cuda_device,
            compile_inference=args.compile_inference,
            load_optimizer=False,
        )
    except (KeyError, OSError, RuntimeError, ValueError):
        logger.exception("Could not load Luna checkpoint")
        return 2
    engine = LunaUciEngine(
        network,
        UciOptions(
            mcts_simulations=max(1, args.mcts_sims),
            minimum_simulations=max(1, args.minimum_sims),
            estimated_simulation_ms=max(0.1, args.estimated_sim_ms),
        ),
    )
    engine.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
