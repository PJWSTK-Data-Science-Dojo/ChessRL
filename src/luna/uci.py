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
from dataclasses import dataclass

import chess
import numpy as np
import torch
from loguru import logger

from .config import MCTSParams
from .game.chess_game import ChessGame, mirror_move, move_to_action, player_from_turn
from .mcts import MCTS
from .network import LunaNetwork


@dataclass
class UciOptions:
    """Runtime options that can be changed with ``setoption``."""

    mcts_simulations: int
    minimum_simulations: int = 8
    estimated_simulation_ms: float = 4.0


class LunaUciEngine:
    """UCI state machine with interruptible background search."""

    def __init__(self, network: LunaNetwork, options: UciOptions) -> None:
        self.game = ChessGame(claim_draw=False)
        self.network = network
        self.options = options
        self.board = chess.Board()
        self._search_thread: threading.Thread | None = None
        self._search_stop: threading.Event | None = None

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

    def _simulation_budget(self, go_tokens: list[str]) -> int:
        values: dict[str, int] = {}
        for index in range(0, len(go_tokens) - 1):
            if go_tokens[index] in {"movetime", "wtime", "btime", "winc", "binc", "nodes"}:
                try:
                    values[go_tokens[index]] = int(go_tokens[index + 1])
                except ValueError:
                    continue

        maximum = max(1, self.options.mcts_simulations)
        minimum = min(maximum, max(1, self.options.minimum_simulations))
        if "nodes" in values:
            return min(maximum, max(minimum, values["nodes"]))

        if "movetime" in values:
            budget_ms = max(1, values["movetime"] - 25)
        else:
            side = "w" if self.board.turn == chess.WHITE else "b"
            remaining = values.get(f"{side}time")
            increment = values.get(f"{side}inc", 0)
            if remaining is None:
                return self.options.mcts_simulations
            # Reserve most of the clock and spend increment aggressively. The UCI
            # bridge remains the final authority and can enforce a hard process limit.
            budget_ms = max(25, remaining // 30 + int(increment * 0.75))

        estimated = max(self.options.estimated_simulation_ms, 0.1)
        affordable = int(budget_ms / estimated)
        return min(maximum, max(minimum, affordable))

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

        restricted_actions = self._searchmove_actions(go_tokens)
        if restricted_actions is not None and not restricted_actions:
            self.send("info string searchmoves contains no legal moves")
            self.send("bestmove 0000")
            return

        simulations = self._simulation_budget(go_tokens)
        current_player = player_from_turn(self.board.turn)
        canonical = self.game.get_canonical_form(self.board, current_player)
        params = MCTSParams(
            num_mcts_sims=simulations,
            dir_noise=False,
            recurrent_policy_topk=256,
        )
        started = time.perf_counter()
        search = MCTS(self.game, self.network, params)
        policy, value = search.search_latent(
            canonical,
            temp=1.0,
            add_exploration_noise=False,
            should_stop=stop_event.is_set if stop_event is not None else None,
        )
        if restricted_actions is None:
            if search.last_action is None:
                raise RuntimeError("Search returned no legal continuation")
            action = search.last_action
        else:
            action = max(restricted_actions, key=lambda candidate: float(policy[candidate]))
        next_board, _ = self.game.get_next_state(self.board, current_player, action)
        move = next_board.peek()
        elapsed_ms = max(1, int((time.perf_counter() - started) * 1_000))
        completed_simulations = int(getattr(search, "last_simulations", simulations))
        if completed_simulations > 0:
            observed_simulation_ms = elapsed_ms / completed_simulations
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
        try:
            self.search(go_tokens, stop_event)
        except Exception as exc:
            logger.exception("UCI search failed")
            self.send(f"info string error {type(exc).__name__}: {exc}")
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

    def _start_search(self, go_tokens: list[str]) -> None:
        self._stop_active_search()
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
            tokens = shlex.split(line)
            command, arguments = tokens[0].lower(), tokens[1:]
            try:
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
            except Exception as exc:
                logger.exception("UCI command failed: {}", line)
                self.send(f"info string error {type(exc).__name__}: {exc}")
                if command == "go":
                    self.send("bestmove 0000")
        self._stop_active_search()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Luna checkpoint as a UCI chess engine.")
    parser.add_argument("--checkpoint", default="./temp/latest.pth.tar")
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
    except Exception:
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
