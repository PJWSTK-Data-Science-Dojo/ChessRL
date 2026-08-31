"""Benchmark Luna (MCTS + network) vs Stockfish with a fixed protocol."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal, cast

import chess
import wandb
from loguru import logger

from luna.config import MCTSParams, TrainingRunConfig, evaluation_mcts_params
from luna.game.arena import Arena
from luna.game.chess_game import ChessGame
from luna.game.player import StockfishPlayer
from luna.mcts import MCTS
from luna.network import LunaNetwork

_WIN = 0.5
_stockfish_module = importlib.import_module("stockfish")
_StockfishException = cast(type[Exception], _stockfish_module.StockfishException)

SkipReason = Literal["too_few_games", "too_many_games", "no_engine", "runtime_error"]

OPENING_SUITE_VERSION = 1
_OPENING_LINES = (
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"),
    ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"),
    ("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"),
    ("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"),
    ("c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6"),
    ("g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "g7g6"),
    ("e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"),
    ("e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4"),
    ("d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"),
    ("d2d4", "f7f5", "g2g3", "g8f6", "f1g2", "g7g6"),
)


@dataclass(frozen=True)
class StockfishEvalScores:
    """Finished Stockfish benchmark (alternating colors)."""

    model_wins: int
    draws: int
    stockfish_wins: int


@dataclass(frozen=True)
class StockfishEvalSkipped:
    """Benchmark did not run or aborted."""

    reason: SkipReason
    message: str = ""


StockfishEvalOutcome = StockfishEvalScores | StockfishEvalSkipped


@dataclass(frozen=True)
class StockfishEvaluationProtocol:
    """Settings that must stay fixed when comparing external scores."""

    opening_suite_version: int
    games: int
    stockfish_elo: int
    stockfish_depth: int
    stockfish_path: str | None
    max_ply: int | None
    mcts: MCTSParams


def _wandb_metrics(scores: StockfishEvalScores, iteration: int | None) -> dict[str, float | int]:
    total = scores.model_wins + scores.draws + scores.stockfish_wins
    win_rate = scores.model_wins / total if total else 0.0
    score = (scores.model_wins + 0.5 * scores.draws) / total if total else 0.0
    metrics: dict[str, float | int] = {
        "stockfish/luna_wins": scores.model_wins,
        "stockfish/draws": scores.draws,
        "stockfish/stockfish_wins": scores.stockfish_wins,
        "stockfish/win_rate": win_rate,
        "stockfish/score": score,
        "stockfish/opening_suite_version": OPENING_SUITE_VERSION,
    }
    if iteration is not None:
        metrics["iteration"] = iteration
    return metrics


def stockfish_evaluation_protocol(run: TrainingRunConfig) -> StockfishEvaluationProtocol:
    """Capture the comparable external-evaluation contract."""
    return StockfishEvaluationProtocol(
        opening_suite_version=OPENING_SUITE_VERSION,
        games=run.stockfish_eval_games,
        stockfish_elo=run.stockfish_elo,
        stockfish_depth=run.stockfish_depth,
        stockfish_path=run.stockfish_path,
        max_ply=run.stockfish_eval_max_ply,
        mcts=evaluation_mcts_params(run),
    )


def _stockfish_player(run: TrainingRunConfig) -> StockfishPlayer:
    return StockfishPlayer(
        elo=run.stockfish_elo,
        depth=run.stockfish_depth,
        path=run.stockfish_path,
    )


def _evaluation_openings(pair_count: int) -> list[chess.Board]:
    if pair_count > len(_OPENING_LINES):
        raise ValueError(f"stockfish_eval_games supports at most {2 * len(_OPENING_LINES)} games")
    openings: list[chess.Board] = []
    for line in _OPENING_LINES[:pair_count]:
        board = chess.Board()
        for move_text in line:
            board.push_uci(move_text)
        openings.append(board)
    return openings


def validate_stockfish_configuration(run: TrainingRunConfig) -> None:
    """Fail before a long run when its configured external benchmark cannot start."""
    try:
        _evaluation_openings(run.stockfish_eval_games // 2)
        player = _stockfish_player(run)
        player.close()
    except (OSError, ImportError, ValueError, RuntimeError, _StockfishException) as exc:
        raise RuntimeError(f"Stockfish benchmark preflight failed: {exc}") from exc


class ArenaMCTSPlayer:
    """Callable player: one MCTS instance, greedy policy (matches batched arena)."""

    def __init__(self, game: ChessGame, nnet: LunaNetwork, mcts_params: MCTSParams) -> None:
        self._mcts = MCTS(game, nnet, mcts_params)

    def __call__(self, canonical_board: chess.Board) -> int:
        self._mcts.search_latent(
            canonical_board,
            temp=0.0,
            add_exploration_noise=False,
        )
        if self._mcts.last_action is None:
            raise RuntimeError("Search returned no legal continuation")
        return self._mcts.last_action


def _score_game_for_model(arena_result: float, model_is_player1: bool) -> str:
    """Map ``Arena.play_game`` return value to model / draw / stockfish."""
    if arena_result > _WIN:
        return "model" if model_is_player1 else "sf"
    if arena_result < -_WIN:
        return "sf" if model_is_player1 else "model"
    return "draw"


def run_stockfish_eval(
    game: ChessGame,
    nnet: LunaNetwork,
    run: TrainingRunConfig,
    *,
    iteration: int | None = None,
) -> StockfishEvalOutcome:
    """Run a balanced fixed-settings benchmark or report an expected engine failure."""
    n_games = run.stockfish_eval_games
    if n_games < 1:
        logger.warning("stockfish_eval_games < 1; skipping Stockfish eval.")
        return StockfishEvalSkipped("too_few_games", "stockfish_eval_games < 1")
    if n_games % 2 == 1:
        n_games -= 1
        logger.warning("stockfish_eval_games was odd; using {} games for balanced colors.", n_games)
    if n_games < 2:
        logger.warning("stockfish_eval_games < 2 after rounding; skipping Stockfish eval.")
        return StockfishEvalSkipped(
            "too_few_games",
            "need at least 2 games after rounding to an even count",
        )
    try:
        openings = _evaluation_openings(n_games // 2)
    except ValueError as exc:
        logger.warning("Stockfish eval skipped (opening suite): {}", exc)
        return StockfishEvalSkipped("too_many_games", str(exc))

    mcts_params = evaluation_mcts_params(run)
    model_player = ArenaMCTSPlayer(game, nnet, mcts_params)
    max_ply = run.stockfish_eval_max_ply

    try:
        sf = _stockfish_player(run)
    except (OSError, ImportError, ValueError, RuntimeError, _StockfishException) as exc:
        logger.warning("Stockfish eval skipped (engine): {}", exc)
        return StockfishEvalSkipped("no_engine", str(exc))

    mw = dr = sw = 0
    try:
        for opening in openings:
            for model_is_p1 in (True, False):
                sf.new_game()
                if model_is_p1:
                    arena = Arena(model_player, sf.play, game)
                else:
                    arena = Arena(sf.play, model_player, game)
                r = arena.play_game(verbose=False, max_ply=max_ply, initial_board=opening)
                out = _score_game_for_model(r, model_is_p1)
                if out == "model":
                    mw += 1
                elif out == "sf":
                    sw += 1
                else:
                    dr += 1
    except (OSError, ValueError, RuntimeError, _StockfishException) as exc:
        logger.exception("Stockfish eval aborted during games")
        return StockfishEvalSkipped("runtime_error", str(exc))
    finally:
        sf.close()

    scores = StockfishEvalScores(model_wins=mw, draws=dr, stockfish_wins=sw)
    iter_suffix = f" (iter {iteration})" if iteration is not None else ""
    logger.info(
        "Stockfish eval{}: Luna {} — {} — {} Stockfish | MCTS sims={} SF elo={} depth={} games={} openings=v{}",
        iter_suffix,
        mw,
        dr,
        sw,
        mcts_params.num_mcts_sims,
        run.stockfish_elo,
        run.stockfish_depth,
        n_games,
        OPENING_SUITE_VERSION,
    )
    if wandb.run is not None:
        wandb.log(_wandb_metrics(scores, iteration))

    return scores
