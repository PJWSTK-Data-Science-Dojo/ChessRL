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

SkipReason = Literal["too_few_games", "no_engine", "runtime_error"]


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


def _stockfish_player(run: TrainingRunConfig) -> StockfishPlayer:
    return StockfishPlayer(
        elo=run.stockfish_elo,
        depth=run.stockfish_depth,
        path=run.stockfish_path,
    )


def validate_stockfish_configuration(run: TrainingRunConfig) -> None:
    """Fail before a long run when its configured external benchmark cannot start."""
    try:
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
        for g in range(n_games):
            model_is_p1 = g % 2 == 0
            if model_is_p1:
                arena = Arena(model_player, sf.play, game)
            else:
                arena = Arena(sf.play, model_player, game)
            r = arena.play_game(verbose=False, max_ply=max_ply)
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

    total = mw + dr + sw
    wr = mw / total if total else 0.0
    iter_suffix = f" (iter {iteration})" if iteration is not None else ""
    logger.info(
        "Stockfish eval{}: model {} — {} — {} SF | MCTS sims={} SF elo={} depth={} games={}",
        iter_suffix,
        mw,
        dr,
        sw,
        mcts_params.num_mcts_sims,
        run.stockfish_elo,
        run.stockfish_depth,
        n_games,
    )
    if wandb.run is not None:
        log_payload = {
            "stockfish/model_wins": mw,
            "stockfish/draws": dr,
            "stockfish/opponent_wins": sw,
            "stockfish/win_rate": wr,
        }
        if iteration is not None:
            log_payload["iteration"] = iteration
        wandb.log(log_payload)

    return StockfishEvalScores(model_wins=mw, draws=dr, stockfish_wins=sw)
