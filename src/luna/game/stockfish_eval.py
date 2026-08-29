"""Benchmark Luna (MCTS + network) vs Stockfish with a fixed protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loguru import logger

from ..config import MCTSParams, TrainingRunConfig, evaluation_mcts_params
from ..mcts import MCTS
from ..network import LunaNetwork
from .arena import Arena
from .chess_game import ChessGame
from .player import StockfishPlayer

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[misc, assignment]

_WIN = 0.5

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


class ArenaMCTSPlayer:
    """Callable player: one MCTS instance, greedy policy (matches batched arena)."""

    def __init__(self, game: ChessGame, nnet: LunaNetwork, mcts_params: MCTSParams) -> None:
        self._mcts = MCTS(game, nnet, mcts_params)

    def __call__(self, canonical_board) -> int:
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
    """Play ``run.stockfish_eval_games`` games vs Stockfish (alternating colors).

    Uses :func:`~luna.config.evaluation_mcts_params` for MCTS (no exploration noise).
    Returns :class:`StockfishEvalScores` on success, or :class:`StockfishEvalSkipped` if the
    run could not complete (too few games, engine missing, or error during games).
    """
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
        sf = StockfishPlayer(
            elo=run.stockfish_elo,
            skill_level=run.stockfish_skill_level,
            depth=run.stockfish_depth,
            think_time=run.stockfish_think_time,
            path=run.stockfish_path,
        )
    except (OSError, ImportError, ValueError, RuntimeError) as exc:
        logger.warning("Stockfish eval skipped (engine): {}", exc)
        return StockfishEvalSkipped("no_engine", str(exc))
    except Exception as exc:
        logger.exception("Stockfish eval skipped (unexpected error starting engine)")
        return StockfishEvalSkipped("no_engine", f"{type(exc).__name__}: {exc}")

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
    except Exception as exc:
        logger.exception("Stockfish eval aborted during games")
        return StockfishEvalSkipped("runtime_error", str(exc))

    total = mw + dr + sw
    wr = mw / total if total else 0.0
    iter_suffix = f" (iter {iteration})" if iteration is not None else ""
    logger.info(
        "Stockfish eval{}: model {} — {} — {} SF | MCTS sims={} SF elo={} depth={} think_ms={} games={}",
        iter_suffix,
        mw,
        dr,
        sw,
        mcts_params.num_mcts_sims,
        run.stockfish_elo,
        run.stockfish_depth,
        run.stockfish_think_time,
        n_games,
    )
    if wandb is not None and wandb.run is not None:
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
