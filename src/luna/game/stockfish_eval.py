"""Benchmark Luna against fixed and adaptive UCI-engine opponents."""

from __future__ import annotations

import hashlib
import math
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

import chess
import chess.engine
import wandb
from loguru import logger

from luna.config import MCTSParams, TrainingRunConfig, evaluation_mcts_params
from luna.game.arena import Arena
from luna.game.chess_game import ChessGame
from luna.game.player import StockfishPlayer
from luna.mcts import MCTS
from luna.network import LunaNetwork

_WIN = 0.5
_StockfishException = cast(type[Exception], chess.engine.EngineError)

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
class EngineMatchSettings:
    """Complete external-engine contract for one balanced match."""

    opponent_name: str
    games: int
    elo: int
    depth: int
    path: str | None
    max_ply: int | None


@dataclass(frozen=True)
class StockfishEvaluationProtocol:
    """Settings that must stay fixed when comparing external scores."""

    opening_suite_version: int
    games: int
    stockfish_elo: int
    stockfish_depth: int
    stockfish_path: str | None
    stockfish_binary_sha256: str
    max_ply: int | None
    mcts: MCTSParams


def _score_interval(scores: StockfishEvalScores) -> tuple[float, float]:
    """Approximate a 95% Wilson interval with draws treated as half a point."""
    total = scores.model_wins + scores.draws + scores.stockfish_wins
    if total <= 0:
        return 0.0, 1.0
    score = (scores.model_wins + 0.5 * scores.draws) / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (score + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((score * (1.0 - score) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _wandb_metrics(
    scores: StockfishEvalScores,
    iteration: int | None,
    *,
    prefix: str = "benchmark",
    opponent_elo: int | None = None,
    duration_seconds: float | None = None,
) -> dict[str, float | int]:
    total = scores.model_wins + scores.draws + scores.stockfish_wins
    win_rate = scores.model_wins / total if total else 0.0
    score = (scores.model_wins + 0.5 * scores.draws) / total if total else 0.0
    decisive_games = scores.model_wins + scores.stockfish_wins
    decisive_win_rate = scores.model_wins / decisive_games if decisive_games else 0.0
    ci_low, ci_high = _score_interval(scores)
    metrics: dict[str, float | int] = {
        f"{prefix}/luna_wins": scores.model_wins,
        f"{prefix}/draws": scores.draws,
        f"{prefix}/stockfish_wins": scores.stockfish_wins,
        f"{prefix}/games": total,
        f"{prefix}/win_rate": win_rate,
        f"{prefix}/decisive_win_rate": decisive_win_rate,
        f"{prefix}/score": score,
        f"{prefix}/score_approx_ci95_low": ci_low,
        f"{prefix}/score_approx_ci95_high": ci_high,
        f"{prefix}/opening_suite_version": OPENING_SUITE_VERSION,
    }
    if opponent_elo is not None:
        metrics[f"{prefix}/opponent_elo"] = opponent_elo
    if duration_seconds is not None:
        metrics[f"{prefix}/duration_seconds"] = duration_seconds
    if iteration is not None:
        metrics["iteration"] = iteration
    return metrics


def resolve_engine_path(path: str | None) -> Path:
    """Resolve an explicit or PATH-discovered engine executable."""
    command = path if path is not None else "stockfish"
    discovered = shutil.which(command)
    resolved = Path(discovered if discovered is not None else command).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"External engine executable not found: {resolved}")
    return resolved


def engine_binary_sha256(path: str | None) -> str:
    """Hash the exact external-engine binary used by an evaluation contract."""
    digest = hashlib.sha256()
    with resolve_engine_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_match_settings(run: TrainingRunConfig) -> EngineMatchSettings:
    """Build the immutable official-Stockfish benchmark settings."""
    return EngineMatchSettings(
        opponent_name="Stockfish",
        games=run.stockfish_eval_games,
        elo=run.stockfish_elo,
        depth=run.stockfish_depth,
        path=run.stockfish_path,
        max_ply=run.stockfish_eval_max_ply,
    )


def ladder_match_settings(run: TrainingRunConfig, elo: int) -> EngineMatchSettings:
    """Build one Fairy-Stockfish ladder-rung contract."""
    return EngineMatchSettings(
        opponent_name="Fairy-Stockfish",
        games=run.ladder_eval_games,
        elo=elo,
        depth=run.ladder_depth,
        path=run.ladder_path,
        max_ply=run.ladder_eval_max_ply,
    )


def stockfish_evaluation_protocol(run: TrainingRunConfig) -> StockfishEvaluationProtocol:
    """Capture the comparable external-evaluation contract."""
    return StockfishEvaluationProtocol(
        opening_suite_version=OPENING_SUITE_VERSION,
        games=run.stockfish_eval_games,
        stockfish_elo=run.stockfish_elo,
        stockfish_depth=run.stockfish_depth,
        stockfish_path=run.stockfish_path,
        stockfish_binary_sha256=engine_binary_sha256(run.stockfish_path),
        max_ply=run.stockfish_eval_max_ply,
        mcts=evaluation_mcts_params(run),
    )


def _stockfish_player(settings: EngineMatchSettings) -> StockfishPlayer:
    return StockfishPlayer(
        elo=settings.elo,
        depth=settings.depth,
        path=settings.path,
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


def _validate_engine_settings(settings: EngineMatchSettings, *, require_fairy: bool) -> tuple[str, tuple[int, int]]:
    _evaluation_openings(settings.games // 2)
    player = _stockfish_player(replace(settings, depth=1))
    try:
        engine_name = player.engine_name
        normalized_name = engine_name.strip().casefold()
        is_fairy = normalized_name == "fairy-stockfish" or normalized_name.startswith("fairy-stockfish ")
        is_official = normalized_name == "stockfish" or normalized_name.startswith("stockfish ")
        if require_fairy and not is_fairy:
            raise ValueError(f"Adaptive ladder requires Fairy-Stockfish, but the binary reports {engine_name!r}")
        if not require_fairy and not is_official:
            raise ValueError(f"Fixed benchmark requires official Stockfish, but the binary reports {engine_name!r}")
        elo_range = player.elo_range
        player.play(chess.Board())
    except Exception as exc:
        try:
            player.close()
        except Exception as cleanup_exc:
            exc.add_note(f"External-engine preflight cleanup also failed: {cleanup_exc}")
        raise
    player.close()
    return engine_name, elo_range


def validate_stockfish_configuration(run: TrainingRunConfig) -> None:
    """Fail before a long run when its fixed benchmark cannot start."""
    try:
        _validate_engine_settings(fixed_match_settings(run), require_fairy=False)
    except (OSError, ImportError, ValueError, RuntimeError, _StockfishException) as exc:
        raise RuntimeError(f"Stockfish benchmark preflight failed: {exc}") from exc


def validate_ladder_configuration(run: TrainingRunConfig) -> None:
    """Verify the pinned Fairy binary and its complete configured Elo range."""
    try:
        name, elo_range = _validate_engine_settings(
            ladder_match_settings(run, run.ladder_start_elo),
            require_fairy=True,
        )
        expected_range = (500, 2850)
        if elo_range != expected_range:
            raise ValueError(f"Fairy-Stockfish {name!r} advertises UCI_Elo {elo_range}, expected {expected_range}")
        if not elo_range[0] <= run.ladder_max_elo <= elo_range[1]:
            raise ValueError(f"ladder_max_elo {run.ladder_max_elo} is outside engine range {elo_range}")
    except (OSError, ImportError, ValueError, RuntimeError, _StockfishException) as exc:
        raise RuntimeError(f"Fairy-Stockfish ladder preflight failed: {exc}") from exc


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
    settings: EngineMatchSettings | None = None,
    metric_prefix: str | None = "benchmark",
) -> StockfishEvalOutcome:
    """Run a balanced UCI-engine match or report an expected engine failure."""
    match = settings if settings is not None else fixed_match_settings(run)
    started_at = time.perf_counter()
    n_games = match.games
    if n_games < 1:
        logger.warning("External-engine games < 1; skipping evaluation.")
        return StockfishEvalSkipped("too_few_games", "evaluation games < 1")
    if n_games % 2 == 1:
        n_games -= 1
        logger.warning("External-engine game count was odd; using {} games for balanced colors.", n_games)
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
    max_ply = match.max_ply

    try:
        sf = _stockfish_player(match)
    except (OSError, ImportError, ValueError, RuntimeError, _StockfishException) as exc:
        logger.warning("Stockfish eval skipped (engine): {}", exc)
        return StockfishEvalSkipped("no_engine", str(exc))

    mw = dr = sw = 0
    runtime_failure: StockfishEvalSkipped | None = None
    close_failure: str | None = None
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
        runtime_failure = StockfishEvalSkipped("runtime_error", str(exc))
    finally:
        try:
            sf.close()
        except (OSError, ValueError, RuntimeError, _StockfishException) as exc:
            close_failure = str(exc)

    if runtime_failure is not None:
        if close_failure is not None:
            logger.warning("Stockfish cleanup also failed after the benchmark error: {}", close_failure)
        return runtime_failure
    if close_failure is not None:
        logger.error("Stockfish eval cleanup failed: {}", close_failure)
        return StockfishEvalSkipped("runtime_error", f"failed to close Stockfish: {close_failure}")

    scores = StockfishEvalScores(model_wins=mw, draws=dr, stockfish_wins=sw)
    iter_suffix = f" (iter {iteration})" if iteration is not None else ""
    logger.info(
        "{} eval{}: Luna {} — {} — {} {} | MCTS sims={} opponent elo={} depth={} games={} openings=v{}",
        match.opponent_name,
        iter_suffix,
        mw,
        dr,
        sw,
        match.opponent_name,
        mcts_params.num_mcts_sims,
        match.elo,
        match.depth,
        n_games,
        OPENING_SUITE_VERSION,
    )
    if wandb.run is not None and metric_prefix is not None:
        wandb.log(
            _wandb_metrics(
                scores,
                iteration,
                prefix=metric_prefix,
                opponent_elo=match.elo,
                duration_seconds=time.perf_counter() - started_at,
            )
        )

    return scores
