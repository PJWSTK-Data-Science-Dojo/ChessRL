"""Benchmark Luna against fixed and adaptive UCI-engine opponents."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

import chess
import chess.engine
import wandb
from loguru import logger

from luna.config import MCTSParams, TrainingRunConfig, evaluation_mcts_params
from luna.game.arena import Arena
from luna.game.checkpoint_arena import ArenaMCTSPlayer
from luna.game.chess_game import ChessGame
from luna.game.opening_suite import OPENING_SUITE_VERSION, evaluation_openings
from luna.game.player import StockfishPlayer
from luna.game.stockfish_contract import (
    EngineMatchSettings,
    StockfishEvalOutcome,
    StockfishEvalScores,
    StockfishEvalSkipped,
    StockfishEvaluationProtocol,
    _wandb_metrics,
    engine_binary_sha256,
    fixed_match_settings,
    ladder_match_settings,
    resolve_engine_path,
    stockfish_evaluation_protocol,
)
from luna.network import LunaNetwork

__all__ = [
    "OPENING_SUITE_VERSION",
    "ArenaMCTSPlayer",
    "EngineMatchSettings",
    "StockfishEvalOutcome",
    "StockfishEvalScores",
    "StockfishEvalSkipped",
    "StockfishEvaluationProtocol",
    "_StockfishException",
    "_score_game_for_model",
    "_wandb_metrics",
    "engine_binary_sha256",
    "fixed_match_settings",
    "ladder_match_settings",
    "resolve_engine_path",
    "retry_stockfish_eval",
    "run_stockfish_eval",
    "stockfish_evaluation_protocol",
    "validate_ladder_configuration",
    "validate_stockfish_configuration",
]

_WIN = 0.5
_StockfishException = cast(type[Exception], chess.engine.EngineError)


def retry_stockfish_eval(
    evaluate: Callable[[], StockfishEvalOutcome],
    *,
    attempts: int,
    retry_seconds: float,
) -> StockfishEvalOutcome:
    """Retry transient UCI failures without weakening fail-closed evaluation."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if not math.isfinite(retry_seconds) or retry_seconds < 0.0:
        raise ValueError("retry_seconds must be finite and non-negative")
    outcome = evaluate()
    for attempt in range(2, attempts + 1):
        if not isinstance(outcome, StockfishEvalSkipped) or outcome.reason not in {
            "no_engine",
            "runtime_error",
        }:
            return outcome
        logger.warning(
            "External-engine evaluation attempt {}/{} failed ({}): {}; retrying in {}s",
            attempt - 1,
            attempts,
            outcome.reason,
            outcome.message,
            retry_seconds,
        )
        if retry_seconds > 0.0:
            time.sleep(retry_seconds)
        outcome = evaluate()
    return outcome


def _stockfish_player(settings: EngineMatchSettings) -> StockfishPlayer:
    return StockfishPlayer(
        elo=settings.elo,
        depth=settings.depth,
        path=settings.path,
    )


def _validate_engine_settings(settings: EngineMatchSettings, *, require_fairy: bool) -> tuple[str, tuple[int, int]]:
    evaluation_openings(settings.games // 2)
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
    except (chess.engine.EngineError, OSError, TimeoutError, RuntimeError, ValueError) as exc:
        try:
            player.close()
        except (chess.engine.EngineError, OSError, TimeoutError, RuntimeError) as cleanup_exc:
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


def _score_game_for_model(arena_result: float, model_is_player1: bool) -> str:
    """Map ``Arena.play_game`` return value to model / draw / stockfish."""
    if arena_result > _WIN:
        return "model" if model_is_player1 else "sf"
    if arena_result < -_WIN:
        return "sf" if model_is_player1 else "model"
    return "draw"


@dataclass(slots=True)
class _ScoreTally:
    model_wins: int = 0
    draws: int = 0
    stockfish_wins: int = 0

    def record(self, outcome: str) -> None:
        if outcome == "model":
            self.model_wins += 1
        elif outcome == "sf":
            self.stockfish_wins += 1
        else:
            self.draws += 1

    def scores(self) -> StockfishEvalScores:
        return StockfishEvalScores(self.model_wins, self.draws, self.stockfish_wins)


@dataclass(frozen=True, slots=True)
class _MatchContext:
    game: ChessGame
    model_player: Callable[[chess.Board], int]
    stockfish: StockfishPlayer
    max_ply: int | None


@dataclass(frozen=True, slots=True)
class _EvaluationReport:
    match: EngineMatchSettings
    scores: StockfishEvalScores
    mcts_params: MCTSParams
    game_count: int
    iteration: int | None
    metric_prefix: str | None
    started_at: float


def _balanced_game_count(requested_games: int) -> int | StockfishEvalSkipped:
    if requested_games < 1:
        logger.warning("External-engine games < 1; skipping evaluation.")
        return StockfishEvalSkipped("too_few_games", "evaluation games < 1")
    game_count = requested_games
    if game_count % 2 == 1:
        game_count -= 1
        logger.warning("External-engine game count was odd; using {} games for balanced colors.", game_count)
    if game_count < 2:
        logger.warning("stockfish_eval_games < 2 after rounding; skipping Stockfish eval.")
        return StockfishEvalSkipped(
            "too_few_games",
            "need at least 2 games after rounding to an even count",
        )
    return game_count


def _openings_for_match(game_count: int) -> list[chess.Board] | StockfishEvalSkipped:
    try:
        return list(evaluation_openings(game_count // 2))
    except ValueError as exc:
        logger.warning("Stockfish eval skipped (opening suite): {}", exc)
        return StockfishEvalSkipped("too_many_games", str(exc))


def _open_stockfish(match: EngineMatchSettings) -> StockfishPlayer | StockfishEvalSkipped:
    try:
        return _stockfish_player(match)
    except (OSError, ImportError, ValueError, RuntimeError, _StockfishException) as exc:
        logger.warning("Stockfish eval skipped (engine): {}", exc)
        return StockfishEvalSkipped("no_engine", str(exc))


def _play_match(
    context: _MatchContext,
    openings: list[chess.Board],
) -> StockfishEvalOutcome:
    tally = _ScoreTally()
    runtime_failure: StockfishEvalSkipped | None = None
    close_failure: str | None = None
    try:
        for opening in openings:
            _play_opening_pair(context, opening, tally)
    except (OSError, ValueError, RuntimeError, _StockfishException) as exc:
        logger.exception("Stockfish eval aborted during games")
        runtime_failure = StockfishEvalSkipped("runtime_error", str(exc))
    finally:
        try:
            context.stockfish.close()
        except (OSError, ValueError, RuntimeError, _StockfishException) as exc:
            close_failure = str(exc)
    return _resolve_match_outcome(tally, runtime_failure, close_failure)


def _play_opening_pair(
    context: _MatchContext,
    opening: chess.Board,
    tally: _ScoreTally,
) -> None:
    for model_is_player1 in (True, False):
        context.stockfish.new_game()
        if model_is_player1:
            arena = Arena(context.model_player, context.stockfish.play, context.game)
        else:
            arena = Arena(context.stockfish.play, context.model_player, context.game)
        result = arena.play_game(verbose=False, max_ply=context.max_ply, initial_board=opening)
        tally.record(_score_game_for_model(result, model_is_player1))


def _resolve_match_outcome(
    tally: _ScoreTally,
    runtime_failure: StockfishEvalSkipped | None,
    close_failure: str | None,
) -> StockfishEvalOutcome:
    if runtime_failure is not None:
        if close_failure is not None:
            logger.warning("Stockfish cleanup also failed after the benchmark error: {}", close_failure)
        return runtime_failure
    if close_failure is not None:
        logger.error("Stockfish eval cleanup failed: {}", close_failure)
        return StockfishEvalSkipped("runtime_error", f"failed to close Stockfish: {close_failure}")
    return tally.scores()


def _report_completed_evaluation(report: _EvaluationReport) -> None:
    iteration_suffix = f" (iter {report.iteration})" if report.iteration is not None else ""
    logger.info(
        "{} eval{}: Luna {} — {} — {} {} | MCTS sims={} opponent elo={} depth={} games={} openings=v{}",
        report.match.opponent_name,
        iteration_suffix,
        report.scores.model_wins,
        report.scores.draws,
        report.scores.stockfish_wins,
        report.match.opponent_name,
        report.mcts_params.num_mcts_sims,
        report.match.elo,
        report.match.depth,
        report.game_count,
        OPENING_SUITE_VERSION,
    )
    if wandb.run is not None and report.metric_prefix is not None:
        wandb.log(
            _wandb_metrics(
                report.scores,
                report.iteration,
                prefix=report.metric_prefix,
                opponent_elo=report.match.elo,
                duration_seconds=time.perf_counter() - report.started_at,
            )
        )


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
    game_count = _balanced_game_count(match.games)
    if isinstance(game_count, StockfishEvalSkipped):
        return game_count
    openings = _openings_for_match(game_count)
    if isinstance(openings, StockfishEvalSkipped):
        return openings
    mcts_params = evaluation_mcts_params(run)
    model_player = ArenaMCTSPlayer(game, nnet, mcts_params)
    stockfish = _open_stockfish(match)
    if isinstance(stockfish, StockfishEvalSkipped):
        return stockfish
    outcome = _play_match(_MatchContext(game, model_player, stockfish, match.max_ply), openings)
    if isinstance(outcome, StockfishEvalSkipped):
        return outcome
    _report_completed_evaluation(
        _EvaluationReport(
            match,
            outcome,
            mcts_params,
            game_count,
            iteration,
            metric_prefix,
            started_at,
        )
    )
    return outcome
