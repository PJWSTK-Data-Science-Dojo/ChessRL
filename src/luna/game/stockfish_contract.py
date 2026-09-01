"""Comparable external-engine evaluation contracts and metrics."""

from __future__ import annotations

import hashlib
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from luna.config import MCTSParams, TrainingRunConfig, evaluation_mcts_params

SkipReason = Literal["too_few_games", "too_many_games", "no_engine", "runtime_error"]

OPENING_SUITE_VERSION = 1


@dataclass(frozen=True)
class StockfishEvalScores:
    """Finished Stockfish benchmark with alternating colors."""

    model_wins: int
    draws: int
    stockfish_wins: int


@dataclass(frozen=True)
class StockfishEvalSkipped:
    """Benchmark that did not run or aborted."""

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
