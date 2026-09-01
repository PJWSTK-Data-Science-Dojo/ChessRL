"""Scheduled fixed benchmark and adaptive ladder orchestration."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import wandb
from loguru import logger

from luna.game.benchmark_state import (
    BENCHMARK_STATE_NAME,
    BenchmarkState,
    load_benchmark_state,
    record_benchmark_result,
    write_benchmark_state,
)
from luna.game.stockfish_eval import (
    StockfishEvalSkipped,
    _wandb_metrics,
    retry_stockfish_eval,
    run_stockfish_eval,
    stockfish_evaluation_protocol,
)
from luna.game.stockfish_ladder import (
    LADDER_STATE_NAME,
    load_fairy_ladder_state,
    run_fairy_ladder_eval,
    write_fairy_ladder_state,
)
from luna.network import LunaNetwork

if TYPE_CHECKING:
    from luna.coach import Coach


def external_checkpoint_path(coach: Coach, iteration: int) -> Path:
    candidates: list[Path] = []
    if coach.nnet._loaded_checkpoint_path is not None:
        candidates.append(coach.nnet._loaded_checkpoint_path)
    folder = Path(coach.run.checkpoint).expanduser().resolve()
    candidates.extend((folder / f"checkpoint_{iteration}.pth.tar", folder / "latest.pth.tar"))
    for candidate in candidates:
        if candidate.is_file() and LunaNetwork.checkpoint_trainer_iteration(candidate) == iteration:
            return candidate
    raise FileNotFoundError(f"No immutable checkpoint is available for scheduled evaluation at iteration {iteration}")


def initialize_external_evaluation_sidecars(coach: Coach, iteration: int) -> None:
    folder = Path(coach.run.checkpoint).expanduser().resolve()
    if coach._initialize_evaluation_state:
        folder.mkdir(parents=True, exist_ok=True)
    if coach.run.stockfish_eval_every > 0:
        benchmark_path = folder / BENCHMARK_STATE_NAME
        benchmark_protocol = asdict(stockfish_evaluation_protocol(coach.run))
        required = not coach._initialize_evaluation_state and iteration > coach.run.stockfish_eval_every
        benchmark_state = load_benchmark_state(benchmark_path, benchmark_protocol, required=required)
        if benchmark_state.last_iteration is not None and benchmark_state.last_iteration > iteration:
            raise RuntimeError("Fixed benchmark state is newer than the loaded checkpoint")
        if coach._initialize_evaluation_state and not benchmark_path.exists():
            write_benchmark_state(benchmark_path, benchmark_state)
    if coach.run.ladder_eval_every > 0:
        ladder_path = folder / LADDER_STATE_NAME
        required = not coach._initialize_evaluation_state and iteration > coach.run.ladder_eval_every
        ladder_state = load_fairy_ladder_state(ladder_path, coach.run, required=required)
        if ladder_state.last_iteration is not None and ladder_state.last_iteration > iteration:
            raise RuntimeError("Fairy ladder state is newer than the loaded checkpoint")
        if coach._initialize_evaluation_state and not ladder_path.exists():
            write_fairy_ladder_state(ladder_path, ladder_state)


def run_fixed_benchmark(
    coach: Coach,
    iteration: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> BenchmarkState:
    folder = Path(coach.run.checkpoint).expanduser().resolve()
    state_path = folder / BENCHMARK_STATE_NAME
    protocol = asdict(stockfish_evaluation_protocol(coach.run))
    state = load_benchmark_state(
        state_path,
        protocol,
        required=state_path.exists() or iteration > coach.run.stockfish_eval_every,
    )
    if state.last_iteration is not None and state.last_iteration > iteration:
        raise RuntimeError("Fixed benchmark state is newer than the loaded checkpoint")
    if state.last_iteration == iteration:
        if state.last_checkpoint_sha256 != checkpoint_sha256 or state.last_scores is None:
            raise RuntimeError("Fixed benchmark checkpoint identity differs from its durable result")
        scores = state.last_scores
        logger.info("Fixed benchmark iteration {} already completed; reconciling outputs", iteration)
        duration_seconds = None
    else:
        started_at = time.perf_counter()
        outcome = retry_stockfish_eval(
            lambda: run_stockfish_eval(
                coach.game,
                coach.nnet,
                coach.run,
                iteration=iteration,
                metric_prefix=None,
            ),
            attempts=coach.run.external_eval_attempts,
            retry_seconds=coach.run.external_eval_retry_seconds,
        )
        if isinstance(outcome, StockfishEvalSkipped):
            raise RuntimeError(f"External evaluation did not complete ({outcome.reason}): {outcome.message}")
        scores = outcome
        state = record_benchmark_result(
            state_path,
            protocol,
            iteration=iteration,
            checkpoint_sha256=checkpoint_sha256,
            scores=scores,
        )
        duration_seconds = time.perf_counter() - started_at
    if wandb.run is not None:
        metrics = _wandb_metrics(
            scores,
            iteration,
            opponent_elo=coach.run.stockfish_elo,
            duration_seconds=duration_seconds,
        )
        metrics["benchmark/evaluation_step"] = state.evaluation_step
        wandb.log(metrics)
    coach._update_best_from_stockfish(iteration, scores, checkpoint_path=checkpoint_path)
    return state


def reconcile_current_evaluations(coach: Coach, iteration: int) -> None:
    if iteration < 1:
        return
    fixed_due = coach.run.stockfish_eval_every > 0 and iteration % coach.run.stockfish_eval_every == 0
    ladder_due = coach.run.ladder_eval_every > 0 and iteration % coach.run.ladder_eval_every == 0
    if not fixed_due and not ladder_due:
        return
    checkpoint_path = external_checkpoint_path(coach, iteration)
    checkpoint_sha256 = coach._checkpoint_sha256(checkpoint_path)
    if fixed_due:
        run_fixed_benchmark(coach, iteration, checkpoint_path, checkpoint_sha256)
    if ladder_due:
        run_fairy_ladder_eval(
            coach.game,
            coach.nnet,
            coach.run,
            iteration=iteration,
            checkpoint_sha256=checkpoint_sha256,
            state_required=(Path(coach.run.checkpoint).expanduser().resolve() / LADDER_STATE_NAME).exists(),
        )
