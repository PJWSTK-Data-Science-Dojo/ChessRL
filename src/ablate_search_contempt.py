"""Measure Search-contempt self-play behavior on immutable model weights."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import tyro
from loguru import logger

from luna.coach import Coach
from luna.config import TrainingRunConfig
from luna.game.checkpoint_arena import checkpoint_identity
from luna.game.chess_game import ChessGame
from luna.network import LunaNetwork
from luna.search_contempt_ablation import (
    AblationReport,
    ArmMetrics,
    ArmResult,
    ArmSeedResult,
    SearchContemptAblationCli,
    ablation_protocol,
    aggregate_metrics,
    trajectory_metrics,
    validate_cli,
    write_report,
)

_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class _Arm:
    name: str
    visit_limit: int | None
    temperature_ply: int


@dataclass(frozen=True, slots=True)
class _AblationContext:
    cli: SearchContemptAblationCli
    game: ChessGame
    network: LunaNetwork


def main() -> int:
    cli = tyro.cli(SearchContemptAblationCli)
    _configure_logging(cli.log_level)
    try:
        report = run(cli)
        output = write_report(cli.output, report, overwrite=cli.overwrite)
    except (FileNotFoundError, OSError, RuntimeError, subprocess.CalledProcessError, ValueError):
        logger.exception("Search-contempt ablation failed")
        return 2
    logger.info("Search-contempt ablation written to {}", output)
    sys.stdout.write(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
    return 0


def run(cli: SearchContemptAblationCli) -> AblationReport:
    validate_cli(cli)
    protocol = ablation_protocol(cli)
    if protocol.git_dirty:
        raise RuntimeError("Search-contempt ablation requires a clean Git worktree")
    identity = checkpoint_identity(cli.checkpoint)
    game = ChessGame()
    network = LunaNetwork.from_checkpoint(
        game,
        cli.checkpoint,
        device=cli.device,
        cuda_device=cli.cuda_device,
        compile_inference=cli.compile_inference,
        load_optimizer=False,
    )
    network.warmup_mcts_inference(game)
    results = _run_arms(_AblationContext(cli, game, network), _arms(cli))
    if checkpoint_identity(cli.checkpoint) != identity:
        raise RuntimeError("Checkpoint changed during Search-contempt ablation")
    final_protocol = ablation_protocol(cli)
    if final_protocol.git_commit != protocol.git_commit or final_protocol.git_dirty:
        raise RuntimeError("Git state changed during Search-contempt ablation")
    return AblationReport(
        _SCHEMA_VERSION,
        identity,
        cli.num_mcts_sims,
        cli.gumbel_max_considered_actions,
        cli.games_per_seed,
        protocol,
        results,
    )


def _arms(cli: SearchContemptAblationCli) -> list[_Arm]:
    controls = [
        _Arm("current-control", None, cli.current_temperature_ply),
        _Arm("low-noise-control", None, cli.candidate_temperature_ply),
    ]
    candidates = [_Arm(f"search-contempt-{limit}", limit, cli.candidate_temperature_ply) for limit in cli.node_limits]
    return controls + candidates


def _run_arms(context: _AblationContext, arms: list[_Arm]) -> list[ArmResult]:
    _warm_up(context, arms[0])
    arm_results: dict[str, list[ArmSeedResult]] = {arm.name: [] for arm in arms}
    for seed_index, seed in enumerate(context.cli.seeds):
        for execution_index, arm in enumerate(_counterbalanced_order(arms, seed_index)):
            metrics = _run_seed(context, arm, seed)
            arm_results[arm.name].append(ArmSeedResult(seed, execution_index, metrics))
    return [_arm_result(context.cli, arm, arm_results[arm.name]) for arm in arms]


def _warm_up(context: _AblationContext, arm: _Arm) -> None:
    warmup_games = min(context.cli.parallel_games, context.cli.games_per_seed)
    _seed_everything(2**31 - 1)
    coach = Coach(context.game, context.network, _run_config(context.cli, arm, warmup_games), seed=2**31 - 1)
    coach.execute_episodes_batched(warmup_games, progress=False)
    if context.network.device.type == "cuda":
        torch.cuda.synchronize(context.network.device)


def _counterbalanced_order(arms: list[_Arm], seed_index: int) -> list[_Arm]:
    offset = seed_index % len(arms)
    return arms[offset:] + arms[:offset]


def _run_seed(context: _AblationContext, arm: _Arm, seed: int) -> ArmMetrics:
    _seed_everything(seed)
    coach = Coach(context.game, context.network, _run_config(context.cli, arm), seed=seed)
    started_at = time.perf_counter()
    trajectories = coach.execute_episodes_batched(context.cli.games_per_seed, progress=True)
    return trajectory_metrics(trajectories, time.perf_counter() - started_at)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _arm_result(cli: SearchContemptAblationCli, arm: _Arm, per_seed: list[ArmSeedResult]) -> ArmResult:
    metrics = aggregate_metrics([result.metrics for result in per_seed])
    logger.info(
        "{}: games={} white/draw/black={}/{}/{} truncated={} pos/s={:.1f} TS={:.3f}",
        arm.name,
        metrics.games,
        metrics.white_wins,
        metrics.draws,
        metrics.black_wins,
        metrics.truncated,
        metrics.positions_per_second,
        metrics.thompson_fraction,
    )
    return ArmResult(
        arm.name,
        arm.visit_limit,
        arm.temperature_ply,
        cli.seeds,
        asdict(_run_config(cli, arm)),
        metrics,
        per_seed,
    )


def _run_config(
    cli: SearchContemptAblationCli,
    arm: _Arm,
    num_episodes: int | None = None,
) -> TrainingRunConfig:
    episodes = cli.games_per_seed if num_episodes is None else num_episodes
    return TrainingRunConfig(
        num_mcts_sims=cli.num_mcts_sims,
        search_mode="gumbel",
        gumbel_max_considered_actions=cli.gumbel_max_considered_actions,
        dir_noise=False,
        search_contempt_visit_limit=arm.visit_limit,
        num_episodes=episodes,
        parallel_games=min(cli.parallel_games, episodes),
        temp_threshold=arm.temperature_ply,
        self_play_repetition_guard=True,
        max_ply=cli.max_ply,
        stockfish_eval_every=0,
        ladder_eval_every=0,
    )


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper())


if __name__ == "__main__":
    sys.exit(main())
