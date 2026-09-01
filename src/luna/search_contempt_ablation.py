"""Reporting and metrics for fixed-weight Search-contempt experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from luna.game.checkpoint_arena import CheckpointIdentity
from luna.replay_buffer import Trajectory


@dataclass(frozen=True, slots=True)
class SearchContemptAblationCli:
    checkpoint: Path
    output: Path
    games_per_seed: int = 32
    seeds: tuple[int, ...] = (7, 19, 43)
    node_limits: tuple[int, ...] = (2, 4, 8)
    num_mcts_sims: int = 32
    gumbel_max_considered_actions: int = 8
    current_temperature_ply: int = 257
    candidate_temperature_ply: int = 40
    parallel_games: int = 32
    max_ply: int = 256
    device: str = "cuda"
    cuda_device: int | None = 0
    compile_inference: bool = False
    log_level: str = "INFO"
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class ArmMetrics:
    games: int
    positions: int
    elapsed_seconds: float
    positions_per_second: float
    average_ply: float
    white_wins: int
    black_wins: int
    draws: int
    truncated: int
    policy_entropy: float
    repeated_prefix_8_fraction: float
    repeated_prefix_16_fraction: float
    repeated_prefix_32_fraction: float
    opponent_selections: int
    thompson_selections: int
    thompson_fraction: float
    frozen_nodes: int
    repetition_guard_attempts: int
    repetition_guard_interventions: int
    repetition_guard_forced_fallbacks: int
    repetition_guard_excluded_actions: int
    terminations: dict[str, int]


@dataclass(frozen=True, slots=True)
class ArmSeedResult:
    seed: int
    execution_index: int
    metrics: ArmMetrics


@dataclass(frozen=True, slots=True)
class ArmResult:
    name: str
    visit_limit: int | None
    temperature_ply: int
    seeds: tuple[int, ...]
    run_config: dict[str, object]
    metrics: ArmMetrics
    per_seed: list[ArmSeedResult]


@dataclass(frozen=True, slots=True)
class AblationProtocol:
    created_at_utc: str
    git_commit: str
    git_dirty: bool
    python_version: str
    torch_version: str
    cuda_version: str | None
    gpu_name: str | None
    device: str
    cuda_device: int | None
    compile_inference: bool
    parallel_games: int
    max_ply: int
    seeds: tuple[int, ...]
    node_limits: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AblationReport:
    schema_version: int
    checkpoint: CheckpointIdentity
    num_mcts_sims: int
    gumbel_max_considered_actions: int
    games_per_seed: int
    protocol: AblationProtocol
    arms: list[ArmResult]


def trajectory_metrics(trajectories: list[Trajectory], elapsed: float) -> ArmMetrics:
    games = len(trajectories)
    positions = sum(trajectory.game_length for trajectory in trajectories)
    white_wins, black_wins, draws, truncated = _outcome_counts(trajectories)
    opponent = sum(trajectory.search_contempt_opponent_selections for trajectory in trajectories)
    thompson = sum(trajectory.search_contempt_thompson_selections for trajectory in trajectories)
    return ArmMetrics(
        games=games,
        positions=positions,
        elapsed_seconds=elapsed,
        positions_per_second=positions / elapsed,
        average_ply=positions / games,
        white_wins=white_wins,
        black_wins=black_wins,
        draws=draws,
        truncated=truncated,
        policy_entropy=_policy_entropy(trajectories, positions),
        repeated_prefix_8_fraction=_repeated_prefix_fraction(trajectories, 8),
        repeated_prefix_16_fraction=_repeated_prefix_fraction(trajectories, 16),
        repeated_prefix_32_fraction=_repeated_prefix_fraction(trajectories, 32),
        opponent_selections=opponent,
        thompson_selections=thompson,
        thompson_fraction=thompson / opponent if opponent else 0.0,
        frozen_nodes=sum(item.search_contempt_frozen_nodes for item in trajectories),
        repetition_guard_attempts=sum(item.repetition_guard_attempts for item in trajectories),
        repetition_guard_interventions=sum(item.repetition_guard_interventions for item in trajectories),
        repetition_guard_forced_fallbacks=sum(item.repetition_guard_forced_fallbacks for item in trajectories),
        repetition_guard_excluded_actions=sum(item.repetition_guard_excluded_actions for item in trajectories),
        terminations=_termination_counts(trajectories),
    )


def aggregate_metrics(metrics: list[ArmMetrics]) -> ArmMetrics:
    games = sum(item.games for item in metrics)
    positions = sum(item.positions for item in metrics)
    elapsed = sum(item.elapsed_seconds for item in metrics)
    opponent = sum(item.opponent_selections for item in metrics)
    thompson = sum(item.thompson_selections for item in metrics)
    return ArmMetrics(
        games=games,
        positions=positions,
        elapsed_seconds=elapsed,
        positions_per_second=positions / elapsed,
        average_ply=positions / games,
        white_wins=sum(item.white_wins for item in metrics),
        black_wins=sum(item.black_wins for item in metrics),
        draws=sum(item.draws for item in metrics),
        truncated=sum(item.truncated for item in metrics),
        policy_entropy=_weighted_entropy(metrics, positions),
        repeated_prefix_8_fraction=_mean_prefix_fraction(metrics, 8),
        repeated_prefix_16_fraction=_mean_prefix_fraction(metrics, 16),
        repeated_prefix_32_fraction=_mean_prefix_fraction(metrics, 32),
        opponent_selections=opponent,
        thompson_selections=thompson,
        thompson_fraction=thompson / opponent if opponent else 0.0,
        frozen_nodes=sum(item.frozen_nodes for item in metrics),
        repetition_guard_attempts=sum(item.repetition_guard_attempts for item in metrics),
        repetition_guard_interventions=sum(item.repetition_guard_interventions for item in metrics),
        repetition_guard_forced_fallbacks=sum(item.repetition_guard_forced_fallbacks for item in metrics),
        repetition_guard_excluded_actions=sum(item.repetition_guard_excluded_actions for item in metrics),
        terminations=_aggregate_terminations(metrics),
    )


def validate_cli(cli: SearchContemptAblationCli) -> None:
    if cli.games_per_seed <= 0 or not cli.seeds:
        raise ValueError("games_per_seed and seeds must be non-empty and positive")
    if any(isinstance(seed, bool) or seed < 0 or seed >= 2**32 for seed in cli.seeds):
        raise ValueError("seeds must be integers in NumPy's uint32 range")
    if len(set(cli.seeds)) != len(cli.seeds):
        raise ValueError("seeds must be unique")
    if cli.num_mcts_sims <= 0 or cli.parallel_games <= 0 or cli.max_ply <= 0:
        raise ValueError("search budgets and max_ply must be positive")
    if not cli.node_limits or any(isinstance(limit, bool) or limit <= 0 for limit in cli.node_limits):
        raise ValueError("node_limits must contain positive integers")
    if len(set(cli.node_limits)) != len(cli.node_limits):
        raise ValueError("node_limits must be unique")
    if cli.output.expanduser().resolve() == cli.checkpoint.expanduser().resolve():
        raise ValueError("Ablation output cannot overwrite its checkpoint")
    if cli.output.expanduser().exists() and not cli.overwrite:
        raise FileExistsError(f"Ablation report already exists: {cli.output.expanduser().resolve()}")


def write_report(path: Path, report: AblationReport, *, overwrite: bool) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Ablation report already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    content = json.dumps(asdict(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def ablation_protocol(cli: SearchContemptAblationCli) -> AblationProtocol:
    gpu_name = None
    if cli.device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(cli.cuda_device or 0)
    return AblationProtocol(
        created_at_utc=datetime.now(UTC).isoformat(),
        git_commit=_git_output("rev-parse", "HEAD"),
        git_dirty=bool(_git_output("status", "--porcelain")),
        python_version=sys.version.split()[0],
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        gpu_name=gpu_name,
        device=cli.device,
        cuda_device=cli.cuda_device,
        compile_inference=cli.compile_inference,
        parallel_games=cli.parallel_games,
        max_ply=cli.max_ply,
        seeds=cli.seeds,
        node_limits=cli.node_limits,
    )


def _outcome_counts(trajectories: list[Trajectory]) -> tuple[int, int, int, int]:
    white_wins = black_wins = draws = truncated = 0
    for trajectory in trajectories:
        if trajectory.truncated:
            truncated += 1
        elif trajectory.rewards[-1] == 0.0:
            draws += 1
        elif _white_reward(trajectory) > 0.0:
            white_wins += 1
        else:
            black_wins += 1
    return white_wins, black_wins, draws, truncated


def _white_reward(trajectory: Trajectory) -> float:
    terminal_reward = float(trajectory.rewards[-1])
    return terminal_reward if trajectory.game_length % 2 else -terminal_reward


def _policy_entropy(trajectories: list[Trajectory], positions: int) -> float:
    total = 0.0
    for trajectory in trajectories:
        probabilities = trajectory.root_policies.astype(np.float32)
        positive = probabilities > 0.0
        total -= float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    return total / positions


def _repeated_prefix_fraction(trajectories: list[Trajectory], length: int) -> float:
    prefixes = [tuple(trajectory.actions[:length]) for trajectory in trajectories if trajectory.game_length >= length]
    if not prefixes:
        return 0.0
    counts = Counter(prefixes)
    return sum(counts[prefix] > 1 for prefix in prefixes) / len(prefixes)


def _termination_counts(trajectories: list[Trajectory]) -> dict[str, int]:
    counts = Counter(
        trajectory.termination.name.lower()
        for trajectory in trajectories
        if not trajectory.truncated and trajectory.termination is not None
    )
    return dict(sorted(counts.items()))


def _weighted_entropy(metrics: list[ArmMetrics], positions: int) -> float:
    return sum(item.policy_entropy * item.positions for item in metrics) / positions


def _mean_prefix_fraction(metrics: list[ArmMetrics], length: int) -> float:
    fractions = {
        8: [item.repeated_prefix_8_fraction for item in metrics],
        16: [item.repeated_prefix_16_fraction for item in metrics],
        32: [item.repeated_prefix_32_fraction for item in metrics],
    }
    return float(np.mean(fractions[length]))


def _aggregate_terminations(metrics: list[ArmMetrics]) -> dict[str, int]:
    terminations: Counter[str] = Counter()
    for item in metrics:
        terminations.update(item.terminations)
    return dict(sorted(terminations.items()))


def _git_output(*arguments: str) -> str:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
