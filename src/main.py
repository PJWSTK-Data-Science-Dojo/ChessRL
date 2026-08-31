"""Luna-Chess EfficientZeroV2 training entry point."""

import random
import sys
from dataclasses import asdict
from hashlib import file_digest
from pathlib import Path

import numpy as np
import torch
import tyro
from loguru import logger

from luna.coach import Coach, validate_fresh_checkpoint_target, validate_resume_checkpoint_target
from luna.config import (
    TrainCliConfig,
    validate_training_configuration,
    validate_wandb_resume,
    validate_wandb_run_id,
    validate_wandb_run_name,
)
from luna.game.benchmark_state import BENCHMARK_STATE_NAME, load_benchmark_state
from luna.game.chess_game import ChessGame as Game
from luna.game.stockfish_eval import (
    stockfish_evaluation_protocol,
    validate_ladder_configuration,
    validate_stockfish_configuration,
)
from luna.game.stockfish_ladder import LADDER_STATE_NAME, load_fairy_ladder_state
from luna.network import LunaNetwork


def validate_new_training_phase_target(checkpoint_dir: str) -> None:
    """Require a dedicated empty directory for a weights-only training phase."""
    if not checkpoint_dir.strip():
        raise ValueError("new_training_phase requires a non-empty --run.checkpoint directory")
    target = Path(checkpoint_dir).expanduser().resolve()
    if not target.exists():
        return
    if not target.is_dir():
        raise FileExistsError(f"New training phase target is not a directory: {target}")
    contents = sorted(path.name for path in target.iterdir())
    if contents:
        raise FileExistsError(
            f"New training phase requires an empty checkpoint directory, but {target} contains {contents}. "
            "Choose a new --run.checkpoint directory."
        )


def resolve_resume_checkpoint(requested: Path, target: Path) -> Path:
    """Select the newest immutable checkpoint when ``latest`` lags after a crash."""
    resolved = requested.expanduser().resolve()
    if resolved.name != "latest.pth.tar" or resolved.parent != target.expanduser().resolve():
        return resolved

    candidates: list[tuple[int, Path]] = []
    if resolved.is_file():
        candidates.append((LunaNetwork.checkpoint_trainer_iteration(resolved), resolved))
    for numbered in resolved.parent.glob("checkpoint_*.pth.tar"):
        suffix = numbered.name.removeprefix("checkpoint_").removesuffix(".pth.tar")
        try:
            filename_iteration = int(suffix)
        except ValueError as exc:
            raise RuntimeError(f"Invalid numbered checkpoint name: {numbered}") from exc
        checkpoint_iteration = LunaNetwork.checkpoint_trainer_iteration(numbered)
        if checkpoint_iteration != filename_iteration:
            raise RuntimeError(
                f"Numbered checkpoint iteration {checkpoint_iteration} differs from its filename: {numbered}"
            )
        candidates.append((checkpoint_iteration, numbered))
    if not candidates:
        raise FileNotFoundError(f"No resumable checkpoint in {resolved.parent}")

    _, selected = max(candidates, key=lambda candidate: candidate[0])
    if selected != resolved:
        logger.warning('Recovering from newest immutable checkpoint "{}" instead of "{}"', selected, resolved)
    return selected


def main() -> int:
    torch.set_float32_matmul_precision("medium")

    cfg = tyro.cli(TrainCliConfig)

    logger.remove()
    try:
        logger.add(sys.stderr, level=cfg.log_level.upper())
    except ValueError:
        logger.add(sys.stderr, level="INFO")
        logger.error("Invalid log level: {!r}", cfg.log_level)
        return 2

    run_cfg = cfg.to_training_run()
    learner = cfg.to_learner_config()
    learner.discount = run_cfg.discount
    try:
        validate_training_configuration(run_cfg, learner)
        validate_wandb_run_id(cfg.wandb_run_id)
        validate_wandb_run_name(cfg.wandb_run_name)
        validate_wandb_resume(cfg.wandb_resume)
        if cfg.load_model and cfg.new_training_phase:
            raise ValueError("--load-model and --new-training-phase are mutually exclusive")
        source_checkpoint = Path(cfg.load_checkpoint_dir) / cfg.load_checkpoint_file
        source_parent = source_checkpoint.expanduser().resolve().parent
        target = Path(run_cfg.checkpoint).expanduser().resolve()
        cross_directory_resume = cfg.load_model and source_parent != target
        if cfg.initialize_evaluation_state and not cross_directory_resume:
            raise ValueError(
                "--initialize-evaluation-state is only valid when migrating a loaded checkpoint to a new directory"
            )
        if (
            cross_directory_resume
            and (run_cfg.stockfish_eval_every > 0 or run_cfg.ladder_eval_every > 0)
            and not cfg.initialize_evaluation_state
        ):
            raise ValueError("Cross-directory resume with external evaluation requires --initialize-evaluation-state")
        if cfg.load_model and not cross_directory_resume:
            source_checkpoint = resolve_resume_checkpoint(source_checkpoint, target)
        loaded_iteration = 0
        if cfg.load_model:
            if not source_checkpoint.expanduser().is_file():
                raise FileNotFoundError(f"No model in path {source_checkpoint.expanduser().resolve()}")
            validate_resume_checkpoint_target(
                run_cfg,
                source_checkpoint,
                allow_evaluation_artifacts_only=cfg.initialize_evaluation_state,
            )
            loaded_iteration = LunaNetwork.checkpoint_trainer_iteration(source_checkpoint.expanduser())
            migration_source_sha256: str | None = None
            if cfg.initialize_evaluation_state:
                with source_checkpoint.expanduser().open("rb") as source_stream:
                    migration_source_sha256 = file_digest(source_stream, "sha256").hexdigest()
        elif cfg.new_training_phase:
            if not source_checkpoint.expanduser().is_file():
                raise FileNotFoundError(f"No model in path {source_checkpoint.expanduser().resolve()}")
            validate_new_training_phase_target(run_cfg.checkpoint)
        else:
            validate_fresh_checkpoint_target(run_cfg)
        if run_cfg.stockfish_eval_every > 0:
            validate_stockfish_configuration(run_cfg)
            benchmark_path = target / BENCHMARK_STATE_NAME
            benchmark_state = load_benchmark_state(
                benchmark_path,
                asdict(stockfish_evaluation_protocol(run_cfg)),
                required=(
                    cfg.load_model
                    and not cfg.initialize_evaluation_state
                    and loaded_iteration > run_cfg.stockfish_eval_every
                ),
            )
            if (
                cfg.initialize_evaluation_state
                and benchmark_state.last_iteration is not None
                and (
                    benchmark_state.last_iteration != loaded_iteration
                    or benchmark_state.last_checkpoint_sha256 != migration_source_sha256
                )
            ):
                raise RuntimeError("Migrated benchmark state does not belong to the loaded source checkpoint")
            best_record = Coach.validate_best_evaluation_contract(run_cfg)
            if (
                cfg.initialize_evaluation_state
                and best_record is not None
                and (
                    best_record.get("iteration") != loaded_iteration
                    or best_record.get("source_checkpoint_sha256") != migration_source_sha256
                )
            ):
                raise RuntimeError("Migrated best checkpoint does not belong to the loaded source checkpoint")
        if run_cfg.ladder_eval_every > 0:
            validate_ladder_configuration(run_cfg)
            ladder_path = Path(run_cfg.checkpoint).expanduser().resolve() / LADDER_STATE_NAME
            ladder_state = load_fairy_ladder_state(
                ladder_path,
                run_cfg,
                required=(
                    cfg.load_model
                    and not cfg.initialize_evaluation_state
                    and loaded_iteration > run_cfg.ladder_eval_every
                ),
            )
            if (
                cfg.initialize_evaluation_state
                and ladder_state.last_iteration is not None
                and (
                    ladder_state.last_iteration != loaded_iteration
                    or ladder_state.last_checkpoint_sha256 != migration_source_sha256
                )
            ):
                raise RuntimeError("Migrated ladder state does not belong to the loaded source checkpoint")
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Invalid training setup: {}", exc)
        return 2

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    logger.info("Loading {}...", Game.__name__)
    game = Game()

    logger.info("Loading {}...", LunaNetwork.__name__)
    nnet = LunaNetwork(game, learner)
    nnet.log_model_summary()

    if cfg.load_model:
        logger.info(
            'Loading checkpoint "{}"...',
            source_checkpoint,
        )
        nnet.load_checkpoint(str(source_checkpoint.parent), source_checkpoint.name)
    elif cfg.new_training_phase:
        logger.info(
            'Starting a new training phase from weights in "{}" / "{}"; '
            "optimizer, scaler, counters, and LR schedule will start fresh.",
            cfg.load_checkpoint_dir,
            cfg.load_checkpoint_file,
        )
        nnet.initialize_training_phase(cfg.load_checkpoint_dir, cfg.load_checkpoint_file)
    else:
        logger.info("Starting a new run from randomly initialized weights.")

    if run_cfg.profile:
        logger.info(
            "Profiling on: phase timings each iter; Kineto on iter {} ({} steps) -> chrome in {} | TensorBoard logdir={}",
            run_cfg.profile_torch_iter,
            run_cfg.profile_torch_steps,
            run_cfg.profile_dir,
            run_cfg.profile_tensorboard_logdir,
        )
        logger.info(
            "Kineto traces are *.pt.trace.json under your TensorBoard logdir (no scalars). "
            "Run: uv run tensorboard --logdir <that-dir>  then open the PYTORCH_PROFILER tab "
            "(needs torch-tb-profiler, listed in pyproject). Or load the same .json in chrome://tracing.",
        )

    logger.info("Loading the Coach...")
    c = Coach(
        game,
        nnet,
        run_cfg,
        wandb_project=cfg.wandb_project,
        wandb_run_id=cfg.wandb_run_id,
        wandb_run_name=cfg.wandb_run_name,
        wandb_resume=cfg.wandb_resume,
        initialize_evaluation_state=cfg.initialize_evaluation_state,
        seed=cfg.seed,
    )

    logger.info("Starting EfficientZeroV2 learning process")
    c.learn()

    return 0


if __name__ == "__main__":
    sys.exit(main())
