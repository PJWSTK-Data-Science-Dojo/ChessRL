"""Luna-Chess EfficientZeroV2 training entry point."""

import random
import sys
from dataclasses import asdict, dataclass, replace
from hashlib import file_digest
from pathlib import Path

import numpy as np
import torch
import tyro
import wandb
from loguru import logger

from luna.coach import Coach, validate_fresh_checkpoint_target, validate_resume_checkpoint_target
from luna.coach_checkpoints import publish_bootstrap_checkpoint
from luna.config import (
    EzV2LearnerConfig,
    TrainCliConfig,
    TrainingRunConfig,
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
from luna.network import LunaNetwork, RepresentationCollapseError
from luna.online_checkpoints import resolve_resume_checkpoint, validate_new_training_phase_target


@dataclass(frozen=True)
class _CheckpointPlan:
    source: Path
    target: Path
    cross_directory: bool
    loaded_iteration: int = 0
    migration_source_sha256: str | None = None


@dataclass(frozen=True)
class _TrainingSetup:
    cli: TrainCliConfig
    run: TrainingRunConfig
    learner: EzV2LearnerConfig
    checkpoint: _CheckpointPlan


def _configure_logging(level: str) -> bool:
    logger.remove()
    try:
        logger.add(sys.stderr, level=level.upper())
    except ValueError:
        logger.add(sys.stderr, level="INFO")
        logger.error("Invalid log level: {!r}", level)
        return False
    return True


def _validate_cli(cli: TrainCliConfig, run: TrainingRunConfig, learner: EzV2LearnerConfig) -> None:
    validate_training_configuration(run, learner)
    validate_wandb_run_id(cli.wandb_run_id)
    validate_wandb_run_name(cli.wandb_run_name)
    validate_wandb_resume(cli.wandb_resume)
    if cli.load_model and cli.new_training_phase:
        raise ValueError("--load-model and --new-training-phase are mutually exclusive")


def _checkpoint_plan(cli: TrainCliConfig, run: TrainingRunConfig) -> _CheckpointPlan:
    source = Path(cli.load_checkpoint_dir) / cli.load_checkpoint_file
    target = Path(run.checkpoint).expanduser().resolve()
    cross_directory = cli.load_model and source.expanduser().resolve().parent != target
    if cli.initialize_evaluation_state and not cross_directory:
        raise ValueError(
            "--initialize-evaluation-state is only valid when migrating a loaded checkpoint to a new directory"
        )
    if cross_directory and _evaluation_enabled(run) and not cli.initialize_evaluation_state:
        raise ValueError("Cross-directory resume with external evaluation requires --initialize-evaluation-state")
    if cli.load_model:
        source = resolve_resume_checkpoint(source, source.parent)
    return _CheckpointPlan(source, target, cross_directory)


def _evaluation_enabled(run: TrainingRunConfig) -> bool:
    return run.stockfish_eval_every > 0 or run.ladder_eval_every > 0


def _validate_checkpoint_mode(
    cli: TrainCliConfig,
    run: TrainingRunConfig,
    plan: _CheckpointPlan,
) -> _CheckpointPlan:
    if cli.load_model:
        _require_source_checkpoint(plan.source)
        validate_resume_checkpoint_target(
            run,
            plan.source,
            allow_evaluation_artifacts_only=cli.initialize_evaluation_state,
        )
        iteration = LunaNetwork.checkpoint_trainer_iteration(plan.source.expanduser())
        digest = _checkpoint_sha256(plan.source) if cli.initialize_evaluation_state else None
        return replace(plan, loaded_iteration=iteration, migration_source_sha256=digest)
    if cli.new_training_phase:
        _require_source_checkpoint(plan.source)
        validate_new_training_phase_target(run.checkpoint)
        return plan
    validate_fresh_checkpoint_target(run)
    return plan


def _require_source_checkpoint(source: Path) -> None:
    if not source.expanduser().is_file():
        raise FileNotFoundError(f"No model in path {source.expanduser().resolve()}")


def _checkpoint_sha256(source: Path) -> str:
    with source.expanduser().open("rb") as source_stream:
        return file_digest(source_stream, "sha256").hexdigest()


def _validate_fixed_evaluation(cli: TrainCliConfig, run: TrainingRunConfig, plan: _CheckpointPlan) -> None:
    if run.stockfish_eval_every <= 0:
        return
    validate_stockfish_configuration(run)
    state = load_benchmark_state(
        plan.target / BENCHMARK_STATE_NAME,
        asdict(stockfish_evaluation_protocol(run)),
        required=_sidecar_required(cli, plan.loaded_iteration, run.stockfish_eval_every),
    )
    if cli.initialize_evaluation_state and state.last_iteration is not None:
        _validate_migration_lineage(plan, state.last_iteration, state.last_checkpoint_sha256, "benchmark state")
    best_record = Coach.validate_best_evaluation_contract(run)
    if cli.initialize_evaluation_state and best_record is not None:
        _validate_migration_lineage(
            plan,
            best_record.get("iteration"),
            best_record.get("source_checkpoint_sha256"),
            "best checkpoint",
        )


def _validate_ladder_evaluation(cli: TrainCliConfig, run: TrainingRunConfig, plan: _CheckpointPlan) -> None:
    if run.ladder_eval_every <= 0:
        return
    validate_ladder_configuration(run)
    state = load_fairy_ladder_state(
        plan.target / LADDER_STATE_NAME,
        run,
        required=_sidecar_required(cli, plan.loaded_iteration, run.ladder_eval_every),
    )
    if cli.initialize_evaluation_state and state.last_iteration is not None:
        _validate_migration_lineage(plan, state.last_iteration, state.last_checkpoint_sha256, "ladder state")


def _sidecar_required(cli: TrainCliConfig, loaded_iteration: int, frequency: int) -> bool:
    return cli.load_model and not cli.initialize_evaluation_state and loaded_iteration > frequency


def _validate_migration_lineage(
    plan: _CheckpointPlan,
    iteration: object,
    checkpoint_sha256: object,
    artifact: str,
) -> None:
    if iteration != plan.loaded_iteration or checkpoint_sha256 != plan.migration_source_sha256:
        raise RuntimeError(f"Migrated {artifact} does not belong to the loaded source checkpoint")


def _prepare_training_setup(cli: TrainCliConfig) -> _TrainingSetup:
    run = cli.to_training_run()
    learner = cli.to_learner_config()
    learner.discount = run.discount
    _validate_cli(cli, run, learner)
    plan = _validate_checkpoint_mode(cli, run, _checkpoint_plan(cli, run))
    _validate_fixed_evaluation(cli, run, plan)
    _validate_ladder_evaluation(cli, run, plan)
    return _TrainingSetup(cli, run, learner, plan)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _initialize_network(setup: _TrainingSetup, game: Game) -> LunaNetwork:
    logger.info("Loading {}...", LunaNetwork.__name__)
    network = LunaNetwork(game, setup.learner)
    network.log_model_summary()
    if setup.cli.load_model:
        source = setup.checkpoint.source
        logger.info('Loading checkpoint "{}"...', source)
        network.load_checkpoint(str(source.parent), source.name)
    elif setup.cli.new_training_phase:
        logger.info(
            'Starting a new training phase from weights in "{}" / "{}"; optimizer state starts fresh.',
            setup.cli.load_checkpoint_dir,
            setup.cli.load_checkpoint_file,
        )
        network.initialize_training_phase(setup.cli.load_checkpoint_dir, setup.cli.load_checkpoint_file)
        publish_bootstrap_checkpoint(network, setup.run.checkpoint)
    else:
        logger.info("Starting a new run from randomly initialized weights.")
    return network


def _log_profiling(run: TrainingRunConfig) -> None:
    if not run.profile:
        return
    logger.info(
        "Profiling on: phase timings each iter; Kineto on iter {} ({} steps) -> chrome in {} | TensorBoard logdir={}",
        run.profile_torch_iter,
        run.profile_torch_steps,
        run.profile_dir,
        run.profile_tensorboard_logdir,
    )
    logger.info(
        "Kineto traces are *.pt.trace.json under the TensorBoard logdir. "
        "Run uv run tensorboard --logdir <dir>, or load the trace in chrome://tracing."
    )


def _build_coach(setup: _TrainingSetup, game: Game, network: LunaNetwork) -> Coach:
    logger.info("Loading the Coach...")
    cli = setup.cli
    return Coach(
        game,
        network,
        setup.run,
        wandb_project=cli.wandb_project,
        wandb_run_id=cli.wandb_run_id,
        wandb_run_name=cli.wandb_run_name,
        wandb_resume=cli.wandb_resume,
        initialize_evaluation_state=cli.initialize_evaluation_state,
        restore_replay=cli.load_model,
        seed=cli.seed,
    )


def _learn(coach: Coach) -> int:
    logger.info("Starting EfficientZeroV2 learning process")
    try:
        coach.learn()
    except KeyboardInterrupt:
        logger.info("Training interrupted; the latest completed checkpoint remains available for resume.")
        return 130
    except RepresentationCollapseError as exc:
        logger.critical("Training stopped by the representation-collapse guard: {}", exc)
        return 78
    return 0


def _learn_and_finish_wandb(coach: Coach) -> int:
    exit_code = 1
    try:
        exit_code = _learn(coach)
        return exit_code
    finally:
        if wandb.run is not None:
            wandb.finish(exit_code=exit_code)


def main() -> int:
    torch.set_float32_matmul_precision("medium")

    cfg = tyro.cli(TrainCliConfig)
    if not _configure_logging(cfg.log_level):
        return 2

    try:
        setup = _prepare_training_setup(cfg)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Invalid training setup: {}", exc)
        return 2

    _seed_everything(cfg.seed)
    logger.info("Loading {}...", Game.__name__)
    game = Game()
    network = _initialize_network(setup, game)
    _log_profiling(setup.run)
    return _learn_and_finish_wandb(_build_coach(setup, game, network))


if __name__ == "__main__":
    sys.exit(main())
