"""Validation for Luna training and search configuration."""

import math

from luna.config_models import (
    FAIRY_STOCKFISH_MAX_ELO,
    FAIRY_STOCKFISH_MIN_ELO,
    MAX_STOCKFISH_EVAL_GAMES,
    MODEL_NAMES,
    EzV2LearnerConfig,
    MCTSParams,
    TrainingRunConfig,
)


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _finite_at_least(name: str, value: float, minimum: float) -> None:
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")


def _probability(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def validate_mcts_params(params: MCTSParams) -> None:
    """Reject search settings that cannot produce a finite legal policy."""
    _positive_integer("num_mcts_sims", params.num_mcts_sims)
    _positive_integer("gumbel_max_considered_actions", params.gumbel_max_considered_actions)
    _finite_at_least("gumbel_scale", params.gumbel_scale, 0.0)
    _finite_at_least("gumbel_value_scale", params.gumbel_value_scale, 0.0)
    _finite_at_least("gumbel_maxvisit_init", params.gumbel_maxvisit_init, 0.0)
    _finite_at_least("cpuct", params.cpuct, 0.0)
    _finite_at_least("pb_c_base", params.pb_c_base, math.ulp(0.0))
    _finite_at_least("dir_alpha", params.dir_alpha, math.ulp(0.0))
    _probability("dir_fraction", params.dir_fraction)
    _probability("discount", params.discount)
    if params.recurrent_policy_topk is not None:
        _positive_integer("recurrent_policy_topk", params.recurrent_policy_topk)
    if params.search_contempt_visit_limit is not None:
        _positive_integer("search_contempt_visit_limit", params.search_contempt_visit_limit)


def _validate_training_schedule(run: TrainingRunConfig) -> None:
    for name in (
        "num_iters",
        "num_episodes",
        "parallel_games",
        "self_play_workers",
        "train_steps_per_iter",
        "replay_capacity",
    ):
        _positive_integer(name, getattr(run, name))
    _finite_at_least("self_play_actor_timeout_s", run.self_play_actor_timeout_s, math.ulp(0.0))
    if run.target_replay_ratio is not None:
        _finite_at_least("target_replay_ratio", run.target_replay_ratio, math.ulp(0.0))
    if run.lr_schedule_total_steps is not None:
        _positive_integer("lr_schedule_total_steps", run.lr_schedule_total_steps)
    _non_negative_integer("replay_warmup_positions", run.replay_warmup_positions)
    if run.replay_warmup_positions > run.replay_capacity:
        raise ValueError("replay_warmup_positions cannot exceed replay_capacity")
    _non_negative_integer("temp_threshold", run.temp_threshold)
    _non_negative_integer("stockfish_eval_every", run.stockfish_eval_every)
    _non_negative_integer("ladder_eval_every", run.ladder_eval_every)
    _non_negative_integer("profile_torch_steps", run.profile_torch_steps)
    _positive_integer("profile_torch_iter", run.profile_torch_iter)
    if run.evaluation_num_mcts_sims is not None:
        _positive_integer("evaluation_num_mcts_sims", run.evaluation_num_mcts_sims)
    if run.max_ply is not None:
        _positive_integer("max_ply", run.max_ply)
    if run.checkpoint_top_k is not None:
        _non_negative_integer("checkpoint_top_k", run.checkpoint_top_k)


def _validate_external_evaluation(run: TrainingRunConfig) -> None:
    _positive_integer("stockfish_depth", run.stockfish_depth)
    _positive_integer("stockfish_elo", run.stockfish_elo)
    _positive_integer("external_eval_attempts", run.external_eval_attempts)
    _finite_at_least("external_eval_retry_seconds", run.external_eval_retry_seconds, 0.0)
    if run.stockfish_eval_every > 0 and (run.stockfish_eval_games < 2 or run.stockfish_eval_games % 2):
        raise ValueError("stockfish_eval_games must be an even integer of at least 2 when evaluation is enabled")
    if run.stockfish_eval_every > 0 and run.stockfish_eval_games > MAX_STOCKFISH_EVAL_GAMES:
        raise ValueError(f"stockfish_eval_games cannot exceed {MAX_STOCKFISH_EVAL_GAMES}")
    if run.stockfish_eval_max_ply is not None:
        _positive_integer("stockfish_eval_max_ply", run.stockfish_eval_max_ply)
    _positive_integer("ladder_depth", run.ladder_depth)
    _positive_integer("ladder_start_elo", run.ladder_start_elo)
    _positive_integer("ladder_step_elo", run.ladder_step_elo)
    _positive_integer("ladder_max_elo", run.ladder_max_elo)
    _positive_integer("ladder_required_passes", run.ladder_required_passes)
    if run.ladder_eval_every > 0 and (run.ladder_eval_games < 2 or run.ladder_eval_games % 2):
        raise ValueError("ladder_eval_games must be an even integer of at least 2 when the ladder is enabled")
    if run.ladder_eval_every > 0 and run.ladder_eval_games > MAX_STOCKFISH_EVAL_GAMES:
        raise ValueError(f"ladder_eval_games cannot exceed {MAX_STOCKFISH_EVAL_GAMES}")
    if run.ladder_start_elo < FAIRY_STOCKFISH_MIN_ELO:
        raise ValueError(f"ladder_start_elo cannot be below Fairy-Stockfish's {FAIRY_STOCKFISH_MIN_ELO} floor")
    if run.ladder_max_elo > FAIRY_STOCKFISH_MAX_ELO:
        raise ValueError(f"ladder_max_elo cannot exceed Fairy-Stockfish's {FAIRY_STOCKFISH_MAX_ELO} ceiling")
    if run.ladder_start_elo > run.ladder_max_elo:
        raise ValueError("ladder_start_elo cannot exceed ladder_max_elo")
    if (run.ladder_max_elo - run.ladder_start_elo) % run.ladder_step_elo:
        raise ValueError("ladder_max_elo must be reachable from ladder_start_elo in exact ladder_step_elo increments")
    if run.ladder_eval_every > 0 and not run.ladder_path.strip():
        raise ValueError("ladder_path cannot be blank when the ladder is enabled")
    if run.ladder_eval_every > 0 and not run.checkpoint.strip():
        raise ValueError("checkpoint cannot be blank when the persistent ladder is enabled")
    if run.ladder_eval_max_ply is not None:
        _positive_integer("ladder_eval_max_ply", run.ladder_eval_max_ply)
    if run.profile and not run.profile_dir.strip():
        raise ValueError("profile_dir cannot be blank when profiling is enabled")
    if run.profile and not run.profile_summary_json.strip():
        raise ValueError("profile_summary_json cannot be blank when profiling is enabled")


def _validate_optimizer(learner: EzV2LearnerConfig) -> None:
    _finite_at_least("lr", learner.lr, 0.0)
    _finite_at_least("lr_min", learner.lr_min, 0.0)
    if learner.lr_min > learner.lr:
        raise ValueError("lr_min cannot exceed lr")
    _non_negative_integer("lr_warmup_steps", learner.lr_warmup_steps)
    _finite_at_least("weight_decay", learner.weight_decay, 0.0)
    _positive_integer("batch_size", learner.batch_size)
    _positive_integer("grad_accum_steps", learner.grad_accum_steps)
    if learner.batch_size % learner.grad_accum_steps:
        raise ValueError("batch_size must be divisible by grad_accum_steps")
    if not math.isfinite(learner.grad_clip_norm) or learner.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive and finite")
    _probability("recurrent_gradient_scale", learner.recurrent_gradient_scale)


def _validate_model(learner: EzV2LearnerConfig) -> None:
    if learner.model_name not in MODEL_NAMES:
        raise ValueError(f"model_name must be one of {MODEL_NAMES}")
    _positive_integer("num_channels", learner.num_channels)
    _positive_integer("proj_dim", learner.proj_dim)
    _positive_integer("support_size", learner.support_size)
    _non_negative_integer("repr_blocks", learner.repr_blocks)
    _non_negative_integer("dyn_blocks", learner.dyn_blocks)
    _positive_integer("unroll_steps", learner.unroll_steps)
    _non_negative_integer("td_steps", learner.td_steps)
    if learner.amp_dtype.lower() not in {"bfloat16", "float16"}:
        raise ValueError("amp_dtype must be 'bfloat16' or 'float16'")
    if learner.device.lower() not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be 'cuda', 'mps', or 'cpu'")
    if learner.cuda_device is not None:
        _non_negative_integer("cuda_device", learner.cuda_device)


def _validate_learning_objective(learner: EzV2LearnerConfig) -> None:
    _probability("discount", learner.discount)
    weights = (
        learner.policy_loss_weight,
        learner.value_loss_weight,
        learner.reward_loss_weight,
        learner.consistency_loss_weight,
        learner.reconstruction_loss_weight,
    )
    for name, value in zip(
        ("policy", "value", "reward", "consistency", "reconstruction"),
        weights,
        strict=True,
    ):
        _finite_at_least(f"{name}_loss_weight", value, 0.0)
    if not any(weights):
        raise ValueError("at least one training loss weight must be positive")
    if learner.reconstruction_loss_weight > 0.0 and learner.model_name != "balanced_reconstruction":
        raise ValueError("reconstruction_loss_weight requires model_name='balanced_reconstruction'")
    _non_negative_integer("dataloader_workers", learner.dataloader_workers)
    _non_negative_integer("reanalyze_mcts_sims", learner.reanalyze_mcts_sims)
    _probability("reanalyze_prob", learner.reanalyze_prob)
    _non_negative_integer("reanalyze_start_step", learner.reanalyze_start_step)


def validate_learner_config(learner: EzV2LearnerConfig) -> None:
    """Validate model, optimizer, and reanalysis settings before allocating the model."""
    _validate_optimizer(learner)
    _validate_model(learner)
    _validate_learning_objective(learner)


def validate_training_configuration(run: TrainingRunConfig, learner: EzV2LearnerConfig) -> None:
    """Validate a complete training run before self-play or model allocation."""
    validate_mcts_params(run)
    validate_learner_config(learner)
    _validate_training_schedule(run)
    _validate_external_evaluation(run)
    _probability("per_alpha", run.per_alpha)
    _probability("per_beta", run.per_beta)
    if run.replay_capacity < learner.batch_size:
        raise ValueError("replay_capacity must be at least batch_size or training can never start")


def validate_wandb_run_id(run_id: str | None) -> None:
    """Validate an optional run ID against the local W&B SDK contract."""
    if run_id is None:
        return
    if not run_id.strip():
        raise ValueError("wandb_run_id cannot be blank")
    if run_id != run_id.strip():
        raise ValueError("wandb_run_id cannot start or end with whitespace")
    reserved_characters = ":;,#?/'"
    if any(character in run_id for character in reserved_characters):
        raise ValueError(f"wandb_run_id cannot contain these characters: {reserved_characters}")


def validate_wandb_run_name(run_name: str | None) -> None:
    """Validate an optional human-readable W&B display name."""
    if run_name is None:
        return
    if not run_name.strip():
        raise ValueError("wandb_run_name cannot be blank")
    if run_name != run_name.strip():
        raise ValueError("wandb_run_name cannot start or end with whitespace")


def validate_wandb_resume(mode: str) -> None:
    """Validate the W&B run-resume policy after programmatic config construction."""
    allowed_modes = ("allow", "never", "must")
    if mode not in allowed_modes:
        raise ValueError(f"wandb_resume must be one of {allowed_modes}, got {mode!r}")
