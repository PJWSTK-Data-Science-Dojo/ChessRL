"""Typed configuration and validation for Luna training."""

import math
from dataclasses import dataclass, field, fields
from typing import Literal

MAX_STOCKFISH_EVAL_GAMES = 20
WandbResumeMode = Literal["allow", "never", "must"]


@dataclass
class MCTSParams:
    """Parameters for latent MuZero search.

    ``search_mode="gumbel"`` uses Gumbel top-m Sequential Halving at the root
    and deterministic improved-policy selection below it. ``"puct"`` keeps the
    classic MuZero PUCT search. Root Dirichlet settings apply only to PUCT.

    ``recurrent_policy_topk``: for batched MCTS only, transfer only the top-K log-prob
    actions from GPU after each recurrent forward. The implementation raises K to
    the exact legal-move count, so every legal action is retained; ``None`` copies
    the full policy vector.

    """

    num_mcts_sims: int = 50
    """Maximum simulations per move; search cost grows approximately linearly with this budget."""

    search_mode: Literal["gumbel", "puct"] = "gumbel"
    """Root selection algorithm; Gumbel search is intended for the small self-play budgets used here."""

    gumbel_max_considered_actions: int = 16
    """Maximum root candidates admitted to Sequential Halving."""

    gumbel_scale: float = 1.0
    """Scale of the sampled Gumbel logits at the root."""

    gumbel_value_scale: float = 0.1
    """Contribution of completed action values to Gumbel policy improvement."""

    gumbel_maxvisit_init: float = 50.0
    """Visit-count normalization constant used by completed-Q policy improvement."""

    cpuct: float = 1.25
    """Initial exploration weight for classic MuZero PUCT."""

    pb_c_base: float = 19_652.0
    """PUCT base controlling how exploration grows with parent visits."""

    dir_noise: bool = True
    """Inject root exploration noise during self-play; evaluation disables it explicitly."""

    dir_alpha: float = 0.3
    """Dirichlet concentration for PUCT root noise."""

    dir_fraction: float = 0.25
    """Fraction of the root prior replaced by Dirichlet noise."""

    discount: float = 1.0
    """Discount for latent rewards; chess uses its undiscounted terminal outcome."""

    recurrent_policy_topk: int | None = 256
    """Host-transfer width for recurrent policies; legal actions are always retained."""


@dataclass
class TrainingRunConfig(MCTSParams):
    """Self-play schedule, replay, external evaluation, and checkpoint paths.

    Inherits MCTS fields from :class:`MCTSParams`.

    Stockfish benchmark: every ``stockfish_eval_every`` iterations (default 50) runs
    ``stockfish_eval_games`` games vs Stockfish (alternating colors) after checkpoint publication.
    With default ``num_iters`` (100) you get two benchmarks (at 50 and 100). Set
    ``stockfish_eval_every`` to 0 to disable.
    """

    num_iters: int = 100
    """Number of self-play and learning iterations."""

    num_episodes: int = 20
    """Self-play games generated per iteration."""

    parallel_games: int = 8
    """Games advanced together per self-play process to batch network inference."""

    self_play_workers: int = 1
    """Spawned self-play processes; one keeps self-play in the learner process."""

    self_play_actor_timeout_s: float = 1_800.0
    """Maximum time for actor startup or one self-play collection."""

    temp_threshold: int = 15
    """Ply after which self-play switches to deterministic action selection."""

    evaluation_num_mcts_sims: int | None = None
    """Independent external-evaluation budget; the self-play budget is used when unset."""

    train_steps_per_iter: int = 200
    """Optimizer steps performed after each self-play collection phase."""

    replay_capacity: int = 100_000
    """Maximum number of positions retained in prioritized replay."""

    per_alpha: float = 0.6
    """Priority exponent; zero reduces sampling to uniform replay."""

    per_beta: float = 0.4
    """Initial importance-sampling correction exponent."""

    checkpoint: str = "./runs/luna-main"
    """Directory for versioned training-state checkpoints."""

    checkpoint_top_k: int | None = 3
    """Numbered checkpoints to retain; zero or ``None`` disables pruning."""

    max_ply: int | None = None
    """Optional self-play safety bound that scores unfinished games as draws."""

    profile: bool = False
    """Collect iteration phase timings."""

    profile_dir: str = "./profiles"
    """Directory for profiler artifacts."""

    profile_summary_json: str = "iter_timings.json"
    """Filename for the iteration timing summary inside the profile directory."""

    profile_torch_steps: int = 0
    """Optimizer steps captured by the PyTorch profiler; zero disables capture."""

    profile_torch_iter: int = 1
    """Training iteration on which to capture the profiler trace."""

    profile_export_chrome: bool = True
    """Export a Chrome-compatible trace when profiling is active."""

    profile_tensorboard_logdir: str | None = None
    """Optional TensorBoard trace destination."""

    profile_with_stack: bool = False
    """Capture Python stacks in traces at additional profiling cost."""

    stockfish_eval_every: int = 50
    """Checkpoint-evaluation interval; zero disables Stockfish evaluation."""

    stockfish_eval_games: int = 20
    """Even number of paired-opening games, up to ``MAX_STOCKFISH_EVAL_GAMES``."""

    stockfish_elo: int = 1320
    """Fixed UCI Elo for the external benchmark; 1320 is Stockfish's supported floor."""

    stockfish_depth: int = 10
    """Maximum Stockfish search depth per move."""

    stockfish_path: str | None = None
    """Optional explicit Stockfish executable path."""

    stockfish_eval_max_ply: int | None = None
    """Optional evaluation-game safety bound; unfinished games score as draws."""


def evaluation_mcts_params(run: TrainingRunConfig) -> MCTSParams:
    """Deterministic MCTS settings for external evaluation."""
    sims = run.evaluation_num_mcts_sims if run.evaluation_num_mcts_sims is not None else run.num_mcts_sims
    return MCTSParams(
        num_mcts_sims=sims,
        search_mode=run.search_mode,
        gumbel_max_considered_actions=run.gumbel_max_considered_actions,
        gumbel_scale=run.gumbel_scale,
        gumbel_value_scale=run.gumbel_value_scale,
        gumbel_maxvisit_init=run.gumbel_maxvisit_init,
        cpuct=run.cpuct,
        pb_c_base=run.pb_c_base,
        dir_noise=False,
        dir_alpha=run.dir_alpha,
        dir_fraction=run.dir_fraction,
        discount=run.discount,
        recurrent_policy_topk=run.recurrent_policy_topk,
    )


@dataclass
class EzV2LearnerConfig:
    """Optimizer, architecture, unroll training, and loss weights for :class:`LunaNetwork`."""

    lr: float = 2e-4
    """Peak AdamW learning rate after warm-up."""

    lr_min: float = 1e-5
    """Final learning rate reached by cosine decay."""

    lr_warmup_steps: int = 1_000
    """Optimizer steps used for linear learning-rate warm-up."""

    weight_decay: float = 1e-4
    """Decoupled AdamW weight decay."""

    batch_size: int = 32
    """Replay positions consumed by one optimizer step before accumulation splitting."""

    num_channels: int = 64
    """Hidden channel width shared by the latent model trunks."""

    support_size: int = 1
    """Half-width of categorical value and reward support; one exactly represents chess outcomes."""

    repr_blocks: int = 4
    """Residual blocks in the observation representation trunk."""

    dyn_blocks: int = 2
    """Residual blocks applied after each latent transition."""

    proj_dim: int = 256
    """Projection width used by the consistency objective."""

    mixed_precision: bool = True
    """Use accelerator autocast and gradient scaling when supported."""

    amp_dtype: str = "bfloat16"
    """Preferred CUDA autocast dtype; unsupported bfloat16 hosts fall back to float16."""

    unroll_steps: int = 5
    """Latent transition steps supervised from each replay position."""

    td_steps: int = 10
    """Bootstrap horizon for alternating-sign value targets."""

    discount: float = 1.0
    """Target discount for direct learner use; the training CLI mirrors ``run.discount`` here."""

    policy_loss_weight: float = 1.0
    """Relative weight of search-policy cross-entropy."""

    value_loss_weight: float = 0.25
    """Relative weight of categorical value prediction."""

    reward_loss_weight: float = 1.0
    """Relative weight of categorical latent-reward prediction."""

    consistency_loss_weight: float = 2.0
    """Relative weight of latent SimSiam consistency."""

    device: str = "cuda"
    """Learner backend: ``cuda``, ``mps``, or ``cpu``."""

    cuda_device: int | None = None
    """CUDA device index; the current default device is used when unset."""

    compile_inference: bool = False
    """Compile the MCTS inference paths after model construction."""

    compile_training: bool = False
    """Compile the unrolled training forward pass."""

    grad_accum_steps: int = 1
    """Microbatches combined into each optimizer step."""

    grad_clip_norm: float = 5.0
    """Maximum global gradient norm before the optimizer update."""

    recurrent_gradient_scale: float = 0.5
    """Scale applied to gradients crossing each recurrent dynamics edge."""

    dataloader_workers: int = 2
    """Replay-prefetch threads; zero keeps sampling on the training thread."""

    reanalyze_mcts_sims: int = 0
    """Search budget for replay reanalysis; zero disables reanalysis."""

    reanalyze_prob: float = 0.25
    """Fraction of sampled replay positions eligible for reanalysis."""

    reanalyze_policy: bool = True
    """Refresh policy targets together with values during reanalysis."""

    reanalyze_start_step: int = 5000
    """First optimizer step eligible for direct value and policy reanalysis."""


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
    _non_negative_integer("temp_threshold", run.temp_threshold)
    _non_negative_integer("stockfish_eval_every", run.stockfish_eval_every)
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
    if run.stockfish_eval_every > 0 and (run.stockfish_eval_games < 2 or run.stockfish_eval_games % 2):
        raise ValueError("stockfish_eval_games must be an even integer of at least 2 when evaluation is enabled")
    if run.stockfish_eval_every > 0 and run.stockfish_eval_games > MAX_STOCKFISH_EVAL_GAMES:
        raise ValueError(f"stockfish_eval_games cannot exceed {MAX_STOCKFISH_EVAL_GAMES}")
    if run.stockfish_eval_max_ply is not None:
        _positive_integer("stockfish_eval_max_ply", run.stockfish_eval_max_ply)
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
    )
    for name, value in zip(("policy", "value", "reward", "consistency"), weights):
        _finite_at_least(f"{name}_loss_weight", value, 0.0)
    if not any(weights):
        raise ValueError("at least one training loss weight must be positive")
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


@dataclass
class TrainCliConfig:
    """Full set of options exposed by ``python main.py`` (tyro).

    Composes :class:`TrainingRunConfig` and :class:`EzV2LearnerConfig` via
    nested dataclasses rather than duplicating their fields.
    """

    seed: int = 0
    log_level: str = "INFO"
    load_model: bool = False
    new_training_phase: bool = False
    """Load only model weights and reset all optimizer and training counters."""

    load_checkpoint_dir: str = "./runs/luna-main"
    load_checkpoint_file: str = "latest.pth.tar"
    wandb_project: str | None = None  # Optional WandB project name for experiment tracking
    wandb_run_id: str | None = None
    """Stable W&B run ID used to continue one run after a restart."""

    wandb_run_name: str | None = None
    """Human-readable W&B display name, independent of the stable run ID."""

    wandb_resume: WandbResumeMode = "allow"
    """Whether W&B may, must not, or must resume the stable run ID."""

    run: TrainingRunConfig = field(default_factory=TrainingRunConfig)
    learner: EzV2LearnerConfig = field(default_factory=EzV2LearnerConfig)

    def to_training_run(self) -> TrainingRunConfig:
        return TrainingRunConfig(**{f.name: getattr(self.run, f.name) for f in fields(self.run)})

    def to_learner_config(self) -> EzV2LearnerConfig:
        return EzV2LearnerConfig(**{f.name: getattr(self.learner, f.name) for f in fields(self.learner)})
