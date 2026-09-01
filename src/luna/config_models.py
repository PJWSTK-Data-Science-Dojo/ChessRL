"""Typed configuration models for Luna training."""

from dataclasses import dataclass, field, fields
from typing import Literal

MAX_STOCKFISH_EVAL_GAMES = 20
FAIRY_STOCKFISH_MIN_ELO = 500
FAIRY_STOCKFISH_MAX_ELO = 2850
WandbResumeMode = Literal["allow", "never", "must"]
ModelName = Literal["baseline", "balanced", "balanced_reconstruction"]
MODEL_NAMES: tuple[ModelName, ...] = ("baseline", "balanced", "balanced_reconstruction")


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
    """Ply after which ordinary self-play exploration stops; Gumbel re-enables it for repeated roots."""

    self_play_repetition_guard: bool = False
    """Retry self-play root search without moves that immediately enable a threefold-repetition claim."""

    evaluation_num_mcts_sims: int | None = None
    """Independent external-evaluation budget; the self-play budget is used when unset."""

    train_steps_per_iter: int = 200
    """Maximum optimizer steps performed after each self-play collection phase."""

    target_replay_ratio: float | None = None
    """Optional target learner samples per newly generated position; the step cap still applies."""

    lr_schedule_total_steps: int | None = None
    """Optional fixed optimizer-step horizon for warm-up and cosine decay."""

    replay_capacity: int = 100_000
    """Maximum number of positions retained in prioritized replay."""

    replay_warmup_positions: int = 0
    """Minimum replay positions required before training; zero falls back to one learner batch."""

    per_alpha: float = 0.6
    """Priority exponent; zero reduces sampling to uniform replay."""

    per_beta: float = 0.4
    """Initial importance-sampling correction exponent."""

    checkpoint: str = "./runs/luna-main"
    """Directory for versioned training-state checkpoints."""

    checkpoint_top_k: int | None = 3
    """Numbered checkpoints to retain; zero or ``None`` disables pruning."""

    max_ply: int | None = None
    """Optional self-play safety bound; unfinished trajectories bootstrap from their boundary state."""

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

    stockfish_elo: int = 1500
    """Fixed official-Stockfish UCI Elo used only for comparable checkpoint promotion."""

    stockfish_depth: int = 10
    """Maximum Stockfish search depth per move."""

    stockfish_path: str | None = None
    """Optional explicit Stockfish executable path."""

    stockfish_eval_max_ply: int | None = None
    """Optional evaluation-game safety bound; unfinished games score as draws."""

    external_eval_attempts: int = 3
    """Maximum attempts for a transient external-engine evaluation failure."""

    external_eval_retry_seconds: float = 5.0
    """Delay between transient external-engine evaluation attempts."""

    ladder_eval_every: int = 0
    """Adaptive Fairy-Stockfish ladder interval; zero disables the ladder."""

    ladder_eval_games: int = 20
    """Even number of paired-opening games played at each ladder rung."""

    ladder_start_elo: int = FAIRY_STOCKFISH_MIN_ELO
    """First real UCI Elo rung; current Fairy-Stockfish supports no value below 500."""

    ladder_step_elo: int = 100
    """Elo increment after Luna scores more wins than losses at a rung."""

    ladder_max_elo: int = 2800
    """Highest 100-point Fairy-Stockfish rung below its native 2850 ceiling."""

    ladder_required_passes: int = 2
    """Consecutive majority-score matches required before advancing one rung."""

    ladder_depth: int = 10
    """Maximum Fairy-Stockfish search depth per move."""

    ladder_path: str = "./vendor/stockfish/fairy-stockfish-14"
    """Pinned Fairy-Stockfish executable used by the adaptive ladder."""

    ladder_eval_max_ply: int | None = None
    """Optional ladder-game safety bound; unfinished games score as draws."""


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

    model_name: ModelName = "baseline"
    """Model factory key, including the state-anchored ``balanced_reconstruction`` variant."""

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

    reconstruction_loss_weight: float = 0.0
    """Relative weight of the training-only piece-position reconstruction objective."""

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

    initialize_evaluation_state: bool = False
    """Explicitly initialize sidecars when migrating a checkpoint into a new evaluation contract."""

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
