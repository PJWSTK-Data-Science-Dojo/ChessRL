"""Typed configuration for MCTS, training loops, EZV2 learner, and the training CLI."""

from dataclasses import dataclass, field, fields
from typing import Literal


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

    CLI speed knobs (no code changes): raise ``parallel_games`` until memory-bound;
    lower ``num_mcts_sims`` / ``max_ply`` for wall time, and use
    ``evaluation_num_mcts_sims`` to give external evaluation a separate search budget.
    """

    # Number of MCTS simulations per move during self-play
    # AlphaZero used 800, but 50-200 is practical for training runs
    # Higher = stronger tactical play but slower iteration time
    num_mcts_sims: int = 50

    # Gumbel MuZero is substantially more simulation-efficient than visit-count
    # PUCT at the small search budgets used during self-play.
    search_mode: Literal["gumbel", "puct"] = "gumbel"
    gumbel_max_considered_actions: int = 16
    gumbel_scale: float = 1.0
    gumbel_value_scale: float = 0.1
    gumbel_maxvisit_init: float = 50.0

    # MuZero PUCT initialization constant (c1 in the dynamic exploration term).
    cpuct: float = 1.25
    # MuZero c2 constant; exploration grows logarithmically with parent visits.
    pb_c_base: float = 19_652.0

    dir_noise: bool = True

    # Dirichlet noise alpha for root exploration (AlphaZero)
    # Lower values = more concentrated noise
    # 0.3 is appropriate for chess (~35 legal moves average)
    # Formula from paper: alpha = 10/n where n is typical branching factor
    dir_alpha: float = 0.3
    dir_fraction: float = 0.25

    # Board games optimize the undiscounted game outcome.
    discount: float = 1.0

    recurrent_policy_topk: int | None = 256


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
    num_episodes: int = 20
    parallel_games: int = 8
    temp_threshold: int = 15
    evaluation_num_mcts_sims: int | None = None
    train_steps_per_iter: int = 200
    replay_capacity: int = 100_000
    per_alpha: float = 0.6
    per_beta: float = 0.4
    checkpoint: str = "./temp/"
    #: Keep the newest ``checkpoint_<iter>.pth.tar`` files. ``0`` or ``None``
    #: disables pruning and keeps every numbered checkpoint.
    checkpoint_top_k: int | None = 3
    max_ply: int | None = None
    profile: bool = False
    profile_dir: str = "./profiles"
    profile_summary_json: str = "iter_timings.json"
    profile_torch_steps: int = 0
    profile_torch_iter: int = 1
    profile_export_chrome: bool = True
    profile_tensorboard_logdir: str | None = None
    profile_with_stack: bool = False

    # Stockfish benchmark (set to 0 to disable). 50 ≈ rare enough vs iteration cost on long runs.
    stockfish_eval_every: int = 50
    stockfish_eval_games: int = 20
    stockfish_elo: int = 1000
    stockfish_skill_level: int = 10
    stockfish_depth: int = 10
    stockfish_think_time: int = 30
    stockfish_path: str | None = None
    stockfish_eval_max_ply: int | None = None


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
    lr_min: float = 1e-5
    lr_warmup_steps: int = 1_000
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_channels: int = 64
    # Chess reward/value targets are bounded to {-1, 0, 1}; three bins are exact.
    support_size: int = 1
    repr_blocks: int = 4
    dyn_blocks: int = 2
    proj_dim: int = 256
    mixed_precision: bool = True
    # CUDA autocast precision. bfloat16 is preferred for its training stability;
    # unsupported devices automatically fall back to float16.
    amp_dtype: str = "bfloat16"
    unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.25
    reward_loss_weight: float = 1.0
    consistency_loss_weight: float = 2.0
    device: str = "cuda"  # "cuda", "mps", or "cpu"
    cuda_device: int | None = None  # Specific CUDA device index (only used if device="cuda")
    compile_inference: bool = False
    compile_training: bool = False
    grad_accum_steps: int = 1
    grad_clip_norm: float = 5.0
    # MuZero scales gradients flowing through each recurrent dynamics step to
    # prevent long unrolls from dominating representation learning.
    recurrent_gradient_scale: float = 0.5
    dataloader_workers: int = 2
    # Search-based value / reanalysis (EZ-V2 Sec. 4.4). Disabled when reanalyze_mcts_sims == 0.
    reanalyze_mcts_sims: int = 0
    reanalyze_prob: float = 0.25
    reanalyze_policy: bool = True
    mixed_value_td_until_step: int = 5000


@dataclass
class TrainCliConfig:
    """Full set of options exposed by ``python main.py`` (tyro).

    Composes :class:`TrainingRunConfig` and :class:`EzV2LearnerConfig` via
    nested dataclasses rather than duplicating their fields.
    """

    seed: int = 0
    log_level: str = "INFO"
    load_model: bool = False
    load_checkpoint_dir: str = "./temp/"
    load_checkpoint_file: str = "latest.pth.tar"
    wandb_project: str | None = None  # Optional WandB project name for experiment tracking
    run: TrainingRunConfig = field(default_factory=TrainingRunConfig)
    learner: EzV2LearnerConfig = field(default_factory=EzV2LearnerConfig)

    def to_training_run(self) -> TrainingRunConfig:
        return TrainingRunConfig(**{f.name: getattr(self.run, f.name) for f in fields(self.run)})

    def to_learner_config(self) -> EzV2LearnerConfig:
        return EzV2LearnerConfig(**{f.name: getattr(self.learner, f.name) for f in fields(self.learner)})
