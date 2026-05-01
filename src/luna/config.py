"""Typed configuration for MCTS, training loops, EZV2 learner, and the training CLI."""

from dataclasses import dataclass, field, fields


@dataclass
class MCTSParams:
    """Parameters for latent PUCT search.

    ``recurrent_policy_topk``: for batched MCTS only, transfer only the top-K log-prob
    actions from GPU after each recurrent forward (renormalized). ``None`` copies the
    full policy (~4k floats per batch row). K≥512 covers typical chess mass; set
    ``None`` for exact full-policy expansion.

    CLI speed knobs (no code changes): raise ``parallel_games`` until VRAM-bound;
    lower ``num_mcts_sims`` / ``max_ply`` for wall time; use ``arena_num_mcts_sims`` and
    lower ``arena_compare`` to shorten evaluation.
    """

    # Number of MCTS simulations per move during self-play
    # AlphaZero used 800, but 50-200 is practical for training runs
    # Higher = stronger tactical play but slower iteration time
    num_mcts_sims: int = 50

    # PUCT exploration constant (Silver et al. 2016, AlphaZero)
    # Controls exploration/exploitation tradeoff in tree search
    # 1.25 is empirically tuned for chess; typical range [1.0, 2.5]
    cpuct: float = 1.25

    dir_noise: bool = True

    # Dirichlet noise alpha for root exploration (AlphaZero)
    # Lower values = more concentrated noise
    # 0.3 is appropriate for chess (~35 legal moves average)
    # Formula from paper: alpha = 10/n where n is typical branching factor
    dir_alpha: float = 0.3

    # Discount factor for n-step TD returns
    # 0.997^40 ≈ 0.88 (minimal discounting over typical 40-move game)
    # Chess is deterministic, so high gamma preserves long-term planning
    # Consider 0.95 for shorter horizon if games become too long
    discount: float = 0.997

    recurrent_policy_topk: int | None = 512


@dataclass
class TrainingRunConfig(MCTSParams):
    """Self-play schedule, replay, arena evaluation, and checkpoint paths.

    Inherits MCTS fields from :class:`MCTSParams`.

    Arena evaluation uses ``arena_num_mcts_sims`` (default: same as self-play) and
    batches up to ``arena_parallel_games`` pit games per GPU step, like self-play.

    ``save_anyway``: if False (default), a new checkpoint is kept only when
    ``nwins / (nwins + pwins) >= update_threshold``; otherwise weights reload from
    ``temp.pth.tar``. Set True for exploratory runs or when arena is mostly draws.
    """

    num_iters: int = 20
    num_episodes: int = 20
    parallel_games: int = 8
    temp_threshold: int = 15
    update_threshold: float = 0.55
    arena_compare: int = 20
    arena_num_mcts_sims: int | None = None
    arena_parallel_games: int = 8
    batch_size: int = 32
    train_steps_per_iter: int = 200
    replay_capacity: int = 100_000
    per_alpha: float = 0.6
    per_beta: float = 0.4
    checkpoint: str = "./temp/"
    save_anyway: bool = False
    max_ply: int | None = None
    profile: bool = False
    profile_dir: str = "./profiles"
    profile_summary_json: str = "iter_timings.json"
    profile_torch_steps: int = 0
    profile_torch_iter: int = 1
    profile_export_chrome: bool = True
    profile_tensorboard_logdir: str | None = None
    profile_with_stack: bool = False


@dataclass
class EzV2LearnerConfig:
    """Optimizer, architecture, unroll training, and loss weights for :class:`LunaNetwork`."""

    lr: float = 2e-4
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_channels: int = 64
    support_size: int = 10
    repr_blocks: int = 4
    dyn_blocks: int = 2
    proj_dim: int = 256
    mixed_precision: bool = True
    unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 0.997
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.25
    reward_loss_weight: float = 1.0
    consistency_loss_weight: float = 2.0
    device: str = "cuda"  # "cuda", "mps", or "cpu"
    cuda_device: int | None = None  # Specific CUDA device index (only used if device="cuda")
    compile_inference: bool = False
    compile_training: bool = False
    grad_accum_steps: int = 1
    dataloader_workers: int = 2
    # Search-based value / reanalysis (EZ-V2 Sec. 4.4). Disabled when reanalyze_mcts_sims == 0.
    reanalyze_mcts_sims: int = 0
    reanalyze_prob: float = 0.25
    reanalyze_policy: bool = False
    mixed_value_td_until_step: int = 5000


@dataclass
class TrainCliConfig:
    """Full set of options exposed by ``python main.py`` (tyro).

    Composes :class:`TrainingRunConfig` and :class:`EzV2LearnerConfig` via
    nested dataclasses rather than duplicating their fields.
    """

    log_level: str = "INFO"
    load_model: bool = False
    load_checkpoint_dir: str = "./pretrained_models/"
    load_checkpoint_file: str = "best.pth.tar"
    wandb_project: str | None = None  # Optional WandB project name for experiment tracking
    run: TrainingRunConfig = field(default_factory=TrainingRunConfig)
    learner: EzV2LearnerConfig = field(default_factory=EzV2LearnerConfig)

    def to_training_run(self) -> TrainingRunConfig:
        return TrainingRunConfig(**{f.name: getattr(self.run, f.name) for f in fields(self.run)})

    def to_learner_config(self) -> EzV2LearnerConfig:
        return EzV2LearnerConfig(**{f.name: getattr(self.learner, f.name) for f in fields(self.learner)})
