# ChessRL: EfficientZeroV2 Chess Engine

ChessRL trains a chess engine from self-play using a **MuZero / EfficientZero-style** latent model:

- **Representation** maps a board observation to a hidden state (per-sample mean/std normalisation in the latent).
- **Dynamics** predicts the next hidden state and **scalar reward** (categorical support) given **spatial action planes** (from/to squares).
- **Prediction** outputs **policy** and **value** (categorical support) from a hidden state.
- **Latent MCTS** (PUCT) plans in imagination using `dynamics` + `prediction`; the root uses a real-board observation and legal-move masking. **Batched self-play** groups games by the same visit-count temperature so greedy and exploratory lines do not share a single conservative `temp=1` batch.
- **Unrolled training** on replay with policy / value / reward losses, plus a **SimSiam consistency** loss (cosine similarity with projection/prediction heads) between the rolled latent and `representation(real observation)` at each step. **Policy logits are masked with stored legal moves** at the root and at every unroll step (illegal actions are suppressed like initial inference).
- **Search-based value (reanalysis)** (optional, EfficientZero V2–style): with some probability after a warmup step count, batch preparation can re-run short latent MCTS from replayed board positions and override n-step value bootstraps (and optionally policy targets) so stale `root_values` from data collection are not the only bootstrap signal.
- **Prioritized replay** stores `(trajectory, time index)` transitions for sampling.

No handcrafted chess heuristics are required beyond game rules.

## Architecture details

| Component | Description |
|-----------|-------------|
| `EZV2Networks` | `representation`, `dynamics`, `prediction` + SimSiam projector; categorical value/reward via soft cross-entropy on support bins |
| `DynamicsNetwork` | Spatial action encoding: 2-channel (from_square, to_square) planes concatenated with latent, instead of a dense Linear embedding |
| `SimSiamProjector` | Projection + prediction MLP heads for consistency loss following EfficientZero Eq. 4 |
| `MCTS._latent_simulate` | PUCT selection; expansion with `recurrent_predict`; Q-backup `q = r + γv` with negation for the opposing player |
| `targets.compute_target_value` | n-step return with alternating signs for two-player outcomes; bootstraps from stored MCTS root values unless **root-value overrides** come from reanalysis |
| `network.train_ezv2` | K-step unroll, PER importance weights, scalar-to-support targets, SimSiam cosine consistency, cosine LR schedule; **one optimizer step per training step** with gradient accumulation; optional async batch prefetch (disabled while reanalysis is active so MCTS sees up-to-date weights) |
| `Coach` | Aligns `learner.discount` to `run.discount`; passes `discount` and `mcts_for_reanalyze` into training so MCTS and TD targets agree |

### Action encoding

Actions use a `from_square * 64 + to_square` encoding (4096 entries). Promotions share the same base index; `get_next_state` auto-detects queen promotion and tries all promotion types for pawn-to-back-rank moves.

### Defaults (4 GB VRAM friendly)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_channels` | 64 | ~10M total params |
| `batch_size` | 32 | Fits 4 GB VRAM with mixed precision |
| `lr` | 2e-4 | With cosine annealing to 1e-5 |
| `repr_blocks` | 4 | Residual blocks in representation |
| `dyn_blocks` | 2 | Residual blocks in dynamics |
| `proj_dim` | 256 | SimSiam projection dimension |

Scale up with `--num-channels 128 --batch-size 64` on 24 GB VRAM.

## Quick Start

Run commands from the **repository root** so paths like `./temp/` and `src/index.html` (for the web app) resolve correctly.

```bash
uv sync
make train
```

- `uv sync` installs the `luna` package (editable) and dependencies from `pyproject.toml`.
- `make train` runs `uv run python src/main.py`, which runs self-play, replay writes, gradient updates, and arena evaluation in a loop.

### Training readiness

| Requirement | Notes |
|-------------|--------|
| Install | `uv sync` (or `uv sync --extra dev` for tests/lint). |
| Entry point | `src/main.py` imports the installed `luna` package; do not rely on ad-hoc `PYTHONPATH` if you use `uv run`. |
| Checkpoints | `./temp/` is created when saving; optional warm-start: set `--load-model --load-checkpoint-dir ./temp/`. |
| Hardware | **CUDA GPU required** for training (PyTorch raises if CUDA is missing). 4 GB VRAM works with defaults; 24 GB allows `--num-channels 128 --batch-size 64`. |

If `uv run python -c "import torch; print(torch.cuda.is_available())"` prints `False` but the NVIDIA driver works, try **`make torch-fix`** (reinstalls a CUDA-enabled `torch` in `.venv`). Then run **`make profile-smoke`** for one training iteration with wall-clock phase breakdown and a Kineto trace under `./profiles/` (open with TensorBoard’s PyTorch Profiler tab or `chrome://tracing`). With `--run.profile`, the log also prints **self-play detail**: time in the chess/env loop vs MCTS, plus a breakdown of `search_batch` (encode, initial vs recurrent GPU calls, PUCT selection, expand/backup, finalize). The same numbers are stored in `iter_timings.json`.

#### `ImportError: libcudnn.so.9: cannot open shared object file`

PyTorch’s Linux wheels load cuDNN from the `nvidia-cudnn-cu12` package inside the same venv as `torch`. That error almost always means the shared libraries are missing or not on the dynamic linker path.

1. **Check that the `.so` files exist** (paths use your venv’s `site-packages`):

   ```bash
   ls .venv/lib/python3.12/site-packages/nvidia/cudnn/lib/*.so*
   ```

   If you only see empty directories or no `libcudnn.so.9`, the wheel did not install correctly.

2. **Reinstall the cuDNN wheel** named in `uv pip show torch` under *Requires* (usually `nvidia-cudnn-cu12`; CUDA 13 builds use `nvidia-cudnn-cu13`):

   ```bash
   uv pip install --force-reinstall nvidia-cudnn-cu12
   ```

   Use `--no-cache` if a previous download was corrupted.

3. **WSL + repo on `/mnt/d/...` (NTFS)**  
   Large binary wheels occasionally end up incomplete on DrvFS. If reinstalling does not help, use a clone on the Linux filesystem (e.g. under `$HOME`) or run `uv sync` there.

4. **Quick import check**

   ```bash
   uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

The loop starts with `Starting Iter #1 ...` and a **Self Play** tqdm bar once imports and `LunaNetwork` construction finish.

## Self-Play + Training Workflow

1. Generate self-play games with latent MCTS (`num_episodes` per iteration).
2. Store trajectory positions in prioritized replay.
3. Train the network from replay (`train_steps_per_iter` updates).
4. Arena-evaluate new vs previous model.
5. Save checkpoints to `./temp/` (`temp.pth.tar`, `checkpoint_<iter>.pth.tar`, `best.pth.tar`).

Tune behavior by editing CLI args or `src/main.py`.

### Laptop / fast bootstrap

Self-play time grows roughly with **plies per game** \(\times\) **(1 + `num_mcts_sims`)** network evaluations. On a single GPU, cut MCTS cost, cap pathological game length, use a smaller net, shorten arena, and optionally enable compiled inference (PyTorch 2.x).

Example (faster iterations; raise `--run.num-mcts-sims` again when experiments look sensible):

```bash
uv run python src/main.py \
  --run.num-mcts-sims 12 \
  --run.max-ply 120 \
  --run.arena-compare 8 \
  --learner.num-channels 32 \
  --learner.repr-blocks 2 \
  --learner.dyn-blocks 1 \
  --learner.proj-dim 64 \
  --learner.compile-inference
```

Omit `--learner.compile-inference` if you hit `torch.compile` issues on your stack (older drivers / WSL quirks). **GPUs with CUDA capability below 7.0** (e.g. Quadro P620, GTX 10xx Pascal) cannot use Inductor/Triton; the trainer **skips** compile automatically and logs a warning. `--run.max-ply` forces a draw-shaped terminal reward when the cap is reached (same order as natural draws in `get_game_ended`).

### GPU/CPU utilization and profiling

Self-play keeps up to `--run.parallel-games` episodes in a **sliding pool** so recurrent MCTS inference stays batched even when some games finish early. Raise this until you approach VRAM limits. Arena evaluation batches up to `--run.arena-parallel-games` games per ply (same idea). Use `--run.arena-num-mcts-sims N` for cheaper evaluation than self-play (default: same as `--run.num-mcts-sims`).

**Speed without retraining:** lower `--run.num-mcts-sims`, `--run.max-ply`, and `--run.arena-compare`; raise `--run.parallel-games` until VRAM-bound. **`--run.recurrent-policy-topk`** (default `512`) limits GPU→CPU policy transfer after each recurrent forward (renormalized top-K); use `None` for the full action vector (~4k floats per batch row) if you need exact expansion. **`uv sync --extra perf`** installs **Numba** for faster PUCT when nodes have many children.

### Search-based value / reanalysis (learner flags)

These options approximate EfficientZero V2’s mitigation of **off-policy stale bootstraps** (paper Sec. 4.4). They are **off by default** (`reanalyze_mcts_sims=0`).

| Flag | Default | Role |
|------|---------|------|
| `reanalyze_mcts_sims` | `0` | If `>0`, eligible samples may run this many MCTS simulations on the **current** network to refresh value (and optionally policy) targets. |
| `reanalyze_prob` | `0.25` | Per-sample probability of using reanalysis once past the warmup step (see below). Set `0` to disable even if sims > 0. |
| `reanalyze_policy` | `false` | If `true`, also replace stored MCTS policy targets with the reanalyzed visit distribution (more compute). |
| `mixed_value_td_until_step` | `5000` | Before this **global** training step index, use classic TD targets only (no reanalysis), so early training stays cheap and stable. |

Example (light reanalysis after warmup):

```bash
uv run python src/main.py \
  --learner.reanalyze-mcts-sims 16 \
  --learner.reanalyze-prob 0.2 \
  --learner.mixed-value-td-until-step 3000
```

When reanalysis is enabled with positive probability, **batch prefetch runs on the training thread** (no background overlap) so MCTS always uses the latest weights.

To see wall time per phase each iteration and optional PyTorch Kineto traces:

```bash
uv run python src/main.py --run.profile \
  --run.profile-torch-steps 8 \
  --run.profile-torch-iter 1 \
  --run.profile-dir ./profiles
```

Traces land under `--run.profile-dir` (Chrome trace and/or TensorBoard logdir via `--run.profile-tensorboard-logdir`). Aggregated timings are written to `--run.profile-dir` / `--run.profile-summary-json` (default `iter_timings.json`).

## Main Training Parameters

Configured via `python src/main.py --help`:

| Parameter | Description |
|-----------|-------------|
| `num_iters` | Number of train/eval iterations |
| `num_episodes` | Self-play episodes per iteration |
| `num_mcts_sims` | MCTS simulations per move |
| `cpuct` | PUCT exploration constant |
| `unroll_steps` | Unroll length K for latent training |
| `td_steps` | Bootstrap horizon for value targets |
| `train_steps_per_iter` | Gradient steps per iteration |
| `batch_size` | Replay batch size |
| `replay_capacity` | Max stored `(traj, position)` transitions |
| `per_alpha` / `per_beta` | Prioritized replay prioritization / IS correction |
| `num_channels` | Latent channel width (main capacity knob) |
| `lr` / `lr_min` | Learning rate and cosine annealing floor |
| `checkpoint` | Checkpoint output directory |
| `save_anyway` | Default false: keep new weights only if arena beats `update_threshold`; pass `--run.save-anyway` to always accept |
| `max_ply` | Optional cap on plies per self-play game (draw if exceeded); speeds laptop runs |
| `parallel_games` | Self-play pool size: more games in flight → larger GPU batches (watch VRAM) |
| `recurrent_policy_topk` | Batched MCTS: top-K policy rows from GPU (`None` = full vector) |
| `arena_parallel_games` | How many pit games to run in parallel during arena (per ply) |
| `arena_num_mcts_sims` | Optional; if set, arena MCTS uses fewer sims than self-play |
| `profile` / `profile_torch_steps` | Per-iter phase timings; optional Kineto export for CPU vs CUDA breakdown |
| `compile_inference` (learner) | If true, `torch.compile` MCTS inference (`torch>=2.4`); optional warmup at training start |
| `discount` | **`run.discount`** is the single source used for MCTS and TD targets (copied onto the learner in `Coach` if configs disagree) |
| `reanalyze_*` / `mixed_value_td_until_step` (learner) | Optional search-based targets; see table above |

Training regression coverage includes `tests/test_train_ezv2.py` (optimizer stepping / prefetch behaviour) alongside `tests/test_targets.py` and `tests/test_coach.py`.

## Project Structure

```
src/
├── main.py                    # self-play + training entry point
├── luna/
│   ├── coach.py               # training loop orchestration
│   ├── mcts.py                # latent MCTS
│   ├── network.py             # learner wrapper and optimization
│   ├── ezv2_networks.py       # representation / dynamics / prediction / SimSiam
│   ├── config.py              # typed dataclass configs
│   ├── replay_buffer.py       # prioritized replay (sum-tree)
│   ├── targets.py             # n-step & unroll target construction
│   ├── engine.py              # engine inference interface (`Luna`)
│   ├── utils.py               # utilities
│   └── game/
│       ├── chess_game.py      # chess environment + spatial action encoding
│       ├── arena.py           # head-to-head evaluation
│       ├── player.py          # players (human/engine/random)
│       └── state.py           # board state wrapper
└── web_app.py                 # Flask web interface
```

## Development Commands

```bash
uv sync --extra dev
# optional: Numba for faster PUCT in MCTS (`uv sync --extra perf`)
make fmt
make lint
make types
make check
make test
make bench
make serve
```

`make bench` runs `tests/bench_throughput.py`; pass laptop-style flags to match training (for example `--max-ply 80 --mcts-sims 8`).

## References

- Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero), 2020.
- Ye et al., *Mastering Atari Games with Limited Data* (EfficientZero), 2021.
- Wang et al., *EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data*, ICML 2024.
- Chen & He, *Exploring Simple Siamese Representation Learning* (SimSiam), 2021.
