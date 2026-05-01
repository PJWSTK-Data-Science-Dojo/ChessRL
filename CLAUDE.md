# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChessRL is an **EfficientZeroV2-based chess engine** that learns from self-play without handcrafted heuristics. It implements MuZero/EfficientZero-style latent model-based RL with:

- **Representation** network: board observation → hidden state (per-sample normalized)
- **Dynamics** network: predicts next hidden state + scalar reward from spatial action planes
- **Prediction** network: outputs policy + value from hidden state
- **Latent MCTS** (PUCT): plans in imagination using dynamics + prediction
- **Unrolled training**: K-step unroll with policy/value/reward losses + SimSiam consistency loss
- **Search-based value (reanalysis)**: optional EZV2-style fresh MCTS on current network to fix stale bootstraps
- **Prioritized replay**: stores (trajectory, time index) transitions

## Development Commands

**Always run commands from the repository root** so paths like `./temp/` and `src/index.html` resolve correctly.

### Setup
```bash
uv sync                          # Install dependencies from pyproject.toml
uv sync --extra dev              # Install dev tools (ruff, mypy, pytest)
uv sync --extra perf             # Install Numba for faster PUCT in MCTS
```

### Training
```bash
make train                       # Run full training loop (src/main.py)
make train ARGS='--run.num-mcts-sims 20'  # Pass custom CLI args

# RTX 4090 optimized (24GB VRAM): ~500 iters in 2 days, ~5 min/iter
make train-rtx4090-balanced      # Balanced config with reanalysis enabled
```

Entry point is `src/main.py`, which orchestrates self-play → replay → training → arena evaluation in a loop.

### MacBook / CPU Testing
```bash
# Test training pipeline on CPU (3 iterations, small model)
make test-pipeline-macbook

# Same but with MPS backend (Apple Silicon Macs)
make test-pipeline-macbook-mps
```

### Development
```bash
make fmt                         # Format code with ruff
make lint                        # Lint with ruff
make types                       # Type-check with mypy
make check                       # Run lint + types
make test                        # Run pytest suite
make bench                       # Benchmark throughput (tests/bench_throughput.py)
make serve                       # Launch Flask web app (src/web_app.py)
```

### Profiling
```bash
make profile-smoke               # One iter with phase timings + Kineto trace in ./profiles/
```

Use `--run.profile` for per-iteration phase breakdown and optional PyTorch Kineto traces. Traces land in `--run.profile-dir` (default `./profiles/`) with aggregated timings in `iter_timings.json`.

### GPU/CUDA Issues
```bash
uv run python -c "import torch; print(torch.cuda.is_available())"  # Check CUDA
make torch-fix                   # Reinstall CUDA-enabled torch if needed
```

If you hit `ImportError: libcudnn.so.9: cannot open shared object file`:
1. Check `.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/*.so*` exists
2. Run `uv pip install --force-reinstall nvidia-cudnn-cu12`
3. On WSL with repo on `/mnt/d/...` (NTFS), clone to Linux filesystem instead

## Architecture

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| `Coach` | `src/luna/coach.py` | Training loop orchestration: self-play → replay → training → arena |
| `LunaNetwork` | `src/luna/network.py` | Learner wrapper: unroll training, gradient steps, checkpoint I/O |
| `EZV2Networks` | `src/luna/ezv2_networks.py` | Representation/dynamics/prediction + SimSiam projector |
| `MCTS` / `BatchedMCTS` | `src/luna/mcts.py` | Latent MCTS: single-game and batched parallel search |
| `PrioritizedReplayBuffer` | `src/luna/replay_buffer.py` | Sum-tree PER for (trajectory, position) transitions |
| `targets.py` | `src/luna/targets.py` | n-step TD target construction with alternating signs for two-player |
| `ChessGame` | `src/luna/game/chess_game.py` | Chess environment + spatial action encoding (from_square, to_square) |
| `Arena` | `src/luna/game/arena.py` | Head-to-head evaluation (new vs previous model) |
| `config.py` | `src/luna/config.py` | Typed dataclass configs (CLI args via tyro) |

### Package Structure

The `luna` package is installed editable from `src/luna/` (via `uv sync`). Do not rely on `PYTHONPATH` hacks; use `uv run` to execute with the installed package.

### Action Encoding

Actions use `from_square * 64 + to_square` (4096 entries). Promotions share the same base index; `get_next_state` auto-detects queen promotion and tries all promotion types for pawn-to-back-rank moves.

`DynamicsNetwork` uses **spatial action encoding**: 2-channel (from_square, to_square) planes concatenated with latent, instead of dense Linear embedding.

### Training Flow

1. **Self-play**: Generate episodes with batched latent MCTS (`num_episodes` per iteration)
2. **Replay**: Store trajectory positions in prioritized replay buffer
3. **Training**: K-step unroll from replay (`train_steps_per_iter` gradient steps)
4. **Arena**: Evaluate new vs previous model
5. **Checkpoints**: Save to `./temp/` (`temp.pth.tar`, `checkpoint_<iter>.pth.tar`, `best.pth.tar`)

### Key Training Details

- **Discount**: `run.discount` is the single source of truth; `Coach` copies it to learner config if they disagree (warning logged)
- **Reanalysis**: Optional search-based value (EZV2 V2 Sec 4.4) via `--learner.reanalyze-mcts-sims N` (off by default). Refreshes stale bootstraps by re-running MCTS on current network. When enabled, batch prefetch runs on training thread (no background overlap) so MCTS uses latest weights.
- **Batch prefetch**: Async by default, disabled when reanalysis is active
- **Optimizer**: One step per training step with gradient accumulation
- **Loss**: Policy + value + reward (categorical support via soft cross-entropy) + SimSiam consistency (cosine similarity between unrolled latent and `representation(real observation)`)
- **Legal move masking**: Policy logits masked at root and every unroll step (illegal actions suppressed)

### MCTS Implementation

- **Batched self-play**: `BatchedMCTS` maintains a sliding pool of up to `parallel_games` episodes, refilling finished games so recurrent inference stays batched
- **PUCT**: `cpuct * prior * sqrt(total_visits) / (1 + visits)` exploration bonus
- **Latent simulation**: `_latent_simulate` expands nodes with `recurrent_predict`, backs up `q = r + γv` with negation for opposing player
- **Top-K policy transfer**: `--run.recurrent-policy-topk` (default 512) limits GPU→CPU policy rows after recurrent forward (renormalized). Use `None` for full action vector if you need exact expansion.
- **Numba acceleration**: Install `uv sync --extra perf` for faster PUCT when nodes have many children

## Common Workflows

### Quick Laptop/Fast Bootstrap

Self-play time grows with plies × (1 + `num_mcts_sims`) network evals. To speed up:

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

Omit `--learner.compile-inference` on older drivers or if `torch.compile` fails. GPUs with CUDA capability < 7.0 (Pascal) cannot use Inductor/Triton; trainer skips compile automatically with a warning.

### Scaling Up (24 GB VRAM)

```bash
uv run python src/main.py \
  --learner.num-channels 128 \
  --learner.batch-size 64 \
  --run.batch-size 64
```

### Enabling Reanalysis (Search-Based Value)

```bash
uv run python src/main.py \
  --learner.reanalyze-mcts-sims 16 \
  --learner.reanalyze-prob 0.2 \
  --learner.mixed-value-td-until-step 3000
```

### Checkpointing

- Warm-start: `--load-model --load-checkpoint-dir ./temp/`
- Checkpoints saved to `./temp/` by default
- `--run.save-anyway`: always save new weights (default: only if arena beats `update_threshold`)

## Important CLI Flags

All flags via `python src/main.py --help`. Key parameters:

| Flag | Default | Notes |
|------|---------|-------|
| `--run.num-iters` | 1000 | Training iterations |
| `--run.num-episodes` | 100 | Self-play episodes per iter |
| `--run.num-mcts-sims` | 50 | MCTS simulations per move |
| `--run.cpuct` | 1.25 | PUCT exploration constant |
| `--run.discount` | 0.997 | Single source for MCTS + TD targets |
| `--run.unroll-steps` | 5 | K-step unroll length |
| `--run.td-steps` | 5 | n-step bootstrap horizon |
| `--run.train-steps-per-iter` | 100 | Gradient steps per iteration |
| `--run.batch-size` | 32 | Replay batch size |
| `--run.replay-capacity` | 100000 | Max (traj, position) transitions |
| `--run.parallel-games` | 8 | Self-play pool size (more → larger GPU batches) |
| `--run.arena-parallel-games` | 4 | Pit games in parallel per ply |
| `--run.max-ply` | None | Optional cap (draw if exceeded) for faster laptop runs |
| `--run.recurrent-policy-topk` | 512 | Limit GPU→CPU policy transfer (None = full vector) |
| `--learner.num-channels` | 64 | Latent channel width (~10M params) |
| `--learner.lr` | 2e-4 | Learning rate (cosine anneal to `lr_min`) |
| `--learner.compile-inference` | False | `torch.compile` MCTS inference (PyTorch 2.x, Volta+) |
| `--learner.reanalyze-mcts-sims` | 0 | Search-based value (off by default) |
| `--learner.reanalyze-prob` | 0.25 | Per-sample probability of reanalysis after warmup |
| `--learner.mixed-value-td-until-step` | 5000 | Use classic TD only before this global step |

## Testing

```bash
uv run --extra dev pytest tests/ -v
```

Test coverage:
- `test_ezv2_networks.py`: representation/dynamics/prediction forward passes
- `test_mcts.py`: latent MCTS logic
- `test_replay_buffer.py`: PER sum-tree
- `test_targets.py`: n-step TD target construction
- `test_train_ezv2.py`: optimizer stepping, prefetch behavior
- `test_coach.py`: discount alignment
- `test_game.py`: chess environment

## Playing Against the Model

### Web Interface
```bash
make serve       # Launch on CUDA (default)
make serve-cpu   # Launch on CPU (MacBook compatible)
make serve-mps   # Launch on MPS (Apple Silicon)
```

Serves `src/index.html` with chessboard UI at http://127.0.0.1:5000

### Command-Line Interface
```bash
# Play via CLI (supports CPU/MPS for MacBook)
uv run python src/play_vs_model.py \
  --checkpoint ./temp/best.pth.tar \
  --device cpu \
  --mcts-sims 25

# Options:
#   --device: cuda, mps, or cpu
#   --mcts-sims: lower = faster response (25-50 recommended for CPU)
#   --human-plays-white: True (default) or False
```

## Key Implementation Notes

- **Device support**: CUDA (GPU), MPS (Apple Silicon), or CPU via `--learner.device`
- **Legal move masking in latent space**: MCTS tracks board positions and masks illegal moves during latent expansion
- **Categorical value/reward**: soft cross-entropy on support bins (not MSE)
- **SimSiam consistency**: cosine similarity between unrolled latent and `representation(real board)` at each step
- **Per-sample normalization**: latent states normalized per-sample (mean/std) before dynamics/prediction
- **Two-player alternating signs**: n-step returns and Q-backups negate for opposing player
- **Legal move masking at root**: stored legal moves mask policy logits at root and every unroll step
- **One optimizer step per training step** with gradient accumulation (not multiple microbatches per step)
- **Warmup for compile**: if `--learner.compile-inference`, trainer warms up compiled MCTS inference at start
- **Defensive illegal move handling**: `get_next_state()` falls back to random legal move if action is illegal

## References

Implementation based on:
- Schrittwieser et al., *MuZero* (2020)
- Ye et al., *EfficientZero* (2021)
- Wang et al., *EfficientZero V2* (ICML 2024)
- Chen & He, *SimSiam* (2021)

Paper included at `papers/ezv2.pdf`.
