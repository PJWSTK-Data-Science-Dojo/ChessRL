# Luna ChessRL

Luna is a model-based reinforcement-learning chess engine trained from self-play. It combines an EfficientZeroV2-style latent world model with Gumbel MuZero search, prioritized replay, search-value reanalysis, and an unrolled consistency objective. The repository includes training, evaluation, an interactive browser experience, a UCI adapter, and a secure Lichess deployment helper.

This is a research and portfolio project, not a claim of engine parity with established tournament systems. Playing strength depends on training data, compute budget, search settings, and evaluation methodology.

## What is implemented

- Representation, dynamics, and prediction networks with per-sample latent normalization.
- Gumbel top-m root selection and Sequential Halving by default; classic PUCT remains available.
- Batched latent MCTS for parallel self-play, exact legal-move masks, and two-player value backups.
- A 4,288-action chess space: 4,096 from/to actions plus distinct knight, rook, and bishop underpromotions.
- Five spatial dynamics action planes: from, to, and one plane for each underpromotion identity.
- A convolutional policy head aligned to source squares, destination squares, and underpromotion identity.
- A 119-plane observation tensor with eight positions of piece/repetition history and complete current rule state.
- K-step unrolled policy, value, reward, and SimSiam-style consistency losses.
- Three-bin categorical value and reward targets for the exact `{-1, 0, 1}` chess outcome range.
- Prioritized trajectory replay with compact `float16` observations/policies and boolean legal masks.
- AdamW, learning-rate warm-up followed by cosine decay, mixed precision, gradient clipping, accumulation, and recurrent gradient scaling.
- Optional batched search-value estimation and policy reanalysis using the current network.
- Versioned, atomic checkpoints that include architecture, optimizer, scaler, step,
  trainer iteration, and the learning-rate schedule horizon.

The implementation follows ideas from [MuZero](https://arxiv.org/abs/1911.08265), [EfficientZero](https://arxiv.org/abs/2111.00210), [EfficientZero V2](https://proceedings.mlr.press/v235/wang24at.html), [Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO), and [SimSiam](https://arxiv.org/abs/2011.10566).

## Quick start

Run every command from the repository root.

```bash
uv sync --extra dev --extra perf
make check
make profile-smoke
make install-fairy-stockfish
```

Start a new training run with the maintained single-accelerator preset:

```bash
command -v stockfish
printf 'uci\nquit\n' | stockfish | rg '^uciok$'
make train CHECKPOINT_DIR=./runs/luna-main
```

The Python adapter uses `python-chess`; engine executables are separate native programs.
Install a current binary from the
[official Stockfish download](https://stockfishchess.org/download/) for the fixed benchmark.
`make install-fairy-stockfish` reproducibly builds the pinned Fairy-Stockfish 14 release
at `vendor/stockfish/fairy-stockfish-14` for the adaptive ladder. Both startup preflights
play a legal smoke-test move and stop the run before self-play if an engine contract is
invalid. For an official Stockfish binary outside `PATH`, pass an absolute path:

```bash
make train CHECKPOINT_DIR=./runs/luna-main \
  ARGS='--run.stockfish-path /absolute/path/to/stockfish'
```

All preset values remain overrideable because `ARGS` is appended last:

```bash
make train CHECKPOINT_DIR=./runs/luna-main \
  ARGS='--run.num-iters 20 --run.num-mcts-sims 24'
```

Resume the most recently published training state:

```bash
make resume CHECKPOINT_DIR=./runs/luna-main
```

`CHECKPOINT_DIR` defaults to `./runs/luna-main`; giving each experiment a named directory keeps
its checkpoints isolated. A fresh run requires a new or empty directory. `make resume`
loads `latest.pth.tar` from the selected directory and uses the same architecture preset
as `make train`. For a custom architecture, pass the same learner flags that created the
checkpoint. The loader intentionally rejects unversioned or incompatible legacy files.
Resuming into a different checkpoint directory is allowed only when that destination is
empty of managed checkpoints and evaluation metadata, preventing two experiment lineages
from being merged accidentally.

Use `uv run python src/main.py --help` for the complete Tyro-generated option reference. A bare `src/main.py` invocation uses the lighter dataclass defaults; `make train` applies the larger maintained experiment preset.

## Strength training phase

To carry the complete model, optimizer, scaler, counters, and LR horizon from the current
`runs/luna-strength-1500-v1/latest.pth.tar` into the separate ladder contract, run this
once, then use `make resume-phase` thereafter:

```bash
make migrate-ladder-phase
```

The target defaults to `runs/luna-fairy-ladder-v1` and must initially be absent or empty.
Replay is process memory and therefore starts empty. If the first migration process is
interrupted after creating its W&B run but before publishing target `latest.pth.tar`, retry
explicitly with `make migrate-ladder-phase ARGS='--wandb-resume allow'`.

`make train-phase` is a different, intentional alternative: it imports only weights from
the pinned, externally evaluated `runs/luna-stockfish16-continuation/best.pth.tar`, then
starts a fresh optimizer, scaler, counters, replay, and learning-rate schedule. It verifies
the source against `NEW_PHASE_SOURCE_SHA256`; override the source and hash together only
when deliberately starting that new lineage.

A local, Git-ignored `.env` must define
`WANDB_API_KEY` and `WANDB_ENTITY` (use `dsc-pjatk-warsaw` for the maintained
experiment). The preflight checks only that both variables are
non-empty and does not print their values.

After a crash or reboot, continue the phase instead of starting it again:

```bash
make resume-phase
```

Resume restores the phase optimizer, scaler, counters, and original learning-rate
schedule. All phase commands pass the same `NEW_PHASE_WANDB_RUN_ID` (default
`luna-fairy-ladder-v1`) through `--wandb-run-id` and the display name
`Luna Fairy Ladder 500+ · Benchmark 1500 v1` through `--wandb-run-name`. `make train-phase` uses
`--wandb-resume never`, so it refuses to append a new phase to an existing remote run;
`make resume-phase` uses `--wandb-resume must`, so a typo or missing remote run fails
instead of silently creating a second dashboard. Change the run ID only when intentionally
starting a different phase. If the remote run was deliberately deleted while its local
checkpoint remains valid, recreate it explicitly with
`make resume-phase ARGS='--wandb-resume allow'`. The general CLI defaults to `allow`.

The maintained phase preset collects 128 self-play episodes with four persistent actors
running up to 32 active games each, trains with batches of 256, and reanalyzes 10% of
eligible samples. Every five iterations it plays a 20-game paired-opening match at the
current Fairy-Stockfish rung. The ladder starts at native `UCI_Elo=500`, advances by 100
after two consecutive matches with more Luna wins than Fairy wins, never demotes, and
stops at 2800. Its atomically persisted state is `fairy_ladder.json`.

Separately, every 25 iterations Luna plays the official Stockfish fixed benchmark at
1500 Elo. Only that immutable benchmark can promote `best.pth.tar`; ladder results never
change the best-checkpoint contract.
The actor pool is controlled by `--run.self-play-workers` and can be overridden through
`ARGS`; set it to `1` for in-process self-play without the pool.

The preset does not keep 1,024 MCTS roots active at once. Larger recurrent-inference
batches improve the GPU kernel less than they increase CPU chess-rule and tree-management
work at that scale; multiple actors keep the accelerator fed more effectively. W&B logs
`selfplay/*` workload data, `performance/*` phase timings and throughput, and `replay/*`
buffer state. Fixed-opponent results use `benchmark/*`; adaptive results use `ladder/*`
with their own monotonic evaluation step, current/highest Elo, decisions, outcomes,
duration, and approximate 95% score interval. The fixed match is a regression sentinel, not a
guarantee of playing strength, and a 20-game sample remains noisy.

## Checkpoint contract

Training writes format-v2 files under `CHECKPOINT_DIR` (`./runs/luna-main` by default):

- `checkpoint_<iteration>.pth.tar` is an immutable numbered snapshot. Retention is controlled by `--run.checkpoint-top-k`.
- `latest.pth.tar` is atomically updated after every completed training iteration. Use it for resume and local testing, not as a public deployment artifact.
- `best.pth.tar` is created or replaced only when a checkpoint improves the fixed Stockfish benchmark score. Its payload embeds the authoritative score, protocol, and evaluated-checkpoint hash, so `best_eval.json` can be repaired after an interrupted write. It is not a synonym for “newest.”
- `benchmark_state.json` records fixed-benchmark completion and makes a scheduled evaluation idempotent across restarts.
- `fairy_ladder.json` atomically stores adaptive-rung progress and is bound to the exact binary hash and evaluation protocol.

Resume restores the optimizer, scaler, global step, trainer iteration, and the original
learning-rate schedule horizon. A resumed invocation cannot silently redefine the
warm-up and cosine schedule by requesting a different training horizon.
If power is lost after a scheduled checkpoint but during evaluation, startup reconciles
that exact checkpoint before continuing. Missing sidecars after established progress
fail closed instead of silently resetting the benchmark or ladder.

Checkpoints and evaluation metadata are runtime artifacts and are ignored by Git. No pretrained weights are bundled.
Public serving uses a read-only, versioned copy of an evaluated checkpoint plus its SHA-256 digest; `make release-web-model RELEASE_ID=<id>` creates that release without overwriting an existing ID.

## Play in the browser

The rebuilt interface supports private browser-session games, random color, three search profiles, hints, undo, move history, captured pieces, evaluation feedback, and an automated observatory mode.

```bash
make serve
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). CPU and Apple Silicon smoke modes are available as `make serve-cpu` and `make serve-mps`.

The JSON API is namespaced under `/api/v1`:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Model readiness, checkpoint, and search profiles |
| `POST` | `/api/v1/games` | Create a human or self-play game |
| `GET` | `/api/v1/games/<id>` | Read session-owned state |
| `DELETE` | `/api/v1/games/<id>` | Delete a game |
| `POST` | `/api/v1/games/<id>/moves` | Submit a UCI move and receive Luna's reply |
| `POST` | `/api/v1/games/<id>/engine-move` | Advance observatory mode |
| `POST` | `/api/v1/games/<id>/hint` | Analyze the human position |
| `POST` | `/api/v1/games/<id>/undo` | Rewind one completed human turn |

See [DEPLOYMENT.md](DEPLOYMENT.md) for immutable model releases, the hardened uv-based container, and accelerator hosting through an outbound Cloudflare named tunnel. Public mode publishes no host port and requires no inbound firewall rule.
The deployment topology is also available as an [editable Excalidraw diagram](docs/public-serving.excalidraw).

## UCI and Lichess

Start the UCI adapter:

```bash
make uci ARGS='--device cuda --mcts-sims 96 --compile-inference'
```

The adapter supports the standard handshake, positions from FEN or move lists, clock-aware and interruptible `go`, restricted `go searchmoves`, and runtime search options. Diagnostics go to stderr so stdout remains protocol-safe.

For a challenge-driven Lichess Bot API setup, including token-safe configuration generation, follow [LICHESS.md](LICHESS.md).
After preparing the upstream bridge, `make lichess-config` writes a credential-free configuration. Export `LICHESS_BOT_TOKEN` only when starting the bridge.

The standalone Lichess engine and web service must not run independently on one accelerator. Each process loads its own model and owns only a process-local inference lock; use separate serving capacity until a shared inference owner is implemented.

## Evaluate

Run an external benchmark without changing checkpoint promotion state:

```bash
uv run python src/eval_vs_stockfish.py \
  --checkpoint ./runs/luna-main/latest.pth.tar \
  --run.stockfish-eval-games 20 \
  --run.stockfish-depth 10
```

Set `--run.stockfish-path` when the executable is not discoverable. The fixed benchmark
defaults to 1500 Elo; keep Elo, depth, engine build, and search
settings fixed when comparing checkpoints. Periodic training evaluation is
controlled by `--run.stockfish-eval-every`; set it to `0` to disable. Each versioned
opening is played with both color assignments. A small match is still a noisy estimate.

## Training flow

1. A sliding pool of games produces batched self-play through latent search.
2. Each position stores observation, action, acting-player reward, improved search policy, root value, and legal mask.
3. Prioritized replay samples positions and constructs K-step unroll targets with alternating player signs.
4. Optional reanalysis refreshes selected value and policy targets with the current model.
5. The learner performs one optimizer update per training step, using accumulation only to form that update.
6. Numbered and `latest` checkpoints are saved atomically; optional external evaluation may promote `best`.

Chess uses an undiscounted terminal objective (`discount=1.0`). Terminal outcomes are represented explicitly as `None` while ongoing and `-1`, `0`, or `1` when complete. Illegal model actions fail fast instead of being replaced silently.

## Useful commands

```bash
make fmt                 # format and apply safe Ruff fixes
make format-check        # verify Ruff formatting without editing files
make lint                # Ruff checks
make types               # mypy over src
make test                # pytest suite
make check               # format check + lint + types + tests
make audit               # audit locked runtime dependencies (network required)
make bench               # throughput benchmark
make profile-smoke       # bounded end-to-end profile
make train-phase         # start the dedicated strength continuation phase
make resume-phase        # resume that phase after an interruption
make test-pipeline-cpu   # short CPU training smoke test
make test-pipeline-mps   # short MPS training smoke test
make release-web-model RELEASE_ID=<id>  # immutable evaluated web artifact
make web-config          # validate private Compose configuration
make web-up              # start loopback-only container deployment
make web-public-config   # validate GPU + named-tunnel deployment
make web-public-up       # start public named-tunnel deployment
```

If CUDA is requested but unavailable, verify the environment first:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Install the PyTorch build appropriate for the host using the official PyTorch instructions when the environment is incorrect.

## Repository map

```text
src/main.py                    training entry point
src/web_app.py                 Flask app factory and interactive API
src/eval_vs_stockfish.py       standalone external evaluation
src/luna/config.py             typed configuration
src/luna/coach.py              self-play/training/checkpoint orchestration
src/luna/network.py            learner, inference, and checkpoint I/O
src/luna/ezv2_networks.py      neural architecture and support transforms
src/luna/mcts.py               Gumbel MuZero and PUCT latent search
src/luna/replay_buffer.py      prioritized trajectory replay
src/luna/targets.py            TD and unroll targets
src/luna/uci.py                UCI adapter
src/luna/lichess_config.py     secure lichess-bot config generator
src/luna/game/                 chess rules, arena, players, Stockfish eval
tests/                         unit, integration, protocol, and regression tests
```

## Reproducibility notes

- `--seed` seeds Python, NumPy, and PyTorch. Some accelerator kernels may remain nondeterministic.
- Training checkpoints restore optimizer, scaler, global step, and trainer iteration. The in-memory replay buffer is not serialized.
- `torch.compile` is optional. Disable it with `--learner.no-compile-inference` or the corresponding web flag when unsupported.
- Measure throughput with the included benchmark and profiler before changing batch, parallel-game, or search budgets.
