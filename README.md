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
- Versioned, atomic checkpoints that include architecture, optimizer, scaler, step, and trainer-iteration metadata.

The implementation follows ideas from [MuZero](https://arxiv.org/abs/1911.08265), [EfficientZero](https://arxiv.org/abs/2111.00210), [EfficientZero V2](https://proceedings.mlr.press/v235/wang24at.html), [Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO), and [SimSiam](https://arxiv.org/abs/2011.10566).

## Quick start

Run every command from the repository root.

```bash
uv sync --extra dev --extra perf
make check
make profile-smoke
```

Start a new training run with the maintained single-accelerator preset:

```bash
command -v stockfish
printf 'uci\nquit\n' | stockfish | rg '^uciok$'
make train
```

The locked `stockfish` Python package is an engine controller, not the Stockfish
executable. Install a current binary from the
[official Stockfish download](https://stockfishchess.org/download/) before using the
maintained preset. Its startup preflight deliberately stops the run if the scheduled
external benchmark cannot launch. For a binary outside `PATH`, pass an absolute path:

```bash
make train ARGS='--run.stockfish-path /absolute/path/to/stockfish'
```

All preset values remain overrideable because `ARGS` is appended last:

```bash
make train ARGS='--run.num-iters 20 --run.num-mcts-sims 24'
```

Resume the most recently published training state:

```bash
make resume
```

`make resume` uses `./temp/latest.pth.tar` and the same architecture preset as `make train`. For a custom architecture, pass the same learner flags that created the checkpoint. The loader intentionally rejects unversioned or incompatible legacy files.

Use `uv run python src/main.py --help` for the complete Tyro-generated option reference. A bare `src/main.py` invocation uses the lighter dataclass defaults; `make train` applies the larger maintained experiment preset.

## Checkpoint contract

Training writes format-v2 files under `./temp/`:

- `checkpoint_<iteration>.pth.tar` is an immutable numbered snapshot. Retention is controlled by `--run.checkpoint-top-k`.
- `latest.pth.tar` is atomically updated after every completed training iteration. Use it for resume and local testing, not as a public deployment artifact.
- `best.pth.tar` is created or replaced only when a numbered checkpoint improves the external Stockfish benchmark score. It is not a synonym for “newest.”

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
  --checkpoint ./temp/latest.pth.tar \
  --run.stockfish-eval-games 20 \
  --run.stockfish-depth 10
```

Set `--run.stockfish-path` when the executable is not discoverable. Elo limiting defaults
to Stockfish's supported floor of 1320; keep Elo, depth, engine build, and search
settings fixed when comparing checkpoints. Periodic training evaluation is
controlled by `--run.stockfish-eval-every`; set it to `0` to disable. Alternating colors
make comparisons more useful, but a small match is still a noisy estimate.

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
make lint                # Ruff checks
make types               # mypy over src
make test                # pytest suite
make check               # lint + types + tests
make bench               # throughput benchmark
make profile-smoke       # bounded end-to-end profile
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
