# Luna ChessRL

Luna is a model-based reinforcement-learning chess engine trained from self-play. It combines an EfficientZeroV2-style latent world model with Gumbel MuZero search, prioritized replay, n-step value targets, and optional auxiliary representation objectives. The repository includes training, evaluation, an interactive browser experience, a UCI adapter, and a secure Lichess deployment helper.

This is a research and portfolio project, not a claim of engine parity with established tournament systems. Playing strength depends on training data, compute budget, search settings, and evaluation methodology.

## What is implemented

- Representation, dynamics, and prediction networks with per-sample latent normalization.
- Gumbel top-m root selection and Sequential Halving by default; classic PUCT remains available.
- Batched latent MCTS for parallel self-play, exact legal-move masks, and two-player value backups.
- A 4,288-action chess space: 4,096 from/to actions plus distinct knight, rook, and bishop underpromotions.
- Five spatial dynamics action planes: from, to, and one plane for each underpromotion identity.
- A convolutional policy head aligned to source squares, destination squares, and underpromotion identity.
- A 119-plane observation tensor with eight positions of piece/repetition history and complete current rule state.
- K-step unrolled policy, value, and reward losses; optional SimSiam consistency and training-only board reconstruction.
- Three-bin categorical value and reward supports for scalar targets in the `[-1, 1]` chess value range.
- Prioritized trajectory replay with compact `float16` observations/policies and boolean legal masks.
- AdamW, learning-rate warm-up followed by cosine decay, mixed precision, gradient clipping, accumulation, and recurrent gradient scaling.
- Optional batched search-value estimation and policy reanalysis using the current network.
- Versioned, atomic checkpoints that include architecture, optimizer, scaler, step,
  trainer iteration, and the learning-rate schedule horizon.

The implementation follows ideas from [MuZero](https://arxiv.org/abs/1911.08265), [EfficientZero](https://arxiv.org/abs/2111.00210), [EfficientZero V2](https://proceedings.mlr.press/v235/wang24at.html), and [Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO).

## Quick start

Run every command from the repository root.

```bash
uv sync --frozen --extra dev --extra perf
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

The maintained preset uses four persistent self-play actors with 32 games each, a
256-position learner batch split into two microbatches, a five-step latent unroll,
a 32-step bootstrapped value horizon, and a 300,000-position replay window. It selects
`balanced_reconstruction`: the 128-channel asymmetric SE-ResNet plus a training-only
13-class piece decoder. SimSiam remains disabled; after 5,000 optimizer steps, 2% of
sampled trajectories receive an eight-simulation value and policy reanalysis. BF16 and
compiled training remain enabled. External evaluation uses eager inference because its variable
search workload proved unstable under long-running Inductor compilation. Root exploration remains active throughout the bounded
256-ply self-play game, and a self-play-only guard reruns a root search when the chosen
move would immediately enable a threefold-repetition claim. The full legal mask is still
stored as the learning target. If a game reaches the operational ply limit, its final
post-action state is evaluated and stored as a bootstrap boundary; truncation is never
treated as a chess draw. The replay capacity counts positions, not trajectories.

After a fresh process or resume, learning waits for 50,000 replay positions. Thereafter
the learner targets two sampled positions per newly generated position, capped at 200
optimizer steps per iteration, while the cosine schedule keeps an explicit 72,000-step
horizon. These limits avoid letting
unusually short self-play games multiply their own distribution through a fixed update count.

## Expert PGN warm start

For a single-GPU run, the maintained expert warm start trains the same
`balanced_reconstruction` network before returning it to online Gumbel MuZero self-play.
It supervises the played expert action, the position value, all five recurrent dynamics
steps, and board reconstruction. Positions with a Lichess `%eval` annotation use its
Stockfish WDL expectation from the side-to-move perspective; the final game result is the
fallback. Reward, SimSiam, and reanalysis losses are disabled in this offline phase.
The next online phase resets AdamW and the LR schedule while preserving the pretrained
weights and source-checkpoint SHA-256 provenance.

The default reproducible corpus is the July 2026
[Lichess Broadcast database](https://database.lichess.org/broadcast/lichess_db_broadcast_2026-07.pgn.zst)
export: 40,038 official
broadcast games under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The archive is not committed. Its download target verifies the pinned official SHA-256
`714d0eb99f99fca8d791142038b6c59b5ca6a51b3339bd3891a92f4bdffcbf0c`,
and the importer keeps at most 300,000 positions from complete standard games where both
players are rated at least 2000. Training and validation are split by whole-game hash, so
positions from one game cannot leak across the split.

```bash
make download-pgn-data
make verify-pgn-data
make pretrain-pgn
```

`src/pretrain_pgn.py` is the typed CLI behind `make pretrain-pgn`. A fresh invocation uses
the current state-anchor `best.pth.tar` as weights-only input; a repeated invocation recovers
the newest healthy immutable PGN checkpoint, including when `latest.pth.tar` is missing or
corrupt. Both paths use the explicit W&B ID `luna-balanced-pgn-pretrain-v1`, which is also
part of the resume contract. Both use W&B's documented `resume=allow` behavior with that
fixed ID, closing the crash window between remote run creation and the first local
checkpoint; the local resume contract rejects a changed dataset, seed, or W&B identity.

The ten 1,000-step milestones are retained because supervised validation is not a chess
strength measurement. Benchmark at least the 2k, 5k, and 10k milestones with the same
paired openings before choosing one:

```bash
make eval-pgn-warmstart PGN_EVAL_CHECKPOINT=./runs/luna-balanced-pgn-pretrain-v1/pretrain_step_00002000.pth.tar
make eval-pgn-warmstart PGN_EVAL_CHECKPOINT=./runs/luna-balanced-pgn-pretrain-v1/pretrain_step_00005000.pth.tar
make eval-pgn-warmstart PGN_EVAL_CHECKPOINT=./runs/luna-balanced-pgn-pretrain-v1/pretrain_step_00010000.pth.tar
make train-pgn-warmstart PGN_SELECTED_CHECKPOINT=./runs/luna-balanced-pgn-pretrain-v1/pretrain_step_00005000.pth.tar
```

The last command starts a weights-only online phase under the separate W&B ID
`luna-balanced-ezv2-pgn-warmstart-v1`; subsequent calls resume it without requiring the
selection argument. The PGN phase is a bounded initialization, not a permanent
human-policy regularizer: online search targets replace imitation targets so the model
can recover from human mistakes and states outside the expert distribution.

## LCZero policy/value calibration

The second offline option consumes native [LCZero training data](https://storage.lczero.org/files/training_data/)
without expanding the archive on disk. The pinned pilot is
`training-run2-test91-20260901-1317.tar`: 508,417 V6 records in 4,416 game members.
`make download-lc0-data` retrieves it atomically and verifies SHA-256
`d6fe77a11c71d758dfbff0d07e80958f04440d26fa1f925e0e3683e1a3ad7409`.
The collection is provided under ODbL 1.0 and its individual contents under DBCL 1.0;
the upstream [license text](https://storage.lczero.org/files/training_data/LICENSE.txt)
is also embedded in the archive. The data remains Git-ignored.

The streaming adapter accepts LCZero V6/V7 records with input formats 1 through 4,
filters whole Chess960 games, and maps the official 1,858-entry policy into Luna's
4,288 actions. Castling, en passant, the context-dependent knight-promotion index,
and all explicit promotion types retain their chess semantics. The value target is
the complete side-to-move `[loss, draw, win]` distribution; it is never collapsed to
a scalar. Train and validation are split by whole game, preventing adjacent positions
from leaking across the boundary.

This phase deliberately trains only `prediction.policy_head` and
`prediction.value_head`. Representation, dynamics, reward, reconstruction, and
consistency parameters are frozen and checked by a SHA-256 invariant after every
chunk. This preserves the latent basis learned by online MuZero. It is a bounded head
calibration, not a substitute for learning dynamics or a guarantee of higher Elo:
the same prediction heads serve recurrent MCTS states, so lower held-out cross-entropy
can still coincide with weaker search.

```bash
make download-lc0-data
make verify-lc0-data
make pretrain-lc0
```

The maintained pilot uses approximately one pass: 1,000 optimizer steps, batch 512,
BF16, peak LR `1e-4`, a 50-step warm-up, and checkpoints at steps 250, 500, 750, and
1,000. It starts from the immutable PGN-online iteration-25 source whose pinned SHA-256
is `79376fa55a6f276f59af30479dc12f6bc939c87d91d342947578157782d4f7c6`.
Crashes recover the newest healthy `lc0_step_*.pth.tar` together with its optimizer and
resume the explicit W&B run `luna-balanced-lc0-heads-pretrain-v1`.

Do not select a milestone from validation loss alone. Compare the retained checkpoints
under identical 32-simulation search settings, first at Fairy-Stockfish 600 because the
source already passes the 500 rung. The fixed Stockfish-1500 match remains telemetry;
the source's 0-20 result makes it too difficult to rank early candidates.

```bash
make eval-lc0-warmstart \
  LC0_EVAL_CHECKPOINT=./runs/luna-balanced-lc0-heads-pretrain-v1/lc0_step_00000250.pth.tar \
  ARGS='--run.ladder-start-elo 600'
make eval-lc0-warmstart \
  LC0_EVAL_CHECKPOINT=./runs/luna-balanced-lc0-heads-pretrain-v1/lc0_step_00000500.pth.tar \
  ARGS='--run.ladder-start-elo 600'
make eval-lc0-warmstart \
  LC0_EVAL_CHECKPOINT=./runs/luna-balanced-lc0-heads-pretrain-v1/lc0_step_00000750.pth.tar \
  ARGS='--run.ladder-start-elo 600'
make eval-lc0-warmstart \
  LC0_EVAL_CHECKPOINT=./runs/luna-balanced-lc0-heads-pretrain-v1/lc0_step_00001000.pth.tar \
  ARGS='--run.ladder-start-elo 600'
```

Only a checkpoint that does not regress the source under MCTS should start online
learning. That transition resets AdamW and the LR schedule and uses the distinct W&B
identity `luna-balanced-ezv2-lc0-warmstart-v1`:

```bash
make train-lc0-warmstart \
  LC0_SELECTED_CHECKPOINT=./runs/luna-balanced-lc0-heads-pretrain-v1/lc0_step_00000500.pth.tar
```

## Historical phase commands

To carry the complete model, optimizer, scaler, counters, and LR horizon from the current
`runs/luna-strength-1500-v1/latest.pth.tar` into the separate ladder contract, run this
once, then use `make resume-migrated-phase` thereafter:

```bash
make migrate-ladder-phase
```

That complete-state migration target defaults to `runs/luna-fairy-ladder-v1` and must
initially be absent or empty.
Replay is process memory and therefore starts empty. If the first migration process is
interrupted after creating its W&B run but before publishing target `latest.pth.tar`, retry
explicitly with `make migrate-ladder-phase ARGS='--wandb-resume allow'`.

`make train-phase` and `make resume-phase` preserve the old weights-only experiment
contract for forensic reproducibility. Do not use their default iter-40 source for new
training: diversity audits showed that it was already collapsed. New experiments must
use `make train` with a new checkpoint directory and W&B identity, which starts from
random weights. `make migrate-ladder-phase` remains available only for deliberate
complete-state migrations of a known healthy lineage.

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
`luna-balanced-ezv2-anti-collapse-v2`) through `--wandb-run-id` and the display name
`Luna Balanced EZ-V2 · Anti-Collapse v2` through `--wandb-run-name`. `make train-phase` uses
`--wandb-resume never`, so it refuses to append a new phase to an existing remote run;
`make resume-phase` uses `--wandb-resume must`, so a typo or missing remote run fails
instead of silently creating a second dashboard. Change the run ID only when intentionally
starting a different phase. If the remote run was deliberately deleted while its local
checkpoint remains valid, recreate it explicitly with
`make resume-phase ARGS='--wandb-resume allow'`. The general CLI defaults to `allow`.

The historical phase preset collects 128 self-play episodes with four persistent actors
running up to 32 active games each, trains with batches of 256 after replay warm-up, and
reanalyzes 2% of eligible samples for value only. Every five iterations it plays a 20-game
paired-opening match at the current Fairy-Stockfish rung. The ladder starts at native
`UCI_Elo=500`, advances by 100
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
buffer state. Repetition-guard interventions, actual optimizer steps, gradient clipping,
reanalysis work, board-reconstruction accuracy, grounded-value coverage, and raw latent
diversity are explicit metrics. Fixed-opponent
results use `benchmark/*`; adaptive results use `ladder/*`
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
2. Each position stores observation, action, acting-player reward, improved search policy, root value, and legal mask. A time-limited trajectory also stores the value of its post-action boundary state.
3. Prioritized replay samples positions and constructs K-step unroll targets with alternating player signs.
4. Optional reanalysis refreshes selected value and policy targets with the current model.
5. The learner performs one optimizer update per training step, using accumulation only to form that update.
6. Numbered and `latest` checkpoints are saved atomically; optional external evaluation may promote `best`.

Chess uses an undiscounted terminal objective (`discount=1.0`). Terminal outcomes are represented explicitly as `None` while ongoing and `-1`, `0`, or `1` when complete. A maximum-ply cutoff is a truncation, not a terminal outcome, so value targets bootstrap across it. Illegal model actions fail fast instead of being replaced silently.

## Useful commands

```bash
make fmt                 # format and apply safe Ruff fixes
make lock-check          # verify pyproject.toml and uv.lock agree
make format-check        # verify Ruff formatting without editing files
make lint                # Ruff checks
make types               # mypy over src
make test                # pytest suite
make check               # format check + lint + types + tests
make audit               # audit locked runtime dependencies (network required)
make bench               # throughput benchmark
make profile-smoke       # bounded end-to-end profile
make download-pgn-data   # fetch and verify the pinned expert corpus
make pretrain-pgn        # start or resume supervised PGN warm-start training
make train-pgn-warmstart # start or resume the online phase from PGN weights
make train-phase         # start the dedicated strength continuation phase
make resume-phase        # resume that phase after an interruption
make resume-migrated-phase  # resume the separate complete-state migration
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

When the locked PyTorch build is unsuitable for a host, update the dependency with
`uv add` and the appropriate official PyTorch package index, then commit the resulting
`pyproject.toml` and `uv.lock` together. Do not mutate the managed environment with `pip`.

## Development standards

The repository uses `uv` as its only Python environment and package manager. Runtime
and development dependencies are locked in `uv.lock`; update them with `uv add` or
`uv lock`, never with an ad-hoc `pip` install. Before opening a change, run:

```bash
make fmt
make check
```

`make check` is the same quality gate used in CI: Ruff formatting and linting, strict
mypy over production code, and the complete pytest suite. Python source and test modules
must remain below 400 lines; split a growing module by responsibility instead of hiding
the limit with generated files or exclusions. Tests should be deterministic and must not
require a GPU, network connection, W&B credentials, or a local chess-engine binary unless
the dependency is explicitly mocked.

## Repository map

```text
src/main.py                    training entry point
src/web_app.py                 stable web entry point and public imports
src/web_*.py                   engine service, game state, security, and API routes
src/eval_vs_stockfish.py       standalone external evaluation
src/luna/config*.py            typed configuration and validation
src/luna/coach*.py             self-play, training, evaluation, and checkpoints
src/luna/network*.py           learner, inference, diagnostics, and checkpoint I/O
src/luna/model_factory.py      configured model architecture selection
src/luna/ezv2_networks.py      baseline architecture and support transforms
src/luna/balanced_networks.py  asymmetric SE-ResNet and state-anchored variant
src/luna/mcts*.py              Gumbel MuZero, PUCT, and batched latent search
src/luna/replay_buffer.py      prioritized trajectory replay
src/luna/targets.py            TD and unroll targets
src/luna/pgn_*.py              expert-PGN ingestion, validation, and warm-start training
src/luna/uci.py                UCI adapter
src/luna/lichess_config.py     secure lichess-bot config generator
src/luna/game/                 chess rules, arena, players, Stockfish eval
tests/                         unit, integration, protocol, and regression tests
```

## Reproducibility notes

- `--seed` seeds Python, NumPy, and PyTorch. Some accelerator kernels may remain nondeterministic.
- `--learner.model-name` selects `baseline`, `balanced`, or `balanced_reconstruction`; the maintained `make train` preset uses the last option with 128 channels, 10 representation blocks, one dynamics block, and a training-only piece decoder.
- Gumbel search does not use Dirichlet root noise. Barlow Twins, playout-cap randomization, FP8/TensorRT, and alternative chess-rule backends remain ablation candidates rather than defaults; each changes training semantics or runtime behavior and requires a controlled benchmark.
- Training checkpoints restore optimizer, scaler, global step, and trainer iteration. The in-memory replay buffer is not serialized.
- `torch.compile` is optional. Disable it with `--learner.no-compile-inference` or the corresponding web flag when unsupported.
- Measure throughput with the included benchmark and profiler before changing batch, parallel-game, or search budgets.
