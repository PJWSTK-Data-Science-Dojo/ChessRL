# AGENTS.md

This is the authoritative repository guide for coding agents and contributors.

## Project

Luna ChessRL is an EfficientZeroV2-style chess engine trained from self-play. It contains a latent representation/dynamics/prediction model, Gumbel MuZero and PUCT search, prioritized replay, unrolled learning, search-value reanalysis, external evaluation, an interactive Flask UI, UCI support, and Lichess tooling.

Run commands from the repository root. The `luna` package is installed from `src/luna`; use `uv run` rather than ad-hoc `PYTHONPATH` changes.

## Commands

```bash
uv sync --extra dev --extra perf
make check                 # Ruff, mypy, and pytest
make train                 # maintained single-accelerator experiment preset
make resume                # resume temp/latest.pth.tar with that preset
make profile-smoke         # bounded end-to-end profile
make bench                 # throughput benchmark
make serve                 # browser UI with temp/latest.pth.tar
make serve-cpu
make serve-mps
make uci
make lichess-config
make eval-stockfish
make test-pipeline-cpu
make test-pipeline-mps
```

Use `uv run python src/main.py --help`, `uv run python src/web_app.py --help`, and `uv run luna-uci --help` for the live CLI contract.

## Python engineering rules

### Toolchain

- Use `uv` exclusively for dependency and environment management. Never use pip, Poetry, or Conda in commands, scripts, containers, or documentation.
- Format and lint with Ruff: `uv run ruff format` and `uv run ruff check --fix`. Do not introduce Black, Flake8, isort, or Pylint.
- Run mypy in strict mode and fix the underlying type contract. Do not use `# type: ignore` or `# noqa` to silence a checker.
- Run tests with `uv run pytest`. Keep tests fast, independent, and focused on one behavior using Arrange–Act–Assert structure.

### Types and APIs

- Annotate every function signature. Use modern Python syntax such as `str | None`, built-in generic types, `match`, `tomllib`, and `Self` where they improve clarity.
- Prefer frozen dataclasses or Pydantic models for structured data that crosses an API boundary. Do not pass growing dictionaries with undocumented keys between subsystems.
- Never use mutable default arguments. Prefer absolute imports grouped as standard library, third party, then `luna` modules.
- Use intention-revealing names. Classes are nouns and methods are verbs. Keep functions focused on one abstraction level; fewer than 20 lines and at most three arguments are design targets, not reasons to create single-use wrappers.
- Do not return `None` as a hidden error or as a substitute for an empty collection. `None` remains valid when it is the explicit domain value, such as an ongoing chess outcome, or the natural return of a mutating command.

### Errors, documentation, and structure

- Raise specific exceptions with actionable context. Never return error flags, swallow failures, use bare `except`, or catch `Exception` only to continue silently.
- Follow Google docstring conventions for public APIs with non-obvious constraints or semantics. Omit docstrings on trivial private helpers and omit `Args` or `Returns` sections that only repeat annotations.
- Comments explain why a constraint exists, never what the following statement does. Do not add banner comments, commented-out code, unsolicited demo blocks, debug prints, placeholder TODOs, or self-congratulatory prose.
- Apply DRY, YAGNI, and KISS together. Delete verified dead code. Do not add a factory, manager, handler, compatibility shim, version branch, configuration hook, or custom stdlib replacement for one use case.
- Match the surrounding module while moving touched code toward these rules. Avoid unrelated refactors whose risk is larger than their measured benefit.

### Security

- Read secrets from the environment or a gitignored private file. Never hardcode, print, log, persist in generated artifacts, or commit tokens, credentials, personal data, or private paths.
- Validate boundary inputs and fail loudly on contract violations. Avoid defensive defaults that turn malformed internal data into plausible output.
- Keep code, comments, and log messages professional and emoji-free.

## Core files

| File | Responsibility |
|---|---|
| `src/luna/coach.py` | self-play, replay, training schedule, evaluation, checkpoints |
| `src/luna/network.py` | learner, inference APIs, optimizer, checkpoint I/O |
| `src/luna/ezv2_networks.py` | representation, dynamics, prediction, SimSiam heads |
| `src/luna/mcts.py` | Gumbel MuZero and classic PUCT latent search |
| `src/luna/replay_buffer.py` | compact trajectory storage and prioritized sampling |
| `src/luna/targets.py` | alternating-sign TD and K-step targets |
| `src/luna/game/chess_game.py` | chess rules, canonicalization, observations, actions |
| `src/luna/config.py` | dataclass and Tyro configuration |
| `src/web_app.py` | Flask factory, session-isolated games, `/api/v1` |
| `src/luna/uci.py` | UCI state machine and clock-aware search |
| `src/luna/lichess_config.py` | secure upstream lichess-bot config generator |

## Invariants

- The action size is 4,288. Indices `0..4095` encode from/to moves and queen promotions. Three 64-entry ranges encode knight, rook, and bishop underpromotions separately.
- The policy head is spatial: source squares are tensor locations, destination squares are output channels, and dedicated channels preserve underpromotion identity.
- Dynamics receives five spatial action planes: from, to, and the three underpromotion identities.
- Observations have 119 planes: eight temporal positions with separate piece/repetition planes plus current rule state. History must survive canonical mirroring.
- `get_game_outcome()` returns `None` for ongoing play and `-1.0`, `0.0`, or `1.0` from the requested player's perspective.
- Stored rewards are from the parent/acting-player perspective. Two-player backups and TD targets alternate signs.
- Chess uses `discount=1.0` by default.
- Legal masks apply at roots, recurrent expansions, and every policy-loss step. Illegal model actions raise; never replace them with a random move.
- Gumbel search is the default. Its selected self-play action and improved training policy are related but distinct outputs. PUCT remains an explicit fallback mode.
- Reanalysis batches replayed positions and uses direct search-value overrides. Do not silently reinterpret them as an unrelated bootstrap position.
- One optimizer step corresponds to one training step; gradient accumulation forms that step rather than multiplying it.
- Recurrent latent edges use configurable gradient scaling. Optimizer scheduling uses warm-up and cosine decay based on restored global step.

## Checkpoints

Only format version 2 is supported. It stores model weights, architecture metadata, optimizer, AMP scaler, global step, and trainer iteration. Loading is strict and fail-closed.

- `checkpoint_<iteration>.pth.tar`: numbered snapshot.
- `latest.pth.tar`: atomically published resume and runtime checkpoint.
- `best.pth.tar`: only an externally evaluated checkpoint promoted by Stockfish score.

No checkpoint belongs in Git. Do not add compatibility logic for deleted legacy formats unless a user explicitly requests a separate conversion tool.

The replay buffer is in memory and is not checkpointed. Resume restores learner/trainer state, not past trajectories.

## Training and evaluation

The maintained `make train` preset uses batched self-play, small-budget Gumbel search, mixed precision, AdamW, compact replay, gradient accumulation, and partial reanalysis. Treat it as an experiment starting point, not a strength or completion guarantee.

Keep Stockfish settings fixed when comparing checkpoints. A short alternating-color match has high variance. `best.pth.tar` promotion is based on the configured external score; training loss must not determine “best.”

Profile before changing throughput settings. The main controls are parallel games, simulations, batch size, model width/depth, unroll length, and reanalysis rate.

## Web and UCI

`create_app(engine)` accepts an injected `LunaEngineService`; production `wsgi.py` constructs the verified engine first. The health route is `/api/v1/health`.

Web game state is process-local and session-owned. Production must use one Gunicorn worker; threads are acceptable because inference has a process-wide lock.

UCI stdout is protocol-only. Send diagnostics to stderr. Preserve `position startpos`, `position fen`, clocked and interruptible `go`/`stop`, and `go searchmoves`, including canonical Black action mapping. Claimable draws still require a legal move until the remote game is actually over.

The Lichess bridge is documented in `LICHESS.md`. Tokens come only from environment input during generation, are never printed, and the generated config is mode `0600`.

## Change discipline

- Preserve unrelated user changes in a dirty worktree.
- Use `rg`/`rg --files` for discovery and `apply_patch` for edits.
- Add regression tests for behavior changes, especially game perspective, terminal handling, action identity, search, checkpoints, and protocol parsing.
- Run focused tests while iterating, then `make check` when practical.
- Never commit credentials, checkpoints, profiler traces, runtime configs, or generated logs.
- Keep user-facing text in professional English.
- Do not make unmeasured playing-strength or training-time claims.
