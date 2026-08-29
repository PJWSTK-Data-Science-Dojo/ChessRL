# CLAUDE.md

Use [AGENTS.md](AGENTS.md) as the authoritative development guide for this repository.

The most important constraints are:

- Run commands from the repository root through `uv run` or the maintained Make targets.
- Preserve the 4,288-action underpromotion-aware encoding and 119-plane temporal observation contract.
- Keep legal masking and two-player value/reward perspectives exact throughout search and training.
- Gumbel MuZero is the default search; PUCT is an explicit alternative.
- Use only format-v2 checkpoints. `latest.pth.tar` is the resume/runtime artifact; `best.pth.tar` is reserved for external evaluation promotion.
- The production web API is `/api/v1`, is built with `create_app()`, and must use one process because game state is in memory.
- Keep UCI stdout protocol-safe and retain `go searchmoves` behavior.
- Do not add credentials, checkpoints, runtime configuration, logs, or profiler output to Git.
- Do not make unmeasured playing-strength or training-duration claims.

Before handing off a change, run the narrow regression tests and then `make check` when practical.
