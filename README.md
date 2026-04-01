# ChessRL: EfficientZeroV2 Chess Engine

ChessRL trains a chess engine from self-play using an EfficientZeroV2-style setup:

- latent-state representation + dynamics + prediction networks
- latent MCTS for move search
- prioritized replay buffer for training data
- iterative self-play -> train -> evaluate loop

No handcrafted chess heuristics are required beyond game rules.

## Quick Start

```bash
uv sync
make train
```

- `uv sync` installs dependencies from `pyproject.toml`.
- `make train` runs `python src/main.py`, which performs self-play and training together.

## Self-Play + Training Workflow

The default training run in `src/main.py` does all major stages:

1. Generate self-play games with MCTS (`numEps` per iteration).
2. Store trajectories in prioritized replay.
3. Train the network from replay (`train_steps_per_iter` updates).
4. Arena-evaluate new vs previous model.
5. Save checkpoints to `./temp/` (`temp.pth.tar`, `checkpoint_<iter>.pth.tar`, `best.pth.tar`).

You can tune run behavior by editing `args` in `src/main.py`.

## Main Training Parameters

Configured in `src/main.py`:

| Parameter | Description |
|-----------|-------------|
| `numIters` | Number of train/eval iterations |
| `numEps` | Self-play episodes per iteration |
| `numMCTSSims` | MCTS simulations per move |
| `cpuct` | PUCT exploration constant |
| `unroll_steps` | Unroll length for latent training |
| `td_steps` | Bootstrap horizon for value targets |
| `train_steps_per_iter` | Gradient steps per iteration |
| `batch_size` | Replay batch size |
| `replay_capacity` | Replay buffer capacity |
| `per_alpha` / `per_beta` | Prioritized replay parameters |
| `checkpoint` | Checkpoint output directory |
| `save_anyway` | Accept new model without arena threshold |

## Project Structure

```
src/
├── main.py                    # self-play + training entry point
├── luna/
│   ├── coach.py               # training loop orchestration
│   ├── mcts.py                # latent MCTS
│   ├── network.py             # learner wrapper and optimization
│   ├── ezv2_model.py          # representation/dynamics/prediction model
│   ├── replay_buffer.py       # prioritized replay
│   ├── targets.py             # unroll target construction
│   ├── eval.py                # evaluation helpers
│   ├── luna.py                # engine interface
│   ├── utils.py               # utilities
│   └── game/
│       ├── luna_game.py       # chess environment
│       ├── arena.py           # head-to-head evaluation
│       ├── player.py          # players (human/engine/random)
│       └── state.py           # board state representation
└── luna_html_wrapper.py       # Flask web interface
```

## Development Commands

```bash
uv sync --extra dev
make fmt
make lint
make types
make check
make test
make bench
make serve
```
