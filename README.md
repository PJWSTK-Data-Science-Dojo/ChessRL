# Luna Chess: A Self-Learning Chess Engine

Luna Chess is a single-thread, single-GPU chess engine rated around 1850, trained entirely through self-play without any human knowledge except the rules of the game. It uses deep reinforcement learning and Monte Carlo Tree Search (MCTS) to develop its playing strategy.

## Project Overview

The neural network at the heart of Luna Chess is parameterized by theta (θ) and takes the state of the chess board as input. It produces two outputs:
- A continuous value evaluation v∈[-1,1] of the board position from the current player's perspective
- A policy distribution p that represents probabilities over all possible actions

During training, the network learns from examples of the form (s_t, π_t, z_t), where:
- s_t is the state
- π_t is an estimate of the probability from state s_t
- z_t is the final game outcome ∈ [-1,1]

## Project Structure

```
/
├── LICENSE
├── README.md
├── makefile
├── poetry.lock
├── pyproject.toml
├── requirements.txt
├── runs/                  # Training runs data
├── src/
│   ├── index.html         # Web interface main page
│   ├── luna/              # Core engine components
│   │   ├── NNet.py        # Neural Network wrapper
│   │   ├── coach.py       # Training orchestration
│   │   ├── eval.py        # Evaluation utilities
│   │   ├── game/          # Game mechanics
│   │   │   ├── arena.py   # Self-play arena
│   │   │   ├── luna_game.py # Chess game logic
│   │   │   ├── player.py  # Player implementations
│   │   │   └── state.py   # Game state representation
│   │   ├── luna.py        # Main interface for the engine
│   │   ├── luna_NN.py     # Neural network architecture
│   │   ├── mcts.py        # Monte Carlo Tree Search
│   │   └── utils.py       # Utility functions
│   ├── luna_html_wrapper.py # Web interface backend
│   ├── main.py            # Main training entry point
│   ├── playground.py      # Development playground
│   └── static/            # Web assets
│       ├── chessboard.min.css
│       ├── chessboard.min.js
│       ├── img/
│       └── jquery.min.js
└── temp/                  # Checkpoint storage
```

## Requirements and Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies include:
   - Python 3.7+
   - PyTorch
   - chess
   - numpy
   - Flask (for web interface)
   - stockfish (optional, for comparison)

2. (Optional) If you want to use GPU training, ensure you have CUDA installed and compatible with your PyTorch version.

## How to Train the Model

Training is managed by the `Coach` class, which handles self-play, learning, and model evaluation.

1. Start training from scratch:
   ```bash
   python src/main.py
   ```

2. Resume training from a checkpoint:
   ```bash
   python src/main.py --load_model=True
   ```

### Training Parameters

Edit the `args` dictionary in `src/main.py` to adjust training settings:

```python
args = dotdict({
    'numIters': 5,         # Number of training iterations
    'numEps': 10,          # Number of self-play games per iteration
    'tempThreshold': 10,   # Temperature threshold
    'updateThreshold': 0.6,# Required win rate for new model acceptance
    'maxlenOfQueue': 20000,# Maximum number of game examples to store
    'numMCTSSims': 10,     # Number of MCTS simulations per move
    'arenaCompare': 10,    # Number of games to compare new vs old model
    'cpuct': 1,            # Exploration constant in MCTS
    'checkpoint': './temp/',# Directory to save checkpoints
    'load_model': False,   # Whether to load existing model
    'load_examples': True, # Whether to load saved examples
    'load_folder_file': ('./pretrained_models/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 5,
    'dir_noise': True,     # Add Dirichlet noise for exploration
    'dir_alpha': 1.4,      # Dirichlet alpha parameter
    'save_anyway': True    # Always save model
})
```

Training will produce checkpoint files in the `./temp/` directory, with the best model saved as `best.pth.tar`.

## How to Play Against Luna

Luna Chess includes a web interface to play against the trained model.

1. Start the web server:
   ```bash
   python src/luna_html_wrapper.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. The web interface provides three modes:
   - Play as White: You play as white, Luna plays as black
   - Play as Black: Luna plays as white, you play as black
   - Self-Play: Watch Luna play against itself (access via `/selfplay`)

## Web Interface Features

- **Play as White/Black**: Choose your color and play against Luna
- **Reset Game**: Start a new game
- **Self-Play Mode**: Watch Luna play against itself by navigating to `/selfplay`

### Self-Play Controls

In self-play mode:
- **Start Self Play**: Begin a game with Luna playing against itself
- **Stop**: Pause the self-play demonstration
- **Reset Board**: Start over with a fresh board

## Advanced Usage

### Evaluating Against Stockfish

Luna can be pitted against Stockfish for evaluation:

```python
from luna.game.player import StockFishPlayer
from luna.luna import Luna

# Initialize Luna
luna = Luna()

# Initialize Stockfish (adjust parameters as needed)
stockfish = StockFishPlayer(elo=1800, skill_level=10, depth=10)

# Set up comparison and play
# See luna/game/arena.py for implementation details
```

### Neural Network Architecture

Luna's neural network uses a 3D convolutional architecture:

```
- Input: 8x8x6 (board position serialized)
- Conv3D layers with batch normalization
- Several fully connected layers
- Outputs:
  - Policy head: probability distribution over moves
  - Value head: scalar evaluation of position
```

The network is trained to minimize the loss function:
```
l = ∑(vθ(st)-zt)² - →πt⋅log(→pθ(st))
```

### Monte Carlo Tree Search Implementation

The MCTS implementation follows these steps:
1. Initialize a search tree with the current position as root
2. In each simulation:
   - Select moves that maximize upper confidence bound
   - Expand the tree with a new node when a leaf is reached
   - Evaluate the position using the neural network
   - Backpropagate the evaluation up the search path
3. After simulations, choose a move based on visit counts

## Key Files and Their Functions

- `luna.py`: Main interface to the chess engine
- `luna_NN.py`: Neural network architecture
- `NNet.py`: Neural network training and inference wrapper
- `mcts.py`: Monte Carlo Tree Search implementation
- `coach.py`: Self-play training orchestration
- `luna_game.py`: Chess game environment
- `luna_html_wrapper.py`: Web interface for human play

## Common Issues and Troubleshooting

- **Missing endpoints in Self-Play mode**: If you see a "Not found" error for `/next_move`, ensure you've updated your `luna_html_wrapper.py` with the latest self-play implementation.
- **Board orientation issues**: The board may reset orientation after a game ends. This is a UI issue that can be fixed in the JavaScript code.
- **CUDA out of memory**: Reduce batch size or number of MCTS simulations if you encounter memory issues during training.
- **Training plateau**: If performance plateaus, try increasing Dirichlet noise (`dir_alpha`) for more exploration or adjust the learning rate.

## Future Improvements

1. Enhanced opening book integration
2. Time management for competitive play
3. Multi-GPU training support
4. Endgame tablebases integration
5. Progressive network pruning for speed optimization

## License and Credit

Luna Chess is provided under an open-source license. The project draws inspiration from AlphaZero's approach to chess learning through pure self-play.
