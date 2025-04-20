# Luna Chess: A Self-Learning Chess Engine

Luna Chess is a Python-based chess engine trained entirely through self-play, inspired by the principles of AlphaZero. It learns to play chess without relying on human games or opening books, using only the basic rules of the game. At its core, Luna combines a deep neural network with Monte Carlo Tree Search (MCTS) to evaluate positions and select moves.

## Project Overview

The project implements a self-learning loop where a neural network (`LunaNN`) is iteratively improved. The network, parameterized by weights θ, takes a representation of the chess board state as input. It outputs:
1.  **Policy (p):** A probability distribution over all possible moves from the current state.
2.  **Value (v):** A scalar estimate between -1 and +1, representing the expected outcome of the game from the current player's perspective (-1 for loss, 0 for draw, +1 for win).

The training process cycles through:
1.  **Self-Play:** The current best version of the engine (NN + MCTS) plays games against itself. During these games, MCTS uses the NN's predictions to guide its search and generate a stronger policy (π) than the raw NN output (p). These game states, the MCTS-derived policies (π), and the final game outcomes (z) are collected as training examples.
2.  **Training:** The neural network is trained on the collected examples to make its raw outputs (p and v) closer to the MCTS-derived policy (π) and the observed game outcome (z). The loss function used is the sum of a mean-squared error for the value head and cross-entropy for the policy head: `l = ∑(vθ(st)-zt)² - →πt⋅log(→pθ(st))`.
3.  **Arena:** The newly trained network is pitted against the previous best network in a series of games. If the new network performs significantly better (wins a certain percentage of games), it becomes the new "best" version and replaces the old one for subsequent self-play iterations.

This iterative process allows the engine to continuously improve its play.

## Key Concepts

*   **Self-Play:** Generating training data by having the AI play against itself.
*   **Monte Carlo Tree Search (MCTS):** A search algorithm that uses random sampling in the search space to estimate the value of states and moves. It balances exploration (trying less visited moves) and exploitation (pursuing promising moves) using the Upper Confidence Bound for Trees (UCT) formula, guided by the neural network's policy and value predictions.
*   **Neural Network (ResNet):** A deep learning model that learns to predict move probabilities (policy) and position evaluation (value) directly from the board state.
*   **Training Examples (s, π, z):** Data points collected from self-play games, where 's' is the state, 'π' is the MCTS-improved policy for that state, and 'z' is the final outcome of the game the state came from.
*   **Canonicalization:** Representing board states consistently for the neural network input, typically from White's perspective, regardless of whose turn it is.
*   **Symmetry:** Augmenting training data by considering symmetrical board positions (e.g., horizontal or vertical flips) as additional training examples.

## Project Structure

```txt
/
├── LICENSE
├── README.md
├── makefile # (May contain build/run commands, not detailed in code)
├── poetry.lock # (Dependency management file, not detailed in code)
├── pyproject.toml # (Dependency management file, not detailed in code)
├── requirements.txt # Lists Python dependencies
├── runs/ # Directory for logging training runs (e.g., WandB)
├── src/
│ ├── config/
│ │ └── config.yaml # Primary Configuration File
│ ├── index.html # Frontend HTML for the main web interface
│ ├── luna/ # Core engine components package
│ │ ├── init.py # Package initializer for luna and luna.game subpackage
│ │ ├── NNet.py # Neural Network wrapper: handles training, prediction, loading/saving
│ │ ├── coach.py # Orchestrates the training loop (Self-Play, Train, Arena)
│ │ ├── eval.py # Utility for simple evaluation (deprecated)
│ │ ├── game/ # Chess game mechanics package
│ │ │ ├── init.py # Package initializer for game
│ │ │ ├── arena.py # Defines the sequential game player and parallel worker for Arena
│ │ │ ├── luna_game.py # Core chess logic, board representation, actions, game end
│ │ │ ├── player.py # Example players (Random, Human, Stockfish) - not used in core training loop
│ │ │ └── state.py # Older/alternative board state serialization logic (second_serialize_board used by luna_game now)
│ │ ├── luna.py # Main Python interface for the engine (used by web wrapper)
│ │ ├── luna_NN.py # Defines the PyTorch Neural Network architecture (ResNet)
│ │ ├── mcts.py # Monte Carlo Tree Search implementation
│ │ └── utils.py # Helper functions (e.g., AverageMeter, path utilities)
│ ├── luna_html_wrapper.py # Flask web server backend for the GUI
│ ├── main.py # Entry point for training the engine
│ ├── playground.py # Script for testing models (e.g., against themselves)
│ └── static/ # Static web assets (CSS, JS, images)
│ ├── chessboard.min.css # CSS for chessboard.js
│ ├── chessboard.min.js # JavaScript library for interactive chessboard
│ ├── img/ # Chess piece images
│ └── jquery.min.js # JavaScript library
└── temp/ # Default directory for saving/loading checkpoints and training examples
```
## Requirements and Installation

1.  **Clone the Repository:** If you haven't already, clone the project repository.
    ```bash
    git clone <repository_url>
    cd ChessRL # Or wherever you cloned it
    ```
    *(Note: Based on your path `/mnt/d/ChessRL/ChessRL`, it seems you are working within a specific directory structure. Ensure you are in the root of the project.)*

2.  **Install Dependencies:** Navigate to the root directory of the project where `requirements.txt` is located.
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include `torch`, `chess`, `numpy`, `flask`, `omegaconf`, `coloredlogs`, `tqdm`, `wandb` (optional, for logging), and `python-stockfish` (optional).

3.  **(Optional) GPU Support:** If you have a compatible NVIDIA GPU and CUDA installed, PyTorch will automatically use it. Ensure your CUDA and PyTorch versions are compatible (check PyTorch website for installation instructions if needed). The project uses `torch.cuda.is_available()` to determine GPU usage based on the `cuda` flag in the configuration, which is automatically set.

## Understanding the Code

This section provides a deeper look into the core components:

### Game Mechanics (`luna_game.py`)

*   **Board Representation:** Uses the `python-chess` library's `chess.Board` object to represent the standard 8x8 chess board state.
*   **Feature Planes:** Converts a `chess.Board` into a multi-plane NumPy array (`_board_to_feature_array`) for the neural network. Each plane represents a specific piece type or game state aspect (e.g., White Pawns, Black Rooks, Castling Rights, Turn, En Passant, Move Counts, Repetition flags). There are 22 such feature planes per board state.
*   **History:** Maintains a `collections.deque` (`history_len` configured, typically 8) storing the `_board_to_feature_array` for the current and previous board states. This provides the NN with temporal information. The `toArray` method stacks these history frames into the final NN input tensor shape (e.g., `(22 * 8, 8, 8)`).
*   **Actions:** Represents moves in a flat 64\*64 = 4096 action space, corresponding to all possible "from square" to "to square" pairs. The `from_move` function maps a `chess.Move` object to this index. `getNextState` handles applying an action index to the board, defaulting to Queen promotion for pawn moves reaching the back rank. `getValidMoves` generates a binary mask (size 4096) indicating which actions are legal from the current board state.
*   **Game End:** `getGameEnded` checks for checkmate, stalemate, and various draw conditions (insufficient material, 50-move rule, 75-move rule, 3-fold repetition, 5-fold repetition) using `board.outcome(claim_draw=True)`. It returns a value (+1, -1, or a small non-zero for draw) relative to a specified player's perspective.
*   **Canonical Form & Symmetry:** `getCanonicalForm` returns the board state from White's perspective (mirrors the board if it's Black's turn). `getCanonicalHistory` mirrors each frame in the history deque if it's Black's turn. `getSymmetries` generates symmetric training examples by vertically flipping the canonical board state and the MCTS policy vector.
*   **State Key:** `stringRepresentation` uses the board's FEN string as a unique key for MCTS state caching.

### Neural Network (`luna_NN.py`, `NNet.py`)

*   **Architecture (`luna_NN.py`):** Implements a ResNet-inspired architecture with an initial convolutional block, several residual blocks, and separate policy and value heads.
    *   Takes stacked history frames (`num_input_channels`, 8, 8) as input.
    *   `num_channels` and `num_res_blocks` are configurable (see `config.yaml`).
    *   Policy Head: Predicts logits over the 4096 action space.
    *   Value Head: Predicts a scalar value.
*   **Wrapper (`NNet.py`):** Encapsulates the `LunaNN` PyTorch module and provides methods for:
    *   **Initialization:** Sets up the network, optionally moves it to GPU based on config. It can be initialized for *inference only* (no optimizer/scheduler) or for *training* (requires `setup_optimizer_scheduler`).
    *   **Training (`train`):** Runs training for a specified number of epochs per iteration. Processes training examples in batches, calculates `loss_pi` (Cross-Entropy on valid moves) and `loss_v` (Mean Squared Error), performs backpropagation, and steps the optimizer and learning rate scheduler.
    *   **Prediction (`predict`):** Takes a canonical board state (stacked history array) and a valid moves mask. Passes the board state through the network to get raw policy logits and a value prediction. **Crucially, it masks the policy logits using the valid moves mask *before* applying softmax** to ensure only legal moves have non-zero probability. Returns the probability distribution over the 4096 actions and the scalar value prediction.
    *   **Optimizer & Scheduler:** `setup_optimizer_scheduler` initializes an `Adam` optimizer and a learning rate scheduler (`CosineAnnealingLR` or `StepLR`) based on the `training` and `optimizer` sections of the configuration.
    *   **Saving/Loading:** `save_checkpoint` and `load_checkpoint` handle serializing/deserializing the network's weights, and optionally the optimizer and scheduler states, allowing training to be resumed. They also save/load relevant configuration subsets (`model_cfg`, `optimizer_cfg`, `training_cfg`).

### Monte Carlo Tree Search (`mcts.py`)

*   **Purpose:** MCTS performs simulations from a given board state to estimate the value and policy of moves more accurately than the raw NN output.
*   **`getActionProb`:** The main method called by Coach or Luna. It runs `numMCTSSims` simulations starting from a root board and history. It applies Dirichlet noise to the prior probabilities at the root node (if configured) for exploration during self-play. After simulations, it calculates move probabilities based on visit counts in the search tree, applies temperature sampling (higher temp early in games, deterministic later), and returns a policy vector over the 4096 canonical actions.
*   **`search`:** The recursive function for a single simulation.
    *   **Selection:** Traverses the tree from the root by selecting the child node (move) that maximizes the UCT score, balancing the estimated value (Q) and exploration bonus based on visit counts (N) and the NN's prior probability (P), using the `cpuct` constant.
    *   **Expansion:** When a leaf node (a state not yet expanded) is reached, it's expanded. The NN's `predict` method is called to get the prior policy (P) and value (V) for this state. This prior policy is stored, and edges are created for all legal moves from this state. The NN's value prediction is returned.
    *   **Backpropagation:** The value returned from the recursive call (or from the NN at an expanded leaf) is propagated back up the search path. The visit counts (N) for the visited nodes and edges are incremented, and the action values (Q) are updated as an average of the values seen during simulations passing through that edge.
    *   **State Caching:** Stores visited states and their associated search statistics (Qsa, Nsa, Ns, Ps, Es, Vs) using the board's FEN (`stringRepresentation`) as the key to avoid redundant computations.

### Training Orchestration (`coach.py`)

*   **`Coach` Class:** Manages the end-to-end training pipeline.
*   **`learn`:** The main training loop. Iterates for `numIters` times.
    *   **Load/Resume:** Checks config (`loading` section) to see if training should resume from saved examples and/or a model checkpoint. Determines the starting iteration. Loads `trainExamplesHistory` from the latest file. Loads model weights into `nnet` and `pnet`.
    *   **Self-Play:** Runs `numEps` games in parallel using `run_episode_worker` via `multiprocessing.Pool`. Each worker plays a full game, collecting (state, MCTS_policy, outcome, valids) tuples. These examples are added to `trainExamplesHistory`. History size is limited by `maxlenOfQueue` and `numItersForTrainExamplesHistory`.
    *   **Training:** Combines examples from recent iterations in `trainExamplesHistory`, shuffles them, and calls `nnet.train` for `epochs` epochs.
    *   **Arena:** Saves the newly trained model (`nnet`) to a temporary file (`temp.pth.tar`). Runs `arenaCompare` games in parallel using `run_arena_game_worker` via `multiprocessing.Pool`, pitting the new model against the previous best (`best.pth.tar`). Aggregates win/loss/draw counts.
    *   **Model Acceptance:** Compares the win rate (with draws counting as half wins) of the new model against the previous best model. If the win rate exceeds `updateThreshold` (or if `save_anyway` is true), the new model is accepted.
    *   **Save Model:** If the new model is accepted, it is saved as `best.pth.tar` (overwriting the previous best) and also as a dated checkpoint (`checkpoint_XXXX.pth.tar`).
    *   **Reload Model:** If the new model is rejected, the previous best model (`best.pth.tar`) is reloaded into `nnet` to ensure the next self-play iteration starts with the strongest accepted model.
    *   **Save Examples:** Saves the updated `trainExamplesHistory` after each iteration.
*   **`run_episode_worker`:** A function executed by self-play worker processes. Initializes its own Game, NNet, and MCTS instance, loads the current model weights, plays one game, calculates value targets (z) from the game outcome, and returns the collected training examples.
*   **`run_arena_game_worker`:** A function executed by arena worker processes. Initializes its own Game, two NNet instances (one for each model being compared), and two MCTS instances. Loads the specified previous (`pnet`) and new (`nnet`) model weights. Creates player functions that use these MCTS instances. Plays one game using the `Arena` class, and returns the outcome.

### Arena Games (`arena.py`)

*   **`Arena` Class:** A utility class specifically designed to play a *single* game *sequentially* between two provided player functions. It's used by `run_arena_game_worker` and `playground.py`. It handles applying moves and checking game end.
*   **`playGame`:** Runs one game. Takes two player functions (one for White, one for Black). Calls the appropriate player function for each turn to get the chosen action index, applies the move to the board, updates history, and checks for game end. Returns the game outcome relative to White.

### The Luna Engine (`luna.py`)

*   **`Luna` Class:** Provides a simplified interface to use the trained engine, primarily for playing against it (e.g., in the web GUI).
*   Initializes a `ChessGame`, `Luna_Network`, and `MCTS` instance. It loads a model based on the `inference` section of the configuration.
*   Maintains its own internal `chess.Board` and `history` deque representing the current game state for the GUI.
*   **`computer_move`:** Uses its MCTS instance to determine the best move (`temp=0` for deterministic play), translates the chosen canonical action index to an actual move, applies the move to its internal board, and updates its internal history. Returns the move in UCI format.
*   **`reset`:** Resets the internal `chess.Board` and `history` to the starting position.

### Web Interface (`index.html`, `selfplay.html`, `luna_html_wrapper.py`)

*   **Frontend (`.html` files):** Static HTML pages using `chessboard.js` and jQuery to display an interactive chess board in the user's browser. JavaScript code handles user clicks/drags, communicates with the backend via AJAX requests, and updates the board/status based on server responses.
*   **Backend (`luna_html_wrapper.py`):** A Flask web application.
    *   Initializes a global `Luna` engine instance (`htmlWrap`) upon startup, loading the inference model as configured.
    *   Defines API endpoints (`/`, `/move`, `/legal_moves_from`, `/selfplay`, `/next_selfplay_move`, `/reset`) that the frontend JavaScript interacts with.
    *   The route handlers process requests (e.g., human move UCI, request for legal moves, reset command), interact with the `htmlWrap` Luna engine instance to update the game state or get Luna's move, and return JSON responses containing the updated board state (FEN), Luna's move, game status, etc.
    *   Serves the `index.html` and `selfplay.html` template files, and static assets from the `static/` directory.

## Configuration (`src/config/config.yaml`)

The project uses `OmegaConf` for configuration management. Settings are primarily defined in `src/config/config.yaml`. Command-line arguments passed when running `main.py` or `luna_html_wrapper.py` can override settings in the YAML file.

Key sections in `config.yaml`:

*   **`game`**: Defines game-specific parameters, notably `history_len`.
*   **`training`**: Controls the self-play and training loop in `coach.py`. Includes `numIters`, `numEps`, `maxlenOfQueue`, `numItersForTrainExamplesHistory`, `num_workers`, `epochs`, and **learning rate scheduler type and parameters** (`lr_scheduler_type`, `lr_decay_step_size`, `lr_decay_gamma` if using StepLR, or `eta_min` if using CosineAnnealingLR).
*   **`mcts`**: Parameters for MCTS during **self-play** (`numMCTSSims`, `cpuct`, `tempThreshold`, `dir_noise`, `dirichlet_alpha`, `dirichlet_epsilon`).
*   **`arena`**: Parameters for the Arena comparison in `coach.py` (`arenaCompare`, `updateThreshold`, `save_anyway`).
*   **`model`**: Defines the neural network architecture parameters (`num_channels`, `num_res_blocks`, `dropout`).
*   **`optimizer`**: Parameters for the Adam optimizer (`lr`, `weight_decay`, `batch_size`).
*   **`loading`**: Controls loading checkpoints and training examples when running `main.py`.
*   **`inference`**: Parameters for MCTS and model loading when running the engine for **inference** (e.g., in the GUI or playground). These MCTS parameters (`numMCTSSims`, `cpuct`, `dir_noise`, etc.) are separate from the self-play MCTS parameters and are typically set for more deterministic, less exploratory search (e.g., lower `numMCTSSims`, `dir_noise: false`). It also specifies the `load_model`, `load_folder`, and `load_file` for the inference model used by `luna.py` and the web wrapper.

## How to Train the Model

Training is initiated via the `main.py` script and is configured primarily through `src/config/config.yaml`.

1.  **Edit `config.yaml`:** Adjust parameters in `src/config/config.yaml` to suit your hardware and desired training length (e.g., `numIters`, `numEps`, `epochs`, `num_workers`). Ensure the `loading` section is set correctly (`load_model: false`, `load_examples: false` for training from scratch, or `true` to resume).
    ***Important LR Scheduler Fix:*** As discussed, if you want a standard decaying learning rate, make sure `lr_scheduler_type: 'StepLR'` is set in the `training` section, along with appropriate `lr_decay_step_size` and `lr_decay_gamma` values. If you prefer the `CosineAnnealingLR`, the current setup will cause the LR to increase in the latter half of the epochs.

2.  **Start Training:** Run the `main.py` script from the project root directory.
    ```bash
    python src/main.py
    ```
    To override config settings via command line (e.g., just for a single run):
    ```bash
    python src/main.py training.numIters=10 mcts.numMCTSSims=100
    ```
    To resume training from saved examples and model (configure paths in `config.yaml` or via CLI):
    ```bash
    python src/main.py loading.load_model=true loading.load_examples=true
    ```

Training will save model checkpoints (`.pth.tar` files) and training examples (`.pkl` files) in the directory specified by `checkpoint_dir` in your config (defaults to `./temp/`). `best.pth.tar` will always store the model currently deemed the best by the Arena comparison.

## How to Play Against Luna (Web Interface)

Play against a trained Luna model using the included Flask web server.

1.  **Ensure Model Exists:** Make sure you have a trained model checkpoint file (e.g., `best.pth.tar` from training or a downloaded pretrained model) in the directory specified by `inference.load_folder` and `inference.load_file` in your `config.yaml`. Set `inference.load_model: true`.

2.  **Start the Web Server:** Run the `luna_html_wrapper.py` script from the project root directory.
    ```bash
    python src/luna_html_wrapper.py
    ```
    The script will load the specified inference model upon startup.

3.  **Open in Browser:** Open a web browser and navigate to:
    ```
    http://127.0.0.1:5000/
    ```

4.  **Choose Mode:** The interface allows you to:
    *   **Play as White:** You make the first move.
    *   **Play as Black:** Luna makes the first move (as White).
    *   **Watch Self-Play:** Navigate directly to `/selfplay` in your browser or click the button (if available on the index page).

## Web Interface Features

*   **Interactive Board:** Uses `chessboard.js` for draggable pieces and move visualization.
*   **Play as White/Black:** Standard human-vs-AI gameplay.
*   **Legal Move Highlighting:** Clicking a piece highlights squares it can legally move to (implemented via `/legal_moves_from` endpoint).
*   **Status Display:** Shows whose turn it is and game results.
*   **New Game/Back:** Buttons to reset the game state and return to the color selection screen.
*   **Self-Play Mode (`/selfplay`):**
    *   Automated game where Luna plays against itself using the loaded inference model.
    *   **Start Self Play / Stop:** Controls the automated move sequence.
    *   **Reset Board:** Resets the self-play game.
    *   Move history display.

## Advanced Usage

### Evaluating Against Stockfish

While the primary evaluation in the training loop is self-play Arena, you can use `playground.py` or custom scripts to pit your model against other engines like Stockfish.

```python
# Example conceptual usage (refer to playground.py or write a new script)
from luna.game.player import StockFishPlayer # Requires stockfish installed and in PATH
from luna.luna import Luna
from luna.game.arena import Arena
from omegaconf import OmegaConf # Assuming config is loaded

# Load the Luna model with inference config
cfg = OmegaConf.load("src/config/config.yaml")
# Ensure inference loading is true and path is correct in cfg or override via CLI
cfg.inference.load_model = True
# Resolve paths if not handled elsewhere, or ensure they are absolute
# cfg.inference.load_folder = "/path/to/your/models"
# cfg.inference.load_file = "best.pth.tar"

luna_engine = Luna(cfg=cfg) # This instance loads the inference model

# Initialize Stockfish (adjust parameters as needed)
stockfish_player = StockfishPlayer(elo=1800, skill_level=10, depth=10)

# Create player functions compatible with Arena (take board, history, return action index)
def luna_player_func(board, history):
    # Luna.computer_move handles board update internally, need to adjust
    # For Arena, the player func should just RETURN the action.
    # A dedicated wrapper might be needed for the Luna instance.
    # Or simply use the MCTS logic directly:
    current_player = luna_engine.game.who(board.turn)
    canonical_board = luna_engine.game.getCanonicalForm(board, current_player)
    canonical_history = luna_engine.game.getCanonicalHistory(history, current_player)
    pi_canonical = luna_engine.mcts.getActionProb(board, history, temp=0) # Use actual board/history for MCTS root
    action_canonical = np.argmax(pi_canonical)
    action_actual = action_canonical
    if current_player == -1:
        action_actual = luna_engine.game._flip_action_index(action_canonical)
    return action_actual


def stockfish_player_func(board, history):
     # StockfishPlayer.play takes board, returns action index
     return stockfish_player.play(board)

# Initialize the game object
game_instance = luna_engine.game # Reuse game instance from Luna

# Set up comparison using Arena (Luna as White, Stockfish as Black)
arena = Arena(luna_player_func, stockfish_player_func, game_instance)

print("Pitting Luna vs Stockfish...")
result = arena.playGame(verbose=True) # Play one game with verbose output

print("\nGame Result (relative to White - Luna):", result)
```

## Common Issues and Troubleshooting

- Configuration Loading Errors: Ensure config.yaml is correctly formatted and that paths specified (like checkpoint_dir, load_folder) are correct relative to where the script (main.py or luna_html_wrapper.py) is run from, or are absolute paths. The wrapper (luna_html_wrapper.py) includes logic to try and resolve paths relative to the project root.

- Checkpoint Not Found: Verify the paths in the loading and inference sections of your config.yaml. Check that the files (best.pth.tar, train_examples_iter_XXXX.pkl) actually exist in the specified folders.

- Learning Rate Increasing: As identified, this is due to the CosineAnnealingLR scheduler completing its cycle over the epochs within a single training iteration. To fix this and achieve standard decay, set lr_scheduler_type: 'StepLR' in the training section of config.yaml and configure lr_decay_step_size and lr_decay_gamma.

- CUDA Out of Memory: Reduce the batch_size in the optimizer section, the numMCTSSims in the mcts (for training) or inference (for GUI/playground) sections, or the num_workers in the training section.

- Training Not Improving: Adjust hyperparameters in config.yaml. Consider increasing numEps (more self-play games), numMCTSSims (deeper search), or experimenting with cpuct, dirichlet_alpha, and dirichlet_epsilon. If using StepLR, try different decay schedules (lr_decay_step_size, lr_decay_gamma).

- Web Interface "Not Found" or Errors: Ensure luna_html_wrapper.py is running. Check the server logs for error messages. Verify the browser console for JavaScript errors. Make sure the required static files (chessboard.min.css, .js, jquery.min.js, piece images) are in the static/ directory structure relative to where the wrapper script is run.

- Board Orientation Issues in GUI: The frontend JS manages orientation. Ensure the logic correctly handles player turn and potentially game end states. The /reset endpoint should return the board in the correct starting orientation based on the chosen color.

## Future Improvements

- Implement a more sophisticated learning rate schedule or warm-up phase.

- Improve board representation or add more feature planes.

- Enhance MCTS (e.g., adding time management, incorporating endgame knowledge).

- Explore multi-GPU training for faster iterations.

- Optimize the code for performance (e.g., using JIT compilation with TorchScript).

- Implement evaluation against standard chess benchmarks or ELO rating measurement.
