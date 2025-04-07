# Chess Reinforcement Learning

A high-performance self-learning chess AI model built using reinforcement learning through self-play, inspired by techniques like those used in AlphaZero. Optimized for maximum CPU and GPU utilization with containerized deployment.

## Features

- **High-Performance Reinforcement Learning**: The model learns by playing against itself with optimized resource usage
- **GPU-Accelerated Training**: Fully utilizes GPU resources through batch processing and mixed-precision training
- **Optimized Monte Carlo Tree Search**: Parallelized, memory-efficient implementation for better exploration
- **Containerized Deployment**: Separate containers for training and playing
- **Resource Monitoring**: Performance optimizations to ensure CPU and GPU are fully utilized
- **PyGame-based GUI**: Visual interface to play against the trained model
- **Flexible Configuration**: Easily tune parameters for different hardware configurations

## Project Structure

```
ChessRL/
│
├── models/                  # Directory to store trained models
│
├── ChessRL/                 # Package directory
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration parameters
│   ├── network.py           # Neural network architecture
│   ├── encoding.py          # Board encoding/decoding
│   ├── mcts.py              # Monte Carlo Tree Search implementation
│   ├── self_play.py         # Self-play logic
│   ├── replay_buffer.py     # Experience replay buffer
│   └── trainer.py           # Training logic
│
├── train.py                 # Main training script
├── play_gui.py              # GUI for playing against the trained model
├── Dockerfile.train         # Dockerfile for training container
├── Dockerfile.gui           # Dockerfile for GUI container
├── docker-compose.yml       # Docker Compose configuration
├── build_and_run.sh         # Convenience script for building and running containers
└── README.md                # Project documentation

## Requirements

- Python 3.7+ (3.11 recommended for best performance)
- PyTorch 2.0+
- CUDA-compatible GPU (for accelerated training)
- Docker & Docker Compose (for containerized deployment)
- Pygame (for GUI)

## Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/chess-rl.git
   cd chess-rl
   ```

2. Build the Docker containers:
   ```
   ./build_and_run.sh build
   ```

3. Run the training container:
   ```
   ./build_and_run.sh train
   ```

4. Run the GUI container to play against the trained model:
   ```
   ./build_and_run.sh play
   ```

### Manual Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/chess-rl.git
   cd chess-rl
   ```

2. Install dependencies:
   ```
   pip install -e .
   pip install pygame
   ```

3. Run training:
   ```
   python train.py --parallel --self-play-games 50
   ```

4. Play against the trained model:
   ```
   python play_gui.py --model chess_model_final.pt
   ```

## Performance Optimizations

This project includes several optimizations to maximize CPU and GPU utilization:

### Neural Network Optimizations
- **Mixed Precision Training**: Uses FP16 for faster computation on compatible GPUs
- **Memory-Efficient Operations**: Uses in-place operations and proper tensor management
- **TorchScript Compilation**: JIT compilation for faster inference
- **Batch Processing**: Properly batched operations for better GPU utilization

### MCTS Optimizations
- **Parallelized Tree Search**: Explores multiple paths simultaneously
- **Batched Node Evaluation**: Evaluates positions in batches for better GPU utilization
- **Memory Pooling**: Reuses allocated memory to reduce garbage collection overhead
- **Efficient Data Structures**: Uses `__slots__` and optimized containers

### Self-Play Optimizations
- **Multi-Processing**: Distributes self-play games across CPU cores
- **Thread Pool Management**: Optimizes worker thread allocation
- **Efficient Board Encoding**: Minimizes data conversion overhead
- **Progress Monitoring**: Tracks and reports throughput statistics

### System-Level Optimizations
- **Environment Variables**: Sets optimal thread limits for numerical libraries
- **CUDA Configuration**: Optimizes CUDA memory allocation and garbage collection
- **Container Resource Limits**: Properly allocates resources to containers

## Usage

### Training Options

You can customize training with various command-line options:

```
python train.py --iterations 100 --self-play-games 200 --mcts-simulations 32 --parallel
```

Key options:
- `--iterations`: Number of training iterations (default: 50)
- `--seed`: Random seed for reproducibility (default: 42)
- `--load`: Load a saved model to continue training
- `--self-play-games`: Number of self-play games per iteration (higher = better training)
- `--mcts-simulations`: Number of MCTS simulations per move (higher = stronger play)
- `--parallel`: Use parallel processing for faster self-play
- `--batch-size`: Training batch size (adjust based on available GPU memory)

### GUI Controls

The PyGame GUI offers the following controls:
- Click to select and move pieces
- Press 'R' to reset the game
- Press 'S' to switch sides (play as black/white)

## How It Works

1. **Self-Play**: The AI generates training data by playing games against itself using MCTS.
2. **Learning**: The neural network learns from these games, improving its ability to evaluate positions and choose moves.
3. **Iteration**: As training proceeds, the AI gets stronger and generates better training data.

### Neural Network Architecture

The neural network consists of:
- Input layers that encode the chess position (12 feature planes)
- Residual blocks for deep pattern recognition
- Policy head that predicts move probabilities (4672 possible moves)
- Value head that evaluates positions (from -1 to 1)

### Monte Carlo Tree Search

The MCTS algorithm combines:
- Exploration of new moves
- Exploitation of promising variations
- Neural network evaluations to guide the search
- UCB formula for balancing exploration vs. exploitation

## Docker Containers

The project uses two Docker containers:

1. **Training Container**:
   - Focused on optimized training performance
   - GPU-accelerated when available
   - Saves models to a shared volume

2. **GUI Container**:
   - Provides the PyGame interface
   - X11 forwarding for display
   - Loads models from the shared volume

## Datasets for pretraining:
- https://huggingface.co/datasets/angeluriot/chess_games

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This implementation is inspired by the AlphaZero algorithm by DeepMind.
- Thanks to the python-chess library for the chess engine.
- Performance optimizations inspired by EfficientZero and Leela Chess Zero.