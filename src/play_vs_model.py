"""Play chess against a trained Luna model via command-line interface.

Usage:
    uv run python src/play_vs_model.py --checkpoint ./temp/best.pth.tar --device cpu --mcts-sims 25
"""

import sys
from dataclasses import dataclass

import chess
import tyro
from loguru import logger

from luna.config import EzV2LearnerConfig, MCTSParams
from luna.game.chess_game import ChessGame, action_to_move, mirror_move
from luna.mcts import MCTS
from luna.network import LunaNetwork


@dataclass
class PlayConfig:
    """Configuration for playing against the model."""

    checkpoint: str = "./temp/best.pth.tar"
    """Path to model checkpoint"""

    device: str = "cpu"
    """Compute device: cuda, mps, or cpu"""

    mcts_sims: int = 25
    """Number of MCTS simulations per move (higher = stronger but slower)"""

    num_games: int = 1
    """Number of games to play"""

    human_plays_white: bool = True
    """If True, human plays White; otherwise human plays Black"""

    log_level: str = "INFO"
    """Logging level"""


def print_board(board: chess.Board, human_is_white: bool) -> None:
    """Print the chess board to console with coordinates."""
    if human_is_white:
        # Show from white's perspective (rank 8 at top)
        print("\n  a b c d e f g h")
        for rank in range(7, -1, -1):
            rank_str = f"{rank + 1} "
            for file in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece is None:
                    rank_str += ". "
                else:
                    rank_str += piece.symbol() + " "
            rank_str += f"{rank + 1}"
            print(rank_str)
        print("  a b c d e f g h\n")
    else:
        # Show from black's perspective (rank 1 at top)
        print("\n  h g f e d c b a")
        for rank in range(8):
            rank_str = f"{rank + 1} "
            for file in range(7, -1, -1):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece is None:
                    rank_str += ". "
                else:
                    rank_str += piece.symbol() + " "
            rank_str += f"{rank + 1}"
            print(rank_str)
        print("  h g f e d c b a\n")


def get_human_move(board: chess.Board) -> chess.Move:
    """Get a valid move from the human player."""
    while True:
        try:
            print(f"Legal moves: {', '.join(move.uci() for move in board.legal_moves)}")
            move_str = input("Your move (UCI notation, e.g., e2e4): ").strip()
            if move_str.lower() in ["quit", "exit", "q"]:
                sys.exit(0)
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                return move
            else:
                print(f"Illegal move: {move}. Try again.")
        except ValueError:
            print(f"Invalid move format: {move_str}. Use UCI notation (e.g., e2e4)")


def play_game(
    game_obj: ChessGame,
    model_mcts: MCTS,
    human_is_white: bool,
) -> float:
    """Play one game and return result from human's perspective.

    Returns:
        1.0 if human wins, -1.0 if model wins, small value for draw
    """
    board = game_obj.get_init_board()
    move_count = 0

    print("\n" + "=" * 60)
    print("Starting new game!")
    print(f"You are playing {'White' if human_is_white else 'Black'}")
    print("Enter moves in UCI notation (e.g., e2e4)")
    print("Type 'quit' to exit")
    print("=" * 60)

    while True:
        move_count += 1
        print_board(board, human_is_white)

        current_player = 1 if board.turn == chess.WHITE else -1
        human_player = 1 if human_is_white else -1

        # Check if game is over
        result = game_obj.get_game_ended(board, current_player)
        if abs(result) > 1e-8:
            print_board(board, human_is_white)
            if result > 0:
                winner = "White" if board.turn == chess.WHITE else "Black"
                print(f"\nGame over! {winner} wins!")
                if current_player == human_player:
                    print("Congratulations! You won!")
                    return 1.0
                else:
                    print("Model wins. Better luck next time!")
                    return -1.0
            else:
                print("\nGame over! Draw.")
                return 0.0

        if current_player == human_player:
            # Human's turn
            print(f"\nMove {move_count} - Your turn ({'White' if human_is_white else 'Black'})")
            move = get_human_move(board)
        else:
            # Model's turn
            print(f"\nMove {move_count} - Model's turn ({'White' if not human_is_white else 'Black'})")
            print("Thinking...")

            # Get canonical form for the model (always from current player's perspective)
            canonical_board = game_obj.get_canonical_form(board, current_player)

            # Run MCTS to get best move
            policy, _value = model_mcts.search_latent(canonical_board, temp=0.0)
            action = int(max(range(len(policy)), key=lambda a: policy[a]))

            # Convert action index directly to chess move
            # This is more reliable than comparing board states
            move = action_to_move(action)
            if not board.turn:  # Black to move - need to unmirror
                move = mirror_move(move)

            # Verify move is legal (should always be true with legal masking)
            if move not in board.legal_moves:
                # This shouldn't happen with legal masking, but be defensive
                logger.warning(f"Model selected illegal move {move.uci()}, picking best legal alternative")
                # Pick the legal move with highest policy probability
                legal_actions = [a for a, p in enumerate(policy) if policy[a] > 0]
                if legal_actions:
                    action = max(legal_actions, key=lambda a: policy[a])
                    move = action_to_move(action)
                    if not board.turn:
                        move = mirror_move(move)
                else:
                    # Last resort
                    move = list(board.legal_moves)[0]

            print(f"Model plays: {move.uci()}")

        # Execute the move
        board.push(move)


def main() -> int:
    """Main entry point."""
    cfg = tyro.cli(PlayConfig)

    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level.upper())

    # Load game
    logger.info("Loading chess game...")
    game = ChessGame()

    # Configure learner for inference only
    learner_cfg = EzV2LearnerConfig(
        device=cfg.device,
        compile_inference=False,  # Disable compile for inference
    )

    # Load network
    logger.info(f"Loading model from {cfg.checkpoint} on device {cfg.device}...")
    nnet = LunaNetwork(game, learner_cfg)
    try:
        nnet.load_checkpoint(folder="", filename=cfg.checkpoint)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        logger.error("Make sure the checkpoint path is correct and was trained with a compatible version")
        return 1

    # Create MCTS for model
    mcts_params = MCTSParams(
        num_mcts_sims=cfg.mcts_sims,
        cpuct=1.25,
        dir_noise=False,  # No exploration noise during play
    )
    model_mcts = MCTS(game, nnet, mcts_params)

    # Play games
    wins = 0
    losses = 0
    draws = 0

    for game_num in range(cfg.num_games):
        logger.info(f"Game {game_num + 1}/{cfg.num_games}")
        result = play_game(game, model_mcts, cfg.human_plays_white)

        if result > 0.5:
            wins += 1
        elif result < -0.5:
            losses += 1
        else:
            draws += 1

        if game_num + 1 < cfg.num_games:
            response = input("\nPlay another game? (y/n): ").strip().lower()
            if response != "y":
                break

    print("\n" + "=" * 60)
    print("Session Summary:")
    print(f"  Wins:   {wins}")
    print(f"  Losses: {losses}")
    print(f"  Draws:  {draws}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
