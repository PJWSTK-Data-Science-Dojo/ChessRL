import sys
import logging
import coloredlogs
from omegaconf import DictConfig, OmegaConf
import chess.svg
import base64
import traceback
import os
from collections import deque
import chess
from flask import Flask, request, jsonify, render_template
import torch

# Ensure correct imports from the refactored luna package
# Assuming luna package is in src/luna/
from luna.luna import Luna
from luna.game.luna_game import who, _flip_action_index
# Add path utility import if needed, but let's implement path resolution locally

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

# Global variable for Luna engine instance, initialized later in __main__
htmlWrap = None

# Global variable for Flask app, initialized later in __main__
app = None # Initialize as None

# --- Helper function to determine game result message ---
def get_game_result_message(outcome):
    """Return a message describing the game result"""
    if outcome is None:
        return "Game in progress"
    if outcome.winner is None:
        # Handle draw types
        if outcome.termination == chess.Termination.STALEMATE:
            return "Game over: Draw by stalemate"
        elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
            return "Game over: Draw by insufficient material"
        elif outcome.termination == chess.Termination.THREEFOLD_REPETITION:
            return "Game over: Draw by threefold repetition"
        elif outcome.termination == chess.Termination.FIFTY_MOVES:
            return "Game over: Draw by fifty-move rule"
        elif outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
             return "Game over: Draw by fivefold repetition"
        elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
             return "Game over: Draw by seventy-five move rule"
        else:
            return "Game over: It's a draw!" # Generic draw
    elif outcome.winner == chess.WHITE:
        return "Game over: White wins!"
    elif outcome.winner == chess.BLACK:
        return "Game over: Black wins!"
    else:
         return "Game over: Unknown outcome!"


# --- Route Handler Functions (actual logic) ---
# These functions will be called by the route wrappers defined inside __main__

def make_move():
    """Handle human move and respond with Luna's move"""
    global htmlWrap # Access the global Luna instance

    if htmlWrap is None:
         log.error("/move called but Luna engine failed to initialize.")
         return jsonify({"status": "error", "message": "Engine not available."})

    move_uci = request.form.get("move")
    current_board = htmlWrap.board

    if not move_uci:
        log.warning("Received empty move UCI.")
        return jsonify({"status": "error", "message": "No move provided"})

    # Check game over state *before* attempting to push move
    if current_board.is_game_over():
        outcome = current_board.outcome(claim_draw=True)
        log.info(f"Move {move_uci} received but game is already over. Outcome: {outcome}")
        return jsonify({
            "status": "gameover",
            "message": get_game_result_message(outcome),
            "fen": current_board.fen(),
            "is_game_over": True,
            "move": None # No move was made
        })

    try:
        # Attempt to parse and apply the human move
        # Note: The frontend sends UCI, python-chess handles parsing
        move = current_board.parse_uci(move_uci) # Use board.parse_uci to handle promotions correctly

        # This check should technically be redundant if parse_uci and is_legal cover it,
        # but it's a good final validation.
        if move not in current_board.legal_moves:
             log.warning(f"Illegal human move attempted: {move_uci} on board {current_board.fen()}")
             return jsonify({"status": "error", "message": f"Invalid move: {move_uci}"})

        # Push the human move to Luna's internal board
        htmlWrap.board.push(move)

        # Update history after the human move
        board_array_after_human_move = htmlWrap.game._board_to_feature_array(htmlWrap.board)
        htmlWrap.history.append(board_array_after_human_move) # Append the new board state to history

        log.info(f"[HUMAN MOVES] Applied move {move.uci()}. New board: {htmlWrap.board.fen()}")

        # Check if the game is over immediately after the human move
        if htmlWrap.board.is_game_over():
            outcome = htmlWrap.board.outcome(claim_draw=True)
            log.info(f"Game over after human move. Outcome: {outcome}")
            return jsonify({
                "status": "gameover",
                "message": get_game_result_message(outcome),
                "fen": htmlWrap.board.fen(),
                "is_game_over": True,
                "move": move.uci() # Return the human's move UCI
            })

        log.info("Human move legal, game not over. Luna thinking...")
        # Have Luna make a move using its current state (board, history)
        luna_move_uci = htmlWrap.computer_move()

        # After Luna's move, check if the game is over
        is_game_over_after_luna = htmlWrap.board.is_game_over()
        outcome_after_luna = None
        if is_game_over_after_luna:
             outcome_after_luna = htmlWrap.board.outcome(claim_draw=True)
             log.info(f"Game over after Luna move. Outcome: {outcome_after_luna}")


        if luna_move_uci is None:
             # This is an error state - Luna failed to make a legal move
             log.error("Luna failed to return a valid move after human move.")
             outcome = htmlWrap.board.outcome(claim_draw=True) # Get outcome if any
             msg = "Luna failed to make a move."
             if outcome: msg += " " + get_game_result_message(outcome)
             return jsonify({
                  "status": "error", # Or 'gameover' if the board state is actually terminal after Luna's attempt? Error is safer.
                  "message": msg,
                  "fen": htmlWrap.board.fen(), # Return board state as is
                  "is_game_over": is_game_over_after_luna,
                  "move": None # Indicate Luna's move failed
             })


        # If we reach here, Luna made a move successfully
        if is_game_over_after_luna:
             # Game is over after Luna's move
             return jsonify({
                 "status": "gameover",
                 "message": get_game_result_message(outcome_after_luna),
                 "fen": htmlWrap.board.fen(),
                 "move": luna_move_uci, # Return Luna's successful move UCI
                 "is_game_over": True
             })
        else:
            # Game is still in progress after Luna's move
            log.info("Luna move legal, game not over.")
            return jsonify({
                "status": "success",
                "fen": htmlWrap.board.fen(),
                "move": luna_move_uci, # Return Luna's successful move UCI
                "is_game_over": False
            })

    except ValueError as e:
        # python-chess parse_uci or other move errors
        log.warning(f"Error processing human move UCI {move_uci}: {e}. Board: {current_board.fen()}")
        return jsonify({"status": "error", "message": f"Invalid move format: {e}"})
    except Exception as e:
        log.error(f"Unexpected error during /move request for move {move_uci}: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)})

def get_legal_moves_from():
    """Return the target squares for legal moves from a given source square."""
    global htmlWrap # Access the global Luna instance

    if htmlWrap is None:
         log.error("/legal_moves_from called but Luna engine failed to initialize.")
         return jsonify({"status": "error", "message": "Engine not available.", "moves": []})

    source_square_str = request.args.get("square")

    current_board = htmlWrap.board
    # Determine the current player based on the board's turn
    current_player_color = who(current_board.turn) # +1 for white, -1 for black

    if not source_square_str:
        log.warning("No source square provided for legal_moves_from.")
        return jsonify({"status": "error", "message": "No square provided", "moves": []})

    try:
        # Parse the source square string (e.g., 'e2') into a square index (0-63)
        source_square = chess.parse_square(source_square_str)
    except ValueError:
        log.warning(f"Invalid square format received: {source_square_str}")
        return jsonify({"status": "error", "message": f"Invalid square: {source_square_str}", "moves": []})

    # Check if it's the human's turn based on the stored GUI orientation
    # This is a client-side check based on orientation, server provides true turn.
    # Server will validate color/turn correctly when move is submitted.
    # This check here is primarily to avoid unnecessary computation/response.
    human_player_color = 1 if htmlWrap.currentOrientation == 'white' else -1
    if current_player_color != human_player_color:
         log.warning(f"/legal_moves_from called for wrong turn. Board turn: {who(current_board.turn)}, GUI orientation: {htmlWrap.currentOrientation}")
         return jsonify({"status": "error", "message": "It's not your turn.", "moves": []})


    piece = current_board.piece_at(source_square)
    if not piece:
        log.warning(f"No piece at source square {source_square_str} for legal_moves_from.")
        return jsonify({"status": "error", "message": "No piece at that square", "moves": []})

    # Check if the piece color matches the current player's turn color
    piece_color = 1 if piece.color == chess.WHITE else -1
    if piece_color != current_player_color:
        log.warning(f"Piece at {source_square_str} ({piece.symbol()}) does not belong to current player ({who(current_board.turn)}).")
        return jsonify({"status": "error", "message": "Piece is not yours", "moves": []})

    # Get all legal moves for the current board state
    all_legal_moves = list(current_board.legal_moves)

    legal_target_squares = []
    # Iterate through legal moves to find those starting from the source square
    for move in all_legal_moves:
        if move.from_square == source_square:
            legal_target_squares.append(chess.square_name(move.to_square)) # Add the target square name

    # Remove duplicates (shouldn't be any for a given from_square in legal_moves, but safe)
    # Also handle promotions - multiple promotion moves (e.g., e7e8q, e7e8r) will map to the same target square ('e8').
    # The frontend just needs to know the target square is valid.
    legal_target_squares = list(set(legal_target_squares))
    log.info(f"Legal moves from {source_square_str}: {legal_target_squares}")

    return jsonify({"status": "success", "message": "Legal moves fetched", "moves": legal_target_squares})

def reset_game():
    """Reset the game to the starting position and handle initial Luna move if human plays black."""
    global htmlWrap # Access the global Luna instance

    if htmlWrap is None:
         log.error("/reset called but Luna engine failed to initialize.")
         return jsonify({"status": "error", "message": "Engine not available."})

    # Determine player color from form data (sent from frontend buttons)
    color = request.form.get("color", "white") # Default to white if not specified (e.g. full_reset)
    # Check for full_reset flag if needed to differentiate simple color switch vs full reset
    full_reset_flag = request.form.get("full_reset", "false").lower() == "true"

    log.info(f"Resetting game. Player color: {color}, Full reset: {full_reset_flag}")

    # Reset Luna's internal board and history
    htmlWrap.reset()
    htmlWrap.currentOrientation = color # Store the human player's chosen color/orientation

    initial_fen = htmlWrap.board.fen()

    # If human plays as black, Luna (White) makes the first move
    if color == "black":
        log.info("Human playing as black. Luna (White) making first move.")
        # Have Luna compute and make its first move
        luna_first_move_uci = htmlWrap.computer_move() # Luna is Player 1 (White) on initial board

        if luna_first_move_uci is None:
             log.error("Luna failed to make first move when human is black.")
             # Return error state, but still provide the board FEN
             return jsonify({
                  "status": "error",
                  "message": "Luna failed to make initial move.",
                  "fen": htmlWrap.board.fen(), # Return the board state after failed attempt
                  "is_game_over": htmlWrap.board.is_game_over(),
                  "move": None
             })

        # If Luna made a move, check if the game is over (unlikely after 1 move)
        is_game_over_after_luna = htmlWrap.board.is_game_over()
        outcome_after_luna = None
        if is_game_over_after_luna:
             outcome_after_luna = htmlWrap.board.outcome(claim_draw=True)
             log.info(f"Game over after Luna's first move. Outcome: {outcome_after_luna}")


        log.info(f"Luna made first move {luna_first_move_uci} for white player. New board: {htmlWrap.board.fen()}")
        # Return the board state after Luna's move
        return jsonify({
            "status": "success",
            "message": "Luna made the first move.",
            "fen": htmlWrap.board.fen(),
            "move": luna_first_move_uci,
            "is_game_over": is_game_over_after_luna
        })

    # If human plays as white, return the standard starting board
    log.info("Human playing as white. Returning initial board.")
    return jsonify({
        "status": "success",
        "message": "Game reset, your turn (White).",
        "fen": initial_fen, # Standard starting FEN
        "is_game_over": htmlWrap.board.is_game_over(), # Should be False
        "move": None
    })

def selfplay():
    """Serve the self-play page"""
    global htmlWrap # Access the global Luna instance

    log.info("Accessing self-play page.")
    # Reset the board state when the self-play page is accessed
    if htmlWrap is not None:
         htmlWrap.reset()
         htmlWrap.currentOrientation = 'white' # Default orientation for self-play display
         log.info("Self-play board state reset.")
    else:
         log.warning("Cannot reset board for self-play, Luna engine not initialized.")

    try:
        # Use the Flask render_template function to serve the HTML file
        return render_template("selfplay.html")
    except Exception as e:
        log.error(f"Error serving selfplay.html: {e}", exc_info=True)
        return "Error loading self-play page", 500

def next_selfplay_move():
    """Have Luna make the next move in self-play mode"""
    global htmlWrap # Access the global Luna instance

    if htmlWrap is None:
         log.error("/next_selfplay_move called but Luna engine failed to initialize.")
         return jsonify({"status": "error", "message": "Engine not available."})

    current_board = htmlWrap.board

    # Check game over state before making a move
    if current_board.is_game_over():
        outcome = current_board.outcome(claim_draw=True)
        log.info(f"next_selfplay_move called but game is over. Outcome: {outcome}")
        return jsonify({
            "status": "gameover",
            "message": get_game_result_message(outcome),
            "fen": current_board.fen(),
            "is_game_over": True,
            "move": None
        })

    try:
        log.info("Luna thinking for self-play move...")
        # Have Luna compute and make the next move in the sequence
        luna_move_uci = htmlWrap.computer_move() # This updates htmlWrap.board and htmlWrap.history

        # After Luna's move, check if the game is over
        is_game_over_after_luna = htmlWrap.board.is_game_over()
        outcome_after_luna = None
        if is_game_over_after_luna:
            outcome_after_luna = htmlWrap.board.outcome(claim_draw=True)
            log.info(f"Self-play game over after move {luna_move_uci}. Outcome: {outcome_after_luna}")

        if luna_move_uci is None:
             # This is an error state - Luna failed to make a legal move
             log.error("Luna failed to make a move in self-play.")
             outcome = htmlWrap.board.outcome(claim_draw=True) # Get outcome if any
             msg = "Luna failed to make a move in self-play."
             if outcome: msg += " " + get_game_result_message(outcome)
             return jsonify({
                  "status": "error", # Or 'gameover' if terminal
                  "message": msg,
                  "fen": htmlWrap.board.fen(), # Return board state as is
                  "is_game_over": is_game_over_after_luna,
                  "move": None
             })


        # If we reach here, Luna made a move successfully
        if is_game_over_after_luna:
             # Game is over after Luna's move
             return jsonify({
                 "status": "gameover",
                 "message": get_game_result_message(outcome_after_luna),
                 "fen": htmlWrap.board.fen(),
                 "move": luna_move_uci, # Return Luna's successful move UCI
                 "is_game_over": True
             })
        else:
             # Game is still in progress
             log.info(f"Self-play move made: {luna_move_uci}. New board: {htmlWrap.board.fen()}")
             return jsonify({
                 "status": "success",
                 "fen": htmlWrap.board.fen(),
                 "move": luna_move_uci, # Return Luna's successful move UCI
                 "is_game_over": False
             })

    except Exception as e:
        log.error(f"Unexpected error during /next_selfplay_move: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "fen": htmlWrap.board.fen(), # Return current board state
            "is_game_over": htmlWrap.board.is_game_over(), # Current state's game over status
            "move": None
        })


def index():
    """Serve the index page"""
    global htmlWrap # Access the global Luna instance

    log.info("Accessing index page.")
    # Optionally reset the board state when the index page is accessed,
    # or rely on the /reset endpoint called by the frontend buttons.
    # Relying on the frontend buttons calling /reset is better for state management.
    # htmlWrap.reset() # Avoid resetting here directly

    try:
        # Use the Flask render_template function to serve the HTML file
        return render_template("index.html")
    except Exception as e:
        log.error(f"Error serving index.html: {e}", exc_info=True)
        return "Error loading index page", 500


# --- Main Execution Block ---
if __name__ == "__main__":
    # Configure Flask to find templates and static files relative to the script's directory
    # Determine the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine the template and static folders relative to the script directory
    template_folder = script_dir # Templates (index.html, selfplay.html) are in src/
    static_folder = os.path.join(script_dir, 'static') # Static files are in src/static/

    # Initialize Flask app with explicit template and static folders
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

    # Define routes *after* the Flask app instance `app` is created.
    # These are wrappers that call the actual handler functions defined above.
    @app.route("/move", methods=["POST"])
    def make_move_route():
        return make_move()

    @app.route("/legal_moves_from", methods=["GET"])
    def get_legal_moves_from_route():
        return get_legal_moves_from()

    @app.route("/reset", methods=["POST"])
    def reset_game_route():
        return reset_game()

    @app.route("/selfplay")
    def selfplay_route():
        return selfplay()

    # Route for the self-play next move endpoint
    @app.route("/next_selfplay_move", methods=["POST"])
    def next_selfplay_move_route():
         return next_selfplay_move()

    @app.route("/")
    def index_route():
        return index()


    # --- Load Configuration and Initialize Luna Engine ---
    # This block runs only once when the script is executed directly.
    try:
        # Get command line arguments (e.g., 'inference.load_model=true')
        cli_args = OmegaConf.from_cli(sys.argv[1:])

        # Determine the absolute path to the base config file (config.yaml)
        # Assuming config.yaml is in src/config/ relative to the script (in src/)
        config_path_relative_to_script = 'config/config.yaml'
        config_path_abs = os.path.join(script_dir, config_path_relative_to_script)

        # Load the base configuration from the file
        config = OmegaConf.load(config_path_abs)

        # Merge base config with command line arguments (CLI args override file config)
        # The result `cfg` is the effective configuration for this run.
        cfg: DictConfig = OmegaConf.merge(config, cli_args)

        # Add CUDA availability status to the configuration
        cfg.cuda = torch.cuda.is_available()

        # --- Path Resolution ---
        # Paths specified in config.yaml (like checkpoint_dir, load_folder) might be
        # relative to the project root, which is one level up from the 'src' directory.
        # Resolve these paths to be absolute paths based on the project root.
        # Assuming project root is the parent directory of the script directory ('src/').
        try:
             # Calculate the absolute path to the project root
             project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
             log.info(f"Resolved project root: {project_root}")

             # Define a helper function to resolve paths relative to the project root
             def resolve_relative_to_project_root(cfg_path: str | None) -> str | None:
                 if cfg_path is None:
                     return None # Return None if the path is None
                 if os.path.isabs(cfg_path):
                     # If the path is already absolute, use it as is
                     return cfg_path
                 # Assume simple relative paths like './temp/' or './pretrained_models/' are relative to project root
                 # Handle potential '..' in paths if necessary, but simple join usually works
                 return os.path.abspath(os.path.join(project_root, cfg_path))

             # Apply resolution to relevant path configurations
             # Coach related paths (used by Coach, but stored in the main cfg)
             if hasattr(cfg, 'checkpoint_dir'):
                cfg.checkpoint_dir = resolve_relative_to_project_root(cfg.checkpoint_dir)
             if hasattr(cfg, 'loading') and cfg.loading is not None:
                 if hasattr(cfg.loading, 'load_folder') and cfg.loading.load_folder is not None:
                    cfg.loading.load_folder = resolve_relative_to_project_root(cfg.loading.load_folder)
                 if hasattr(cfg.loading, 'load_examples_folder') and cfg.loading.load_examples_folder is not None:
                    cfg.loading.load_examples_folder = resolve_relative_to_project_root(cfg.loading.load_examples_folder)

             # Inference related paths (used by Luna instance for loading)
             if hasattr(cfg, 'inference') and cfg.inference is not None:
                 if hasattr(cfg.inference, 'load_folder') and cfg.inference.load_folder is not None:
                    cfg.inference.load_folder = resolve_relative_to_project_root(cfg.inference.load_folder)
                 # load_file is just a filename, doesn't need path resolution


        except Exception as path_error:
            log.error(f"Error resolving config paths relative to project root in wrapper: {path_error}. Using paths as specified in config, which might be incorrect.", exc_info=True)
            # Fallback: Use paths directly from cfg, which might be incorrect if relative to project root


        log.info("Effective Configuration for Luna Wrapper:\n%s", OmegaConf.to_yaml(cfg))

        # Initialize the global Luna engine instance with the merged configuration.
        # Luna's __init__ will read its specific settings (like model loading) from the provided cfg.
        htmlWrap = Luna(cfg=cfg, verbose=True)
        log.info("Luna engine initialized for HTML wrapper with potentially loaded model.")

    except Exception as e:
        log.error(f"FATAL ERROR: Failed to initialize Luna engine in __main__: {e}", exc_info=True)
        htmlWrap = None # Ensure htmlWrap is None if initialization failed


    log.info("Starting Flask server for Luna Chess GUI...")
    # Start the Flask development server.
    # WARNING: Do not use the development server in a production environment.
    app.run(debug=False, host='0.0.0.0', port=5000)