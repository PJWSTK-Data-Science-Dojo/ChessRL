"""Web interface for playing against Luna."""

import time
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import tyro
from flask import Flask, jsonify, request
from loguru import logger

from luna.engine import Luna
from luna.game.state import LunaState

app = Flask(__name__)


class LunaHtmlWrapper(Luna):
    """Flask GUI wrapper for Luna-Chess."""

    def __init__(
        self,
        verbose: bool = False,
        device: str = "cuda",
        num_mcts_sims: int = 50,
        checkpoint_dir: str = "./temp/",
        checkpoint_file: str = "best.pth.tar",
    ) -> None:
        """Initialize web wrapper for Luna engine.

        Args:
            verbose: Enable verbose logging
            device: Compute device (cuda, mps, or cpu)
            num_mcts_sims: MCTS simulations per move (lower for faster response)
            checkpoint_dir: Model checkpoint directory
            checkpoint_file: Checkpoint filename
        """
        super().__init__(verbose, device, num_mcts_sims, checkpoint_dir, checkpoint_file)
        # Store for API endpoints
        self.num_mcts_sims = num_mcts_sims
        self.device = device

    def _make_state(self) -> LunaState:
        """Single source of truth: create a LunaState from self.board."""
        return LunaState(self.board)

    def _luna_respond(self) -> dict:
        """Have Luna make a move and return a JSON response dict."""
        if self.board.is_game_over():
            return {
                "status": "gameover",
                "message": get_game_result_message(self.board.outcome()),
                "fen": self.board.fen(),
            }

        # Track thinking time
        start_time = time.time()
        state = self._make_state()
        self.computer_move(state)
        think_time = time.time() - start_time

        result: dict = {
            "status": "success",
            "fen": self.board.fen(),
            "move": str(self.board.peek()),
            "think_time": round(think_time, 2),
        }
        if self.board.is_game_over():
            result["status"] = "gameover"
            result["message"] = get_game_result_message(self.board.outcome())
        return result


def get_game_result_message(outcome: chess.Outcome | None) -> str:
    """Return a message describing the game result."""
    if outcome is None:
        return "Game in progress"
    if outcome.winner is None:
        messages = {
            chess.Termination.STALEMATE: "Game over: Draw by stalemate",
            chess.Termination.INSUFFICIENT_MATERIAL: "Game over: Draw by insufficient material",
            chess.Termination.THREEFOLD_REPETITION: "Game over: Draw by threefold repetition",
            chess.Termination.FIFTY_MOVES: "Game over: Draw by fifty-move rule",
        }
        return messages.get(outcome.termination, "Game over: It's a draw!")
    return "Game over: White wins!" if outcome.winner else "Game over: Black wins!"


# Global instance - will be initialized in main()
htmlWrap: LunaHtmlWrapper | None = None


@app.route("/move", methods=["POST"])
def make_move():
    """Handle human move and respond with Luna's move."""
    assert htmlWrap is not None, "Luna engine not initialized. Run via main()."

    move_uci = request.form.get("move")

    if not move_uci:
        return {"status": "error", "message": "No move provided"}

    if htmlWrap.board.is_game_over():
        outcome = htmlWrap.board.outcome()
        return {"status": "gameover", "message": get_game_result_message(outcome), "fen": htmlWrap.board.fen()}

    try:
        move = chess.Move.from_uci(move_uci)
        if move not in htmlWrap.board.legal_moves:
            return {"status": "error", "message": f"Invalid move: {move_uci}"}

        htmlWrap.board.push(move)
        logger.info("[HUMAN MOVES] {}", move_uci)

        if htmlWrap.board.is_game_over():
            outcome = htmlWrap.board.outcome()
            return {"status": "gameover", "message": get_game_result_message(outcome), "fen": htmlWrap.board.fen()}

        return htmlWrap._luna_respond()
    except Exception as e:
        logger.exception("[MOVE ERROR]")
        return {"status": "error", "message": str(e)}


@app.route("/legal_moves", methods=["GET"])
def get_legal_moves():
    """Return the legal moves for the current position."""
    assert htmlWrap is not None, "Luna engine not initialized. Run via main()."

    legal_moves = [move.uci() for move in htmlWrap.board.legal_moves]
    return {"moves": legal_moves}


@app.route("/reset", methods=["POST"])
def reset_game():
    """Reset the game to the starting position."""
    assert htmlWrap is not None, "Luna engine not initialized. Run via main()."

    color = request.form.get("color", "white")
    full_reset = request.form.get("full_reset", "false") == "true"

    htmlWrap.board = chess.Board()

    if color == "black" and not full_reset:
        state = htmlWrap._make_state()
        htmlWrap.computer_move(state)

    return {"status": "success", "fen": htmlWrap.board.fen(), "full_reset": full_reset}


@app.route("/luna_move", methods=["POST"])
def luna_move():
    """Have Luna make a move."""
    assert htmlWrap is not None, "Luna engine not initialized. Run via main()."

    try:
        return htmlWrap._luna_respond()
    except Exception as e:
        logger.exception("[LUNA MOVE ERROR]")
        return {"status": "error", "message": str(e)}


@app.route("/")
def index():
    """Index page."""
    # Try improved version first, fall back to original
    improved_path = Path("src/index_improved.html")
    if improved_path.exists():
        return improved_path.read_text(encoding="utf-8")
    return Path("src/index.html").read_text(encoding="utf-8")


@app.route("/selfplay")
def selfplay():
    """Self play page."""
    assert htmlWrap is not None, "Luna engine not initialized. Run via main()."

    htmlWrap.board = chess.Board()
    # Try improved version first, fall back to original
    improved_path = Path("src/selfplay_improved.html")
    if improved_path.exists():
        return improved_path.read_text(encoding="utf-8")
    return Path("src/selfplay.html").read_text(encoding="utf-8")


@app.route("/next_move", methods=["POST"])
def next_move():
    """Have Luna make the next move in self-play mode."""
    assert htmlWrap is not None, "Luna engine not initialized. Run via main()."

    try:
        return htmlWrap._luna_respond()
    except Exception as e:
        logger.exception("[NEXT MOVE ERROR]")
        return {"status": "error", "message": str(e)}


@app.route("/next_selfplay_move", methods=["POST"])
def next_selfplay_move():
    """Make one move in self-play mode."""
    assert htmlWrap is not None, "Luna engine not initialized. Run via main()."

    try:
        result = htmlWrap._luna_respond()
        result["is_game_over"] = htmlWrap.board.is_game_over()
        return result
    except Exception as e:
        logger.exception("[SELFPLAY MOVE ERROR]")
        return {"status": "error", "message": str(e)}


@app.route("/move_coordinates")
def move_coordinates():
    """Handle move given as coordinate squares."""
    assert htmlWrap is not None, "Luna engine not initialized. Run via main()."

    if htmlWrap.is_game_over():
        logger.info("[GAME STATE] GAME IS OVER")
        return app.response_class(response="game over", status=200)

    source = int(request.args.get("from", default=""))
    target = int(request.args.get("to", default=""))
    promotion = request.args.get("promotion", default="") == "true"

    try:
        chess_move = chess.Move(source, target, promotion=chess.QUEEN if promotion else None)

        if chess_move in htmlWrap.board.legal_moves:
            if htmlWrap.verbose:
                logger.info("[HUMAN MOVES] {}", chess_move.uci())
            htmlWrap.board.push(chess_move)

            state = htmlWrap._make_state()
            htmlWrap.computer_move(state)
        else:
            if htmlWrap.verbose:
                logger.warning("[ILLEGAL MOVE] {} is not legal", chess_move.uci())
    except Exception:
        logger.exception("[MOVE ERROR]")

    return app.response_class(response=htmlWrap.board.fen(), status=200)


@app.route("/newgame")
def newgame():
    """Start a new game."""
    assert htmlWrap is not None, "Luna engine not initialized. Run via main()."

    htmlWrap.board.reset()
    return app.response_class(response=htmlWrap.board.fen(), status=200)


@app.route("/hint", methods=["GET"])
def get_hint():
    """Get Luna's suggested move for the current position."""
    assert htmlWrap is not None, "Luna engine not initialized. Run via main()."

    try:
        if htmlWrap.board.is_game_over():
            return jsonify({"status": "error", "message": "Game is over"})

        # Get Luna's suggested move using MCTS
        state = htmlWrap._make_state()
        from luna.game.chess_game import player_from_turn

        current_player = player_from_turn(state.board.turn)
        canonical_board = htmlWrap.game.get_canonical_form(state.board, current_player)

        # Get action probabilities without making a move
        action_probs = htmlWrap.mcts.get_action_prob(canonical_board, temp=0)
        best_action = int(np.argmax(action_probs))

        # Convert action to UCI move
        next_board, _ = htmlWrap.game.get_next_state(state.board, current_player, best_action)
        if len(next_board.move_stack) > len(state.board.move_stack):
            suggested_move = str(next_board.peek())
            confidence = float(action_probs[best_action])

            return jsonify(
                {
                    "status": "success",
                    "move": suggested_move,
                    "confidence": round(confidence * 100, 1),
                }
            )
        return jsonify({"status": "error", "message": "Could not determine hint"})

    except Exception as e:
        logger.exception("[HINT ERROR]")
        return jsonify({"status": "error", "message": str(e)})


@app.route("/model_info", methods=["GET"])
def get_model_info():
    """Get information about the loaded model."""
    assert htmlWrap is not None, "Luna engine not initialized. Run via main()."

    return jsonify(
        {
            "status": "success",
            "mcts_sims": htmlWrap.num_mcts_sims,
            "device": htmlWrap.device,
        }
    )


@app.route("/game_state", methods=["GET"])
def get_game_state():
    """Get current game state including FEN, legal moves, and game status."""
    assert htmlWrap is not None, "Luna engine not initialized. Run via main()."

    return jsonify(
        {
            "fen": htmlWrap.board.fen(),
            "legal_moves": [move.uci() for move in htmlWrap.board.legal_moves],
            "is_game_over": htmlWrap.board.is_game_over(),
            "turn": "white" if htmlWrap.board.turn else "black",
            "move_count": htmlWrap.board.fullmove_number,
        }
    )


@dataclass
class WebServeConfig:
    """Configuration for web server."""

    host: str = "127.0.0.1"
    """Host address to bind to"""

    port: int = 5000
    """Port number"""

    debug: bool = False
    """Enable Flask debug mode"""

    device: str = "cuda"
    """Compute device: cuda, mps, or cpu"""

    mcts_sims: int = 50
    """MCTS simulations per move (lower = faster response)"""

    checkpoint_dir: str = "./temp/"
    """Model checkpoint directory"""

    checkpoint_file: str = "best.pth.tar"
    """Checkpoint filename"""

    verbose: bool = True
    """Enable verbose logging"""


if __name__ == "__main__":
    cfg = tyro.cli(WebServeConfig)

    # Initialize global Luna instance with config
    htmlWrap = LunaHtmlWrapper(
        verbose=cfg.verbose,
        device=cfg.device,
        num_mcts_sims=cfg.mcts_sims,
        checkpoint_dir=cfg.checkpoint_dir,
        checkpoint_file=cfg.checkpoint_file,
    )

    logger.info(f"Starting Luna web server on {cfg.host}:{cfg.port}")
    logger.info(f"Using device: {cfg.device}")
    logger.info(f"MCTS simulations: {cfg.mcts_sims}")

    app.run(host=cfg.host, port=cfg.port, debug=cfg.debug)
