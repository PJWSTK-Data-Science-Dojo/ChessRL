"""Web interface for playing against Luna."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import tyro
from flask import Flask, request
from loguru import logger

from luna.engine import Luna
from luna.game.state import LunaState

app = Flask(__name__)


class LunaHtmlWrapper(Luna):
    """Flask GUI wrapper for Luna-Chess."""

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(verbose)

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

        state = self._make_state()
        self.computer_move(state)

        result: dict = {"status": "success", "fen": self.board.fen(), "move": str(self.board.peek())}
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


htmlWrap = LunaHtmlWrapper(verbose=True)


@app.route("/move", methods=["POST"])
def make_move():
    """Handle human move and respond with Luna's move."""
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
    legal_moves = [move.uci() for move in htmlWrap.board.legal_moves]
    return {"moves": legal_moves}


@app.route("/reset", methods=["POST"])
def reset_game():
    """Reset the game to the starting position."""
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
    try:
        return htmlWrap._luna_respond()
    except Exception as e:
        logger.exception("[LUNA MOVE ERROR]")
        return {"status": "error", "message": str(e)}


@app.route("/")
def index():
    """Index page."""
    return Path("src/index.html").read_text(encoding="utf-8")


@app.route("/selfplay")
def selfplay():
    """Self play page."""
    htmlWrap.board = chess.Board()
    return Path("src/selfplay.html").read_text(encoding="utf-8")


@app.route("/next_move", methods=["POST"])
def next_move():
    """Have Luna make the next move in self-play mode."""
    try:
        return htmlWrap._luna_respond()
    except Exception as e:
        logger.exception("[NEXT MOVE ERROR]")
        return {"status": "error", "message": str(e)}


@app.route("/next_selfplay_move", methods=["POST"])
def next_selfplay_move():
    """Make one move in self-play mode."""
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
    htmlWrap.board.reset()
    return app.response_class(response=htmlWrap.board.fen(), status=200)


@dataclass
class WebServeConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False


if __name__ == "__main__":
    cfg = tyro.cli(WebServeConfig)
    app.run(host=cfg.host, port=cfg.port, debug=cfg.debug)
