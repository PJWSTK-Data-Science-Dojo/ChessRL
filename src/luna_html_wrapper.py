""" 
    An Html wrapper for Luna
"""

from flask import Flask, Response, request, render_template
import chess.svg
import base64
from luna.luna import Luna
from luna.game.state import LunaState
import traceback
import os
app = Flask(__name__)

class LunaHtmlWrapper(Luna):
    """Html GUI for Luna-Chess using flask"""
    """Luna gives us:
        1. luNNa
        2. luna_eval
        3. board
    """

    def __init__(self, verbose=False) -> None:
        # Init Luna
        super().__init__(verbose)

    def board_to_svg(self, s):
        """Parse board into svg so we can use it in UI"""
        return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')
    
def get_game_result_message(outcome):
    """Return a message describing the game result"""
    if outcome is None:
        return "Game in progress"
    if outcome.winner is None:
        # Determine draw reason
        if outcome.termination == chess.Termination.STALEMATE:
            return "Game over: Draw by stalemate"
        elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
            return "Game over: Draw by insufficient material"
        elif outcome.termination == chess.Termination.THREEFOLD_REPETITION:
            return "Game over: Draw by threefold repetition"
        elif outcome.termination == chess.Termination.FIFTY_MOVES:
            return "Game over: Draw by fifty-move rule"
        else:
            return "Game over: It's a draw!"
    elif outcome.winner:
        return "Game over: White wins!"
    else:
        return "Game over: Black wins!"

htmlWrap = LunaHtmlWrapper(verbose=True)

@app.route("/move", methods=["POST"])
def make_move():
    """Handle human move and respond with Luna's move"""
    move_uci = request.form.get("move")
    
    if not move_uci:
        return {"status": "error", "message": "No move provided"}
    
    # Check if game is already over
    if htmlWrap.board.is_game_over():
        outcome = htmlWrap.board.outcome()
        return {
            "status": "gameover",
            "message": get_game_result_message(outcome),
            "fen": htmlWrap.board.fen()
        }
    
    try:
        # Apply human move
        move = chess.Move.from_uci(move_uci)
        if move not in htmlWrap.board.legal_moves:
            return {"status": "error", "message": f"Invalid move: {move_uci}"}
            
        htmlWrap.board.push(move)
        print(f"[HUMAN MOVES] {move_uci}")
        
        # Check if game is over after human move
        if htmlWrap.board.is_game_over():
            outcome = htmlWrap.board.outcome()
            return {
                "status": "gameover",
                "message": get_game_result_message(outcome),
                "fen": htmlWrap.board.fen()
            }
        
        # Update board state for Luna - ensure we're sending the correct state
        state = LunaState(htmlWrap.board)
        
        # Have Luna respond with a move
        # The Luna engine already knows which side to play based on board.turn
        htmlWrap.computer_move(state, htmlWrap.luna_eval)
        
        # Check if game is over after Luna's move
        if htmlWrap.board.is_game_over():
            outcome = htmlWrap.board.outcome()
            return {
                "status": "gameover",
                "message": get_game_result_message(outcome),
                "fen": htmlWrap.board.fen(),
                "move": str(htmlWrap.board.peek())
            }
        
        # Return the updated board state
        return {
            "status": "success",
            "fen": htmlWrap.board.fen(),
            "move": str(htmlWrap.board.peek())
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.route("/legal_moves", methods=["GET"])
def get_legal_moves():
    """Return the legal moves for the current position"""
    board = htmlWrap.board
    legal_moves = [move.uci() for move in board.legal_moves]
    return {"moves": legal_moves}

@app.route("/reset", methods=["POST"])
def reset_game():
    """Reset the game to the starting position"""
    color = request.form.get("color", "white")
    full_reset = request.form.get("full_reset", "false") == "true"
    
    htmlWrap.board = chess.Board()
    htmlWrap.board_state = LunaState(htmlWrap.board)
    
    # If human plays as black, have Luna make the first move
    if color == "black" and not full_reset:
        state = htmlWrap.board_state
        htmlWrap.computer_move(state, htmlWrap.luna_eval)
    
    return {
        "status": "success",
        "fen": htmlWrap.board.fen(),
        "full_reset": full_reset
    }

@app.route("/luna_move", methods=["POST"])
def luna_move():
    """Have Luna make a move"""
    if htmlWrap.board.is_game_over():
        outcome = htmlWrap.board.outcome()
        return {
            "status": "gameover",
            "message": get_game_result_message(outcome),
            "fen": htmlWrap.board.fen()
        }
    
    try:
        # Have Luna make a move
        state = LunaState(htmlWrap.board)
        htmlWrap.computer_move(state, htmlWrap.luna_eval)
        
        # Check if game is over after Luna's move
        if htmlWrap.board.is_game_over():
            outcome = htmlWrap.board.outcome()
            return {
                "status": "gameover",
                "message": get_game_result_message(outcome),
                "fen": htmlWrap.board.fen(),
                "move": str(htmlWrap.board.peek())
            }
        
        return {
            "status": "success",
            "fen": htmlWrap.board.fen(),
            "move": str(htmlWrap.board.peek())
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.route("/")
def index():
    """Index page"""
    html = open("src/index.html").read()
    return html

@app.route("/selfplay")
def selfplay():
    """Self play page"""
    # Reset the board state
    htmlWrap.board = chess.Board()
    htmlWrap.board_state = LunaState(htmlWrap.board)
    html = open("src/selfplay.html").read()
    return html

@app.route("/next_move", methods=["POST"])
def next_move():
    """Have Luna make the next move in self-play mode"""
    if htmlWrap.board.is_game_over():
        outcome = htmlWrap.board.outcome()
        return {
            "status": "gameover",
            "message": get_game_result_message(outcome),
            "fen": htmlWrap.board.fen()
        }
    
    try:
        # Have Luna make a move
        state = LunaState(htmlWrap.board)
        htmlWrap.computer_move(state, htmlWrap.luna_eval)
        
        # Check if game is over after Luna's move
        if htmlWrap.board.is_game_over():
            outcome = htmlWrap.board.outcome()
            return {
                "status": "gameover",
                "message": get_game_result_message(outcome),
                "fen": htmlWrap.board.fen(),
                "move": str(htmlWrap.board.peek())
            }
            
        return {
            "status": "success",
            "fen": htmlWrap.board.fen(),
            "move": str(htmlWrap.board.peek())
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e)
        }

@app.route("/next_selfplay_move", methods=["POST"])
def next_selfplay_move():
    """Make one move in self-play mode"""
    # If game is over, return the result
    if htmlWrap.board.is_game_over():
        outcome = htmlWrap.board.outcome()
        return {
            "status": "gameover",
            "message": get_game_result_message(outcome),
            "fen": htmlWrap.board.fen()
        }
    
    # Make a move with Luna
    state = LunaState(htmlWrap.board)
    try:
        htmlWrap.computer_move(state, htmlWrap.luna_eval)
        
        # Return the updated state
        return {
            "status": "success",
            "fen": htmlWrap.board.fen(),
            "move": str(htmlWrap.board.peek()),
            "is_game_over": htmlWrap.board.is_game_over()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# move given in algebraic notation
@app.route("/move")
def move():
    if not htmlWrap.is_game_over():
        move = request.args.get('move',default="")
        if move is not None and move != "":
            if htmlWrap.verbose: print(f"[HUMAN MOVES] {move}")
            
            try:
                # Check if move is legal before making it
                chess_move = chess.Move.from_uci(move) if len(move) == 4 else chess.Move.from_san(move)
                if chess_move in htmlWrap.board_state.board.legal_moves:
                    htmlWrap.board_state.board.push(chess_move)
                    htmlWrap.board = htmlWrap.board_state.board  # Keep them in sync
                    htmlWrap.computer_move(htmlWrap.board_state, htmlWrap.luna_eval)
                else:
                    if htmlWrap.verbose: print(f"[ILLEGAL MOVE] {move} is not legal")
            except Exception as e:
                traceback.print_exc()
                if htmlWrap.verbose: print(f"[MOVE ERROR] {e}")
            
            response = app.response_class(
                response=htmlWrap.board.fen(),
                status=200
            )
            return response
    else:
        if htmlWrap.verbose: print("[GAME STATE] GAME IS OVER")
        response = app.response_class(
        response="game over",
        status=200
        )
        return response
    
    if htmlWrap.verbose: print("[FUNCTION CALLS] luna_html_wrapper.py.index() ran")
    return index()

# moves given as coordinates of piece moved
@app.route("/move_coordinates")
def move_coordinates():
    if not htmlWrap.is_game_over():
        source = int(request.args.get('from', default=''))
        target = int(request.args.get('to', default=''))
        promotion = True if request.args.get('promotion', default='') == 'true' else False

        try:
            # Create the move
            chess_move = chess.Move(source, target, promotion=chess.QUEEN if promotion else None)
            
            # Check if move is legal
            if chess_move in htmlWrap.board_state.board.legal_moves:
                if htmlWrap.verbose: print(f"[HUMAN MOVES] {chess_move.uci()}")
                
                # Make the move
                htmlWrap.board_state.board.push(chess_move)
                htmlWrap.board = htmlWrap.board_state.board  # Keep them in sync
                
                # Let computer respond
                htmlWrap.computer_move(htmlWrap.board_state, htmlWrap.luna_eval)
            else:
                if htmlWrap.verbose: print(f"[ILLEGAL MOVE] {chess_move.uci()} is not legal")
        except Exception as e:
            traceback.print_exc()
            if htmlWrap.verbose: print(f"[MOVE ERROR] {e}")
        
        response = app.response_class(
            response=htmlWrap.board.fen(),
            status=200
        )
        return response

    if htmlWrap.verbose: print("[GAME STATE] GAME IS OVER")
    response = app.response_class(
        response="game over",
        status=200
    )
    return response

@app.route("/newgame")
def newgame():
    htmlWrap.board.reset()
    htmlWrap.board_state = LunaState()  # Reset state
    response = app.response_class(
        response=htmlWrap.board.fen(),
        status=200
    )
    return response

if __name__ == "__main__":
    app.run()