"""
    python-chess luna wrapper
"""

from __future__ import print_function
import numpy as np
import chess
import logging # Import logging
from collections import deque # Import deque for history
from omegaconf import DictConfig, OmegaConf # Import OmegaConf types

log = logging.getLogger(__name__) # Setup logger for this module
EPS = 1e-8 # Small epsilon for numerical stability

# --- Helper function for action flipping ---
def _flip_action_index(action_index):
    """Flips a 4096 action index vertically."""
    if action_index < 0 or action_index >= 4096:
        # Handle invalid index if necessary, though it shouldn't happen for legal moves
        log.error(f"Attempted to flip invalid action index: {action_index}")
        return action_index # Or raise an error

    from_sq = action_index // 64
    to_sq = action_index % 64
    flipped_from = chess.square_mirror(from_sq)
    flipped_to = chess.square_mirror(to_sq)
    return flipped_from * 64 + flipped_to
# -----------------------------------------

# Helper Functions (Keep as is)
def to_np(board: chess.Board): # This seems unused now? toArray is the primary method.
    a = [0] * (8*8*6)
    for sq, pc in board.piece_map().items():
        a[sq * 6 + pc.piece_type - 1] = 1 if pc.color else -1
    return np.array(a)

def from_move(move: chess.Move):
    # Map potentially promoting move to its base action index (without promotion info)
    # Assumes 4096 action space based on from/to squares only.
    return move.from_square * 64 + move.to_square

def to_move(action):
    # Convert 4096 action index back to a base move (promotion needs handling elsewhere)
    to_sq = action % 64
    from_sq = int(action / 64)
    return chess.Move(from_sq, to_sq)

def who(turn: bool):
  """Who is playing, 1 for white -1 for black"""
  return 1 if turn else -1

def mirror_move(move: chess.Move): # Keep this helper
  return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), promotion=move.promotion)

# Game Outcomes
CHECKMATE = 1
STALEMATE = 2
INSUFFICIENT_MATERIAL = 3
SEVENTYFIVE_MOVES = 4
FIVEFOLD_REPETITION = 5
FIFTY_MOVES = 6
THREEFOLD_REPETITION = 7

class ChessGame():
    """python-chess wrapper - Updated for Conv2D NN, History, Symmetry, Fixes"""

    # Constants for Feature Normalization
    MAX_TOTAL_MOVES = 200
    MAX_NO_PROGRESS_COUNT = 100

    # --- Updated __init__ for History ---
    def __init__(self, cfg: DictConfig | None = None): # Accept optional config
        super(ChessGame, self).__init__()
        # Update Number of Input Planes (remains 22 features per board state)
        self.num_feature_planes = 22
        self.history_len = cfg.get('game', {}).get('history_len', 8) # Use .get with default dict
        # Total channels = features_per_state * history_frames
        self.num_input_channels = self.num_feature_planes * self.history_len
        log.info(f"ChessGame initialized with history_len={self.history_len}, num_input_channels={self.num_input_channels}")

    def getInitBoard(self):
        """Returns initial board and an empty history deque."""
        board = chess.Board()
        history = deque(maxlen=self.history_len)
        initial_array = self._board_to_feature_array(board)
        # Fill history with the initial state
        for _ in range(self.history_len):
            history.append(initial_array) # Store array directly
        return board, history

    def getBoardSize(self):
        """Board Dimensions for NN input (Channels, Height, Width)"""
        # This now reflects the history stacking
        return (self.num_input_channels, 8, 8)

    def _board_to_feature_array(self, board: chess.Board):
        """Internal helper to serialize a single board to feature planes."""
        plane_shape = (self.num_feature_planes, 8, 8)
        state = np.zeros(plane_shape, dtype=np.float32)
        piece_map = board.piece_map()

        # Plane Index Definitions
        P_WHITE_PAWN = 0
        P_WHITE_KNIGHT = 1
        P_WHITE_BISHOP = 2
        P_WHITE_ROOK = 3
        P_WHITE_QUEEN = 4
        P_WHITE_KING = 5
        P_BLACK_PAWN = 6
        P_BLACK_KNIGHT = 7
        P_BLACK_BISHOP = 8
        P_BLACK_ROOK = 9
        P_BLACK_QUEEN = 10
        P_BLACK_KING = 11
        P_WHITE_COLOR = 12 # 1.0 if white to move, 0.0 otherwise
        P_CASTLE_WK = 13
        P_CASTLE_WQ = 14
        P_CASTLE_BK = 15
        P_CASTLE_BQ = 16
        P_MOVE_COUNT = 17
        P_NO_PROGRESS = 18
        P_REP_COUNT_2 = 19
        P_REP_COUNT_3 = 20
        P_EN_PASSANT = 21

        # Planes 0-11: Piece Locations
        for sq, pc in piece_map.items():
            rank, file = chess.square_rank(sq), chess.square_file(sq)
            plane_idx = pc.piece_type - 1
            if pc.color == chess.BLACK:
                plane_idx += 6
            state[plane_idx, rank, file] = 1

        # Plane 12: Player Color
        if board.turn == chess.WHITE:
            state[P_WHITE_COLOR, :, :] = 1.0

        # Planes 13-16: Castling Rights
        if board.has_kingside_castling_rights(chess.WHITE): state[P_CASTLE_WK, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): state[P_CASTLE_WQ, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): state[P_CASTLE_BK, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): state[P_CASTLE_BQ, :, :] = 1.0

        # Plane 17: Total Move Count (Normalized)
        normalized_move_count = min(1.0, board.ply() / self.MAX_TOTAL_MOVES)
        state[P_MOVE_COUNT, :, :] = normalized_move_count

        # Plane 18: No-Progress Count (Normalized)
        normalized_no_progress = min(1.0, board.halfmove_clock / self.MAX_NO_PROGRESS_COUNT)
        state[P_NO_PROGRESS, :, :] = normalized_no_progress

        # Planes 19-20: Repetition Flags
        # is_repetition(2) checks for *any* repetition of 2 or more times
        if board.is_repetition(2): state[P_REP_COUNT_2, :, :] = 1.0
        # can_claim_threefold_repetition is a distinct check
        if board.can_claim_threefold_repetition(): state[P_REP_COUNT_3, :, :] = 1.0


        # Plane 21: En Passant Target Square
        if board.ep_square is not None:
            rank, file = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
            state[P_EN_PASSANT, rank, file] = 1.0

        return state

    # --- Updated toArray for History ---
    def toArray(self, history: deque):
        """Stack history frames to create the NN input tensor."""
        # History deque contains the last `history_len` feature arrays
        # Ensure history has the expected length (it should due to initialization/updates)
        if len(history) != self.history_len:
            # This might happen if called incorrectly, handle defensively
            log.warning(f"History length mismatch ({len(history)} vs {self.history_len}). Padding/truncating.")
            # Example: Pad with oldest state or zeros if too short
            padded_history = list(history)
            while len(padded_history) < self.history_len:
                padded_history.insert(0, padded_history[0] if padded_history else np.zeros((self.num_feature_planes, 8, 8), dtype=np.float32))
            # Replace the original deque if it was modified (e.g., passed by reference)
            # history.clear()
            # history.extend(padded_history)
            history = deque(padded_history, maxlen=self.history_len) # Create a new deque just in case


        # Stack along the channel dimension (axis=0)
        # The deque stores frames chronologically (oldest first if appended normally)
        # AlphaZero stacks T, T-1, T-2 ... T-N+1 where T is current.
        # Assuming history deque has [T-N+1, ..., T-1, T]
        stacked_state = np.concatenate(list(history), axis=0)

        # Verify shape
        expected_shape = (self.num_input_channels, 8, 8)
        if stacked_state.shape != expected_shape:
             log.error(f"Final stacked state shape mismatch! Got {stacked_state.shape}, expected {expected_shape}. History length: {len(history)}")
             # Handle error: maybe return zeros or raise exception
             return np.zeros(expected_shape, dtype=np.float32)

        return stacked_state

    def getActionSize(self):
        """Number of actions possible (64*64 for from-to square pairs)"""
        return 64*64

 # --- CORRECTED getNextState for 4096 action space and promotions ---
    def getNextState(self, board: chess.Board, player: int, action: int, history: deque | None):
        """
        Get next state (board, player, history) given current board, player, action, and optional history.
        Handles the 4096 action space including promotion variants (defaults to Queen).
        If history is None, returns None for next_history.
        If the action is illegal, returns (original_board, original_player, original_history).
        """
        # Check that the player matches the board's turn
        assert who(board.turn) == player, f"Player mismatch: board.turn is {board.turn}, but player is {player}. Board: {board.fen()}"

        # Convert the action index (0-4095) to a basic move (from_square, to_square)
        from_square = action // 64
        to_square = action % 64

        next_board = board.copy()
        move_to_push = None
        illegal_move_detected = False

        try:
            # Determine the piece type at the 'from' square
            piece_at_from = board.piece_at(from_square)

            # Check if this is a pawn move to the 8th/1st rank (promotion rank)
            is_promotion_move = False
            if piece_at_from and piece_at_from.piece_type == chess.PAWN:
                if (piece_at_from.color == chess.WHITE and chess.square_rank(to_square) == 7) or \
                   (piece_at_from.color == chess.BLACK and chess.square_rank(to_square) == 0):
                    is_promotion_move = True

            if is_promotion_move:
                # For promotion moves in the 4096 space, default to Queen promotion
                move_to_push = chess.Move(from_square, to_square, promotion=chess.QUEEN)
            else:
                # For non-promotion moves, use the basic move
                move_to_push = chess.Move(from_square, to_square)

            # Check if the constructed move is legal on the current board
            if move_to_push in board.legal_moves:
                next_board.push(move_to_push)
                # If the move was a promotion and Queen wasn't legal (highly unlikely for a valid from/to to promo rank)
                # or if it was a promotion move that wasn't a queen promotion in the 4096 space, we might have issues.
                # With a 4096 space, the policy doesn't distinguish promotion type. We MUST default. Queen is standard.
                # The check 'move_to_push in board.legal_moves' handles if the from/to square pair is valid *at all*,
                # including if it implies a promotion to Queen.
            else:
                # The constructed move (either basic or Queen promotion) is not legal.
                # This means the selected action index (from_square, to_square)
                # does not correspond to *any* legal move (basic or any promotion).
                # This indicates an illegal move was selected by MCTS/Policy.
                log.error(f"Illegal action/move attempted in getNextState: action={action} -> from={from_square}, to={to_square}. Constructed move {move_to_push.uci()} is not legal. Board: {board.fen()}")
                illegal_move_detected = True

        except ValueError as e:
            # Catch potential errors during move parsing (should be rare if action index is 0-4095)
            log.error(f"Error creating move from action {action} ({from_square}, {to_square}): {e}. Board: {board.fen()}")
            illegal_move_detected = True
        except Exception as e:
            # Catch any other unexpected errors during move processing
            log.error(f"Unexpected error processing action {action}: {e}. Board: {board.fen()}", exc_info=True)
            illegal_move_detected = True


        if illegal_move_detected:
            # Return the original board state, original player, and original history.
            # The calling code (Coach/MCTS) should check for this pattern.
            return (board, player, history) # Pass original history back

        # If we reached here, a legal move (`move_to_push`) was successfully identified and pushed to `next_board`.

        # Update history if it was provided (MCTS calls will pass None)
        next_history = None
        if history is not None:
            # Create a new deque from the provided history (copy)
            next_history = deque(history, maxlen=self.history_len)
            # Get the feature array for the *new* board state
            next_board_array = self._board_to_feature_array(next_board)
            # Append the new board state array to the history
            next_history.append(next_board_array)

        # Return the new board, the next player, and the updated history (or None)
        return (next_board, who(next_board.turn), next_history)


    def getValidMoves(self, board: chess.Board, player: int):
        """Return a fixed size binary vector indicating valid moves (64*64 space)."""
        # Check that the player matches the board's turn
        current_player = who(board.turn)
        if current_player != player:
             # This should NOT happen if MCTS calls this correctly with player=1 on a canonical board.
             # If it does happen, it indicates a logic error upstream.
             log.warning(f"getValidMoves called with player={player} but board.turn implies player={current_player}. Board FEN: {board.fen()}. Proceeding, but this might be wrong.")
             # We will proceed using the legal moves from the board's perspective,
             # which should match `current_player`. The `player` argument here
             # seems intended to match the perspective (canonical=1), not necessarily
             # the actual turn if the board isn't canonical. Let's rely on board.legal_moves.

        valids = np.zeros(self.getActionSize(), dtype=np.int32)
        try:
            # Get legal moves for the *current* turn indicated by the board object
            # This is crucial: board.legal_moves includes all valid promotion variants (e.g., e7e8q, e7e8r, etc.)
            legal_moves_for_board = board.legal_moves
        except Exception as e:
            # Catch potential errors in python-chess's legal_moves generation
            log.error(f"Error getting legal moves for board {board.fen()}: {e}")
            return valids # Return all zeros on error

        # Iterate through all actual legal moves provided by python-chess
        for move in legal_moves_for_board:
            # Map each legal move (including promotions) to its corresponding 64*64 action index
            # from_move(move) correctly strips the promotion info for this mapping.
            action_index = from_move(move)

            # Ensure the calculated index is within the expected range
            if 0 <= action_index < self.getActionSize():
                 # Mark this from_square -> to_square index as valid.
                 # If multiple legal moves (e.g., different promotions) map to the same index,
                 # this index will be marked valid as long as *at least one* is legal.
                 valids[action_index] = 1
            else:
                 # This should ideally not happen with valid chess.Move objects
                 log.warning(f"Calculated action_index {action_index} for move {move.uci()} is out of bounds [0, {self.getActionSize()-1}]")

        # --- Optional: Verify valids vs actual legal moves ---
        # This is a debugging step, can be removed later
        # actual_valid_indices = {from_move(m) for m in legal_moves_for_board}
        # calculated_valid_indices = {i for i, v in enumerate(valids) if v == 1}
        # if actual_valid_indices != calculated_valid_indices:
        #     log.error(f"Mismatch between calculated valids vector and actual legal moves! Board: {board.fen()}")
        #     log.debug(f"Actual legal move indices: {sorted(list(actual_valid_indices))}")
        #     log.debug(f"Calculated valid indices: {sorted(list(calculated_valid_indices))}")
        # ----------------------------------------------------

        return valids

    def getGameEnded(self, board: chess.Board, player: int) -> float:
        """Check game end, return outcome relative to 'player'."""
        # Use claim_draw=True for 50-move, 3-fold checks
        outcome = board.outcome(claim_draw=True)

        if outcome is not None:
            if outcome.winner == chess.WHITE: return 1.0 * player
            if outcome.winner == chess.BLACK: return -1.0 * player
            # If outcome is not None but winner is None, it's a draw by one of the standard rules
            return 1e-4 # Draw (use a small non-zero value)

        # Check other draw conditions not always caught by outcome (e.g., before claim or less common)
        # Note: outcome(claim_draw=True) *should* cover most standard draws now (stalemate, insufficient material, 75-move, 5-fold, 3-fold, 50-move).
        # Keeping these explicit checks provides a fallback, but might be redundant with claim_draw=True.
        if board.is_stalemate(): return 1e-4
        if board.is_insufficient_material(): return 1e-4
        if board.is_seventyfive_moves(): return 1e-4 # Stricter than 50-move claim
        # is_fivefold_repetition is handled by outcome(claim_draw=True) if the position history is available
        # is_threefold_repetition is handled by outcome(claim_draw=True) if the position history is available

        return 0.0 # Game not ended

    # --- Ensure getCanonicalForm is correct ---
    def getCanonicalForm(self, board: chess.Board, player: int):
        """Return board perspective for the current 'player' (1=white, -1=black)."""
        # The canonical form for AlphaZero is always from White's perspective.
        # So, if the current player is Black (-1), we need to mirror the board.
        if player == 1: # Current player is White
            # The board should already be from White's perspective (board.turn == chess.WHITE)
            # If board.turn is Black, something is wrong upstream or the board wasn't passed correctly.
            # Asserting here helps catch that.
            assert board.turn == chess.WHITE, f"getCanonicalForm called for player 1 but board turn is {board.turn}. Board: {board.fen()}"
            return board # Already in canonical form (White's perspective)
        else: # Current player is Black (-1)
             # The board should be from Black's perspective (board.turn == chess.BLACK)
             # Asserting here helps catch that.
             assert board.turn == chess.BLACK, f"getCanonicalForm called for player -1 but board turn is {board.turn}. Board: {board.fen()}"
             # Mirror the board to get White's perspective
             return board.mirror()
        
            # --- NEW getCanonicalHistory method ---
    def getCanonicalHistory(self, history: deque, player: int) -> deque:
        """
        Returns a new history deque where each board state array is canonicalized
        relative to the player specified, ensuring consistent input for the NN.
        If player is -1 (Black), each board state array in the history is flipped.
        """
        canonical_history = deque(maxlen=self.history_len)
        # Iterate through the history queue (oldest to newest)
        for board_array in history:
            # Create a canonical version of the board array based on the *current* player's perspective
            if player == 1: # If current player is White, history states are already from their perspective
                canonical_history.append(board_array) # Append original array
            else: # If current player is Black, flip each state in history to get White's perspective
                flipped_array = self._flip_feature_array(board_array) # Use the flip helper
                canonical_history.append(flipped_array) # Append flipped array

        return canonical_history
    # --- END NEW getCanonicalHistory ---

    # --- NEW _flip_feature_array helper method ---
    def _flip_feature_array(self, board_array: np.ndarray) -> np.ndarray:
        """
        Flips a single board state feature array vertically.
        Assumes input shape (num_feature_planes, 8, 8).
        Handles piece planes and special planes correctly.
        """
        if board_array.shape != (self.num_feature_planes, 8, 8):
             log.error(f"Attempted to flip feature array with incorrect shape: {board_array.shape}")
             return board_array # Return original on error

        # Flip positional planes vertically (ranks)
        # Planes 0-11 (Pieces) and Plane 21 (En Passant square) encode positions and should be flipped.
        planes_to_flip = list(range(12)) + [21]
        flipped_array = board_array.copy() # Make a copy to avoid modifying original array
        flipped_array[planes_to_flip, :, :] = np.flip(board_array[planes_to_flip, :, :], axis=1) # Flip vertically on ranks

        # Planes that encode global state or relative positions not tied to specific ranks should *not* be flipped.
        # Plane 12 (Player Color): Global flag, should not be flipped.
        # Planes 13-16 (Castling Rights): Existence of rights, not position. Should not be flipped.
        # Plane 17 (Move Count), Plane 18 (No Progress): Global counts. Should not be flipped.
        # Planes 19, 20 (Repetition Flags): Global flags. Should not be flipped.

        # These planes are already copied by flipped_array = board_array.copy()
        # We just ensured the others *were* flipped. So no need to restore them explicitly.

        # Example check: Ensure non-flipped planes are identical to original
        # assert np.array_equal(flipped_array[12,:,:], board_array[12,:,:]) # Should be true
        # assert np.array_equal(flipped_array[13:21,:,:], board_array[13:21,:,:]) # Should be true for these planes

        return flipped_array
    # --- END NEW _flip_feature_array ---

    def getSymmetries(self, board: chess.Board, pi: np.ndarray):
        """Augment training data with vertical board flip symmetry."""
        # pi is the policy vector (numpy array, size 4096) corresponding to the input board (must be canonical)

        # Original (no symmetry)
        symmetries = [(board, pi)]

        # Vertical Flip Symmetry
        # Check if pi is a valid numpy array of the correct size
        if isinstance(pi, np.ndarray) and pi.shape == (self.getActionSize(),):
            flipped_board = board.mirror() # Mirror the board state (this flips colors and ranks)
            pi_flipped = np.zeros_like(pi)

            # Iterate through all possible action indices (0-4095)
            for action_index in range(self.getActionSize()):
                # If the original policy has probability for this action...
                if pi[action_index] > 1e-8: # Use a small epsilon to avoid floating point issues
                    # Calculate the corresponding action index on the flipped board
                    flipped_action = _flip_action_index(action_index)

                    # Ensure the flipped index is valid (should always be for 64x64 space)
                    if 0 <= flipped_action < self.getActionSize():
                         # Assign the original probability to the flipped action index
                         pi_flipped[flipped_action] = pi[action_index]
                    else:
                         # This should not happen with the _flip_action_index logic for 64x64
                         log.warning(f"Flipped action index {flipped_action} out of bounds for original action {action_index}")

            # Note: Renormalization after flipping *might* be necessary if there are floating point issues
            # or if some actions map outside the 4096 space (which shouldn't happen here).
            # Let's add a check and re-normalize if needed.
            sum_pi_original = np.sum(pi)
            sum_pi_flipped = np.sum(pi_flipped)
            if abs(sum_pi_original - sum_pi_flipped) > 1e-6:
                 log.warning(f"Symmetry policy sum mismatch: original={sum_pi_original}, flipped={sum_pi_flipped}. Renormalizing flipped.")
                 if sum_pi_flipped > EPS: # Avoid division by zero
                     pi_flipped /= sum_pi_flipped
                 else:
                     log.error(f"Flipped policy sum is zero after symmetry. Cannot normalize.")
                     pi_flipped = np.zeros_like(pi_flipped) # Set to zero if sum is zero


            symmetries.append((flipped_board, pi_flipped))
        else:
             # This warning indicates an issue upstream where pi is not a valid numpy array of size 4096
             log.warning(f"Invalid policy vector 'pi' passed to getSymmetries. Type: {type(pi)}, Shape: {getattr(pi, 'shape', 'N/A')}")


        return symmetries

# --- CORRECTED stringRepresentation ---
    def stringRepresentation(self, board: chess.Board):
        """Unique string representation for MCTS cache keys."""
        # Use FEN alone for the state key.
        # FEN captures piece positions, turn, castling rights, and en passant square.
        # Repetition and 50-move rule are handled by getGameEnded using the board's internal history stack.
        return board.fen()
    # --- END CORRECTED stringRepresentation ---

    @staticmethod
    def display(board):
        print(board)