from typing import Tuple, Union

import chess
import torch


PIECE_MAP = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6
}


FILE_MAP = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}


def generate_uci_move_map():
    """
    Generate UCI mapping from string to int. Used for compressing move list of the dataset.
    """
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
    # Create a dictionary to store all possible UCI move combinations
    uci_move_map = {}
    # Generate all possible combinations of start and end squares
    integer = 0
    for start_file in files:
        for start_rank in ranks:
            start_square = start_file + start_rank
            for end_file in files:
                for end_rank in ranks:
                    end_square = end_file + end_rank
                    # Skip if start and end squares are the same
                    if start_square != end_square:
                        move_notation = start_square + end_square
                        # Add to the dictionary with a placeholder value
                        # You can replace None with any value useful for your application
                        uci_move_map[move_notation] = integer
                        integer += 1
    # Special cases for promotions (e.g., e7e8q for pawn promotion to queen)
    promotion_pieces = ['q', 'r', 'b', 'n']
    # Handle pawn promotions
    for file in files:
        # White pawn promotions (from 7th rank to 8th rank)
        for target_file in [file] + ([file_left for file_left in files if abs(files.index(file) - files.index(file_left)) == 1 and files.index(file_left) < 8]):
            if target_file in files:
                start_square = file + '7'
                end_square = target_file + '8'

                for piece in promotion_pieces:
                    promotion_move = start_square + end_square + piece
                    uci_move_map[promotion_move] = integer
                    integer += 1
        # Black pawn promotions (from 2nd rank to 1st rank)
        for target_file in [file] + ([file_left for file_left in files if abs(files.index(file) - files.index(file_left)) == 1 and files.index(file_left) < 8]):
            if target_file in files:
                start_square = file + '2'
                end_square = target_file + '1'

                for piece in promotion_pieces:
                    promotion_move = start_square + end_square + piece
                    uci_move_map[promotion_move] = integer
                    integer += 1
    return uci_move_map


WINNER_MAP = {
    "white": 1,
    "black": -1,
    "draw": 0,
}


UCI_TO_INT_MAP = generate_uci_move_map()
UCI_TO_INT_MAP["Terminal"] = len(UCI_TO_INT_MAP)
INT_TO_UCI_MAP = {val: key for key, val in UCI_TO_INT_MAP.items()}


def board_to_tensor(
        board: chess.Board,
        dtype: torch.dtype = torch.uint8,
        device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    """
    Convert a python-chess Board to a PyTorch tensor using the TensorChessBoard encoding:
    - Empty square: 0
    - White pieces: PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6
    - Black pieces: PAWN=7, KNIGHT=8, BISHOP=9, ROOK=10, QUEEN=11, KING=12

    Ultra-optimized implementation using native bitboards and maximum vectorization.
    """
    # Initialize tensor board
    tensor_board = torch.zeros((8, 8), dtype=dtype, device=device)

    # Process all pieces at once using bitboards
    all_squares = []
    all_values = []

    # Loop through colors and piece types
    for color, offset in [(chess.WHITE, 0), (chess.BLACK, 6)]:
        for piece_type in range(1, 7):  # 1=PAWN, 2=KNIGHT, ..., 6=KING
            # Get bitboard for this piece type and color
            bitboard = board.pieces_mask(piece_type, color)

            if bitboard:
                # Convert bitboard to square indices (much faster than loop)
                # This is a bitwise operation that extracts set bits from the bitboard
                squares = []
                bb = bitboard
                while bb:
                    square = bb.bit_length() - 1
                    squares.append(square)
                    bb &= ~(1 << square)  # Clear the processed bit

                # Add to our collections
                all_squares.extend(squares)
                all_values.extend([piece_type + offset] * len(squares))

    # If there are any pieces, process them
    if all_squares:
        # Convert all squares to tensor
        squares_tensor = torch.tensor(all_squares, dtype=torch.long, device=device)

        # Convert to ranks and files in one vectorized operation
        ranks = 7 - (squares_tensor >> 3)  # bitshift is faster than division
        files = squares_tensor & 7          # bitwise AND is faster than modulo

        # Create values tensor
        values = torch.tensor(all_values, dtype=dtype, device=device)

        # Place all values at once with index_put_
        tensor_board.index_put_((ranks, files), values)

    return tensor_board


def uci_to_index(uci: str) -> Tuple[int, int, int, int]:
    """
    Convert UCI notation to index in the board tensor.
    :return Index in form (from_row, from_col, to_row, to_col) for tensor
    indexing can be interpreted (from_dim0, from_dim1, to_dim0, to_dim1).
    """
    from_file_char = uci[0].lower()
    from_rank_num = int(uci[1])
    from_col = FILE_MAP[from_file_char]
    from_row = 8 - from_rank_num  # Flip rank since tensor has 0 at top
    to_file_char = uci[0].lower()
    to_rank_num = int(uci[1])
    to_col = FILE_MAP[to_file_char]
    to_row = 8 - to_rank_num  # Flip rank since tensor has 0 at top
    return from_row, from_col, to_row, to_col


def move_to_tensor(board, move):
    """
    Convert a chess move to AlphaZero's 8×8×73 action tensor representation.

    Parameters:
        board: A chess.Board object representing the current board state
        move: A chess.Move object representing the move to encode

    Returns:
        An 8×8×73 tensor with a 1.0 at the position corresponding to the move
    """
    # Create an empty 8×8×73 tensor (all zeros)
    action_tensor = torch.zeros((8, 8, 73), dtype=torch.float32)

    # Extract move information
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion

    # Convert from chess.Square to row and column (0-7)
    from_row = chess.square_rank(from_square)
    from_col = chess.square_file(from_square)
    to_row = chess.square_rank(to_square)
    to_col = chess.square_file(to_square)

    # Get the piece at the from_square
    piece = board.piece_at(from_square)
    if piece is None:
        raise ValueError(f"No piece found at {chess.square_name(from_square)}")

    if promotion:
        # Determine the direction of the promotion
        file_diff = to_col - from_col
        if file_diff == 0:
            direction_idx = 0  # Straight
        elif file_diff == -1:
            direction_idx = 1  # Left capture
        elif file_diff == 1:
            direction_idx = 2  # Right capture
        else:
            raise ValueError(f"Invalid promotion move: {move}")

        # According to AlphaZero's encoding:
        # - 56 planes for "queen moves" (8 directions × 7 squares)
        # - 8 planes for knight moves
        # - 9 planes for promotions:
        #   * First 3 for queen promotions
        #   * Last 6 for under-promotions (knight, bishop, rook)

        # Queen promotions are already handled by the normal move planes (0-55)
        # So we only need to handle underpromotions here

        # Mapping for underpromotion plane indices (64-72)
        if promotion == chess.KNIGHT:
            # Knight promotions: first 3 underpromotion planes (64-66)
            plane_idx = 64 + direction_idx
        elif promotion == chess.BISHOP:
            # Bishop promotions: middle 3 underpromotion planes (67-69)
            plane_idx = 64 + 3 + direction_idx
        elif promotion == chess.ROOK:
            # Rook promotions: last 3 underpromotion planes (70-72)
            plane_idx = 64 + 6 + direction_idx
        elif promotion == chess.QUEEN:
            # Use standard move planes for queen promotions
            # Calculate like a normal queen move
            row_diff = to_row - from_row
            col_diff = to_col - from_col
            distance = max(abs(row_diff), abs(col_diff))

            row_dir = 0 if row_diff == 0 else row_diff // abs(row_diff)
            col_dir = 0 if col_diff == 0 else col_diff // abs(col_diff)
            direction = (row_dir, col_dir)

            directions = [
                (-1, 0), (-1, 1), (0, 1), (1, 1),
                (1, 0), (1, -1), (0, -1), (-1, -1)
            ]

            if direction not in directions:
                raise ValueError(f"Invalid direction: {direction}")

            dir_idx = directions.index(direction)
            plane_idx = dir_idx * 7 + (distance - 1)
        else:
            raise ValueError(f"Invalid promotion piece type: {promotion}")

    # CASE 2: KNIGHT MOVES
    elif piece.piece_type == chess.KNIGHT:
        # Calculate the knight move pattern
        row_diff = to_row - from_row
        col_diff = to_col - from_col

        # Map the knight move pattern to its index (0-7)
        knight_patterns = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]

        pattern = (row_diff, col_diff)
        if pattern not in knight_patterns:
            raise ValueError(f"Invalid knight move pattern: {pattern}")

        knight_idx = knight_patterns.index(pattern)

        # Base plane index for knight moves (after 56 queen planes)
        plane_idx = 56 + knight_idx
        if plane_idx >= 73:
            raise ValueError(f"Invalid plane KNIGHT: {from_row, from_col, plane_idx}")

    # CASE 3: QUEEN/ROOK/BISHOP/PAWN MOVES (all handled as "queen moves")
    else:
        # Calculate move direction
        row_diff = to_row - from_row
        col_diff = to_col - from_col

        # Calculate distance
        distance = max(abs(row_diff), abs(col_diff))

        # Validate distance (should be 1-7 on an 8x8 board)
        if distance < 1 or distance > 7:
            raise ValueError(f"Invalid move distance: {distance}. Valid distances are 1-7.")

        # Normalize to get direction vector
        row_dir = 0 if row_diff == 0 else row_diff // abs(row_diff)
        col_dir = 0 if col_diff == 0 else col_diff // abs(col_diff)
        direction = (row_dir, col_dir)

        # Map the direction to its index (0-7)
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

        if direction not in directions:
            raise ValueError(f"Invalid direction: {direction}")

        dir_idx = directions.index(direction)

        # Calculate the plane index for this queen move
        plane_idx = dir_idx * 7 + (distance - 1)
        if plane_idx >= 73:
            raise ValueError(f"Invalid plane QUEEN: {from_row, from_col, plane_idx} and {dir_idx, distance}")

    # Set the corresponding position in the tensor
    action_tensor[from_row, from_col, plane_idx] = 1.0

    # Return the tensor without flattening it
    return action_tensor


def tensor_to_move_idx(action_tensor):
    """
    Convert an 8×8×73 action tensor to a single index in the range 0-4671.

    Parameters:
        action_tensor: An 8×8×73 tensor with a 1.0 at the position corresponding to the move

    Returns:
        Integer index in the range 0-4671
    """
    # If the tensor is flattened, reshape it
    if len(action_tensor.shape) == 1:
        action_tensor = action_tensor.reshape(8, 8, 73)

    # If it's a PyTorch tensor
    positions = torch.where(action_tensor > 0)
    if len(positions[0]) == 0:
        raise ValueError("Action tensor has no non-zero elements")
    row, col, plane = positions[0].item(), positions[1].item(), positions[2].item()

    # Calculate the index: (plane * 64) + (row * 8) + col
    return (plane * 64) + (row * 8) + col


def create_legal_moves_tensor(board: chess.Board, color: bool) -> torch.Tensor:
    """
    Creates a 64x64 tensor where tensor[source_square][target_square] is True if
    a piece of the given color on the source square can legally move to the target square.

    Args:
        board (chess.Board): Current board state
        color (chess.Color): Color to generate moves for (chess.WHITE or chess.BLACK)

    Returns:
        numpy.ndarray: Boolean tensor of shape (64, 64)
    """
    # Initialize tensor with all False values
    tensor = torch.zeros((64, 64), dtype=torch.bool)

    # Only process if it's the given color's turn to move
    original_turn = board.turn
    board.turn = color

    # Loop through all squares (0-63)
    for source_square in range(64):
        # Check if there's a piece of the correct color on this square
        piece = board.piece_at(source_square)
        if piece and piece.color == color:
            # Get all legal moves from this square
            for move in board.legal_moves:
                if move.from_square == source_square:
                    target_square = move.to_square
                    tensor[source_square][target_square] = True

    # Restore original turn
    board.turn = original_turn

    return tensor
