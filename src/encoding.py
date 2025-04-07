"""
Functions for encoding chess boards and moves
"""

import chess
import numpy as np
import torch

def encode_board(board, device=None):
    """
    Encode a chess board position into a format suitable for the neural network.
    
    Args:
        board (chess.Board): The chess board to encode
        device (torch.device, optional): Device to place the tensor on
        
    Returns:
        torch.FloatTensor: Encoded board as a tensor of shape (12, 8, 8)
    """
    # 12 pieces (6 white, 6 black)
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    
    # Piece placement and types
    piece_idx = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                 "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = chess.square_rank(square), chess.square_file(square)
            planes[piece_idx[piece.symbol()]][rank][file] = 1
    
    # Return as a PyTorch tensor
    tensor = torch.FloatTensor(planes)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def move_to_index(move):
    """
    Convert a chess.Move to an index for the policy vector.
    Safety-checked to keep indices within bounds.
    
    Args:
        move (chess.Move): The move to convert
        
    Returns:
        int: The index in the policy vector
    """
    from_square = move.from_square
    to_square = move.to_square
    
    # Ensure from_square and to_square are within bounds
    if not (0 <= from_square < 64 and 0 <= to_square < 64):
        # Return a safe default index if out of bounds
        return 0
    
    # Basic move encoding (from_square * 64 + to_square)
    move_idx = from_square * 64 + to_square
    
    # Handle promotions - each promotion type gets its own index
    if move.promotion:
        promotion_offset = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3
        }
        # We use 64*64 as base index for promotions
        # Each from_square has 64*4 indices (to_square * 4 + promotion_type)
        promo_idx = 64*64 + from_square*64*4 + to_square*4 + promotion_offset.get(move.promotion, 0)
        
        # Check if promotion index is within bounds
        if promo_idx < 4672:
            return promo_idx
        else:
            # If out of bounds, default to queen promotion which should be in bounds
            return 64*64 + from_square*64 + to_square
    
    # Return the basic move index, ensuring it's within bounds
    return min(move_idx, 4671)

def index_to_move(index, board):
    """
    Convert an index from the policy vector to a chess.Move.
    
    Args:
        index (int): The index in the policy vector
        board (chess.Board): The current board state
        
    Returns:
        chess.Move: The corresponding move
    """
    # Ensure index is within bounds
    index = max(0, min(index, 4671))
    
    # If it's a promotion move
    if index >= 64*64:
        # Handle the promotion encoding
        remainder = index - 64*64
        from_square = remainder // (64*4)
        remainder = remainder % (64*4)
        to_square = remainder // 4
        promotion_type = remainder % 4
        
        promotion_piece = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][promotion_type]
        return chess.Move(from_square, to_square, promotion_piece)
    
    # Regular move
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)