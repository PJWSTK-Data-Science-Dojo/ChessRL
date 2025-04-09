import torch
import torch.nn as nn
import numpy as np

class ChessTransforms(object):
    """
    Chess-specific transformations for reinforcement learning.
    Simplified version that doesn't require image augmentation libraries.
    """
    def __init__(self, augmentation=None, image_shape=None):
        """
        Initialize the chess transformations.
        
        Args:
            augmentation: Optional list of augmentation types (ignored in chess version)
            image_shape: Optional tuple defining the board shape (ignored in chess version)
        """
        self.augmentation = augmentation or []
        
    @torch.no_grad()
    def transform(self, board_tensors):
        """
        Transform chess board tensors.
        For chess, we keep transformations minimal or optional.
        
        Args:
            board_tensors: Tensor representation of chess boards
                          Expected shape: [batch_size, channels, height, width]
        
        Returns:
            Transformed board tensors
        """
        return board_tensors
    
    def apply_transforms(self, transforms, board):
        """
        Apply a list of transforms to a chess board.
        This is a placeholder for potential future chess-specific transformations.
        
        Args:
            transforms: List of transformation functions
            board: Tensor representation of chess board
            
        Returns:
            Transformed board
        """
        for transform in transforms:
            board = transform(board)
        return board


# Aliasing Transforms to ChessTransforms for backward compatibility
Transforms = ChessTransforms


class Intensity(nn.Module):
    """
    Intensity transformation (kept for compatibility).
    Adjusts the intensity of input tensors.
    """
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


# Optional chess-specific transformations that could be implemented:

class FlipBoard(nn.Module):
    """
    Optionally flip the chess board horizontally.
    This can be used for data augmentation in chess.
    """
    def __init__(self, flip_probability=0.5):
        super().__init__()
        self.flip_probability = flip_probability
        
    @torch.no_grad()
    def forward(self, x):
        if torch.rand(1).item() < self.flip_probability:
            return torch.flip(x, dims=[-1])  # Flip along the width dimension
        return x


class ChessSpecificTransforms(ChessTransforms):
    """
    Extended version with chess-specific transformations.
    Use this class if you want to add chess-specific augmentations.
    """
    def __init__(self, use_flip=False, flip_probability=0.5):
        super().__init__()
        self.transforms = []
        
        if use_flip:
            self.transforms.append(FlipBoard(flip_probability))
    
    @torch.no_grad()
    def transform(self, board_tensors):
        return self.apply_transforms(self.transforms, board_tensors)