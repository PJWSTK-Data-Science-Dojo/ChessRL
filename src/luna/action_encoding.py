"""Spatial encoding of Luna chess action indices."""

import torch

ACTION_PLANES = 5


def _action_to_squares(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode action indices into (from_square, to_square) handling promotions.

    Actions 0..4095 are from_sq*64 + to_sq.  Actions >= 4096 encode underpromotions
    with from_file/to_file offsets (see chess_game.py).

    Underpromotion layout (64 actions each, from_file * 8 + to_file):
        4096..4159  knight
        4160..4223  rook
        4224..4287  bishop
    All underpromotions move from rank 7 (sq 48..55) to rank 8 (sq 56..63).
    """
    is_base = action < 4096
    base_from = action // 64
    base_to = action % 64

    promo_offset = (action - 4096) % 64
    from_file = promo_offset // 8
    to_file = promo_offset % 8
    promo_from = from_file + 48  # rank 7
    promo_to = to_file + 56  # rank 8

    from_sq = torch.where(is_base, base_from, promo_from)
    to_sq = torch.where(is_base, base_to, promo_to)
    return from_sq.clamp(0, 63), to_sq.clamp(0, 63)


def action_index_to_planes(action: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert action indices to five spatial planes.

    Planes 0-1 identify from/to squares. Planes 2-4 identify knight, rook,
    and bishop underpromotions at the destination square. Queen promotions use
    the base action and need no extra plane.
    """
    B = action.shape[0]
    from_sq, to_sq = _action_to_squares(action)
    rows = torch.arange(B, device=device)
    planes = torch.zeros(B, ACTION_PLANES, 64, device=device)
    planes[rows, 0, from_sq] = 1.0
    planes[rows, 1, to_sq] = 1.0
    promotion_plane = torch.where(
        action >= 4224,
        4,
        torch.where(action >= 4160, 3, torch.where(action >= 4096, 2, -1)),
    )
    is_underpromotion = promotion_plane >= 0
    planes[rows[is_underpromotion], promotion_plane[is_underpromotion], to_sq[is_underpromotion]] = 1.0
    return planes.view(B, ACTION_PLANES, 8, 8)


def action_int_to_planes(action: int, device: torch.device) -> torch.Tensor:
    """Single action index -> ``(1, 5, 8, 8)`` spatial planes."""
    action_t = torch.tensor([action], device=device)
    return action_index_to_planes(action_t, device)
