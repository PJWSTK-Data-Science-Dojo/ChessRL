"""Legacy AlphaZero-style network architecture.

Kept for reference. The active model is in `ezv2_model.py` (`EZV2Networks`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, optim

from .game.luna_game import ChessGame
from .utils import dotdict


class LegacyAlphaZeroNetwork(nn.Module):
    """Original AlphaZero-style policy+value network (deprecated)."""

    def __init__(self, game: ChessGame, args: dotdict) -> None:
        super().__init__()

        self.board_x, self.board_y, self.board_z = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        num_channels = args.num_channels
        self.conv1 = nn.Conv3d(1, num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(num_channels, num_channels * 2, 3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(num_channels * 2, num_channels * 2, 3, stride=1)
        self.conv4 = nn.Conv3d(num_channels * 2, num_channels * 2, 3, stride=1)
        self.conv5 = nn.Conv3d(num_channels * 2, num_channels, 1, stride=1)

        self.bn1 = nn.BatchNorm3d(num_channels)
        self.bn2 = nn.BatchNorm3d(num_channels * 2)
        self.bn3 = nn.BatchNorm3d(num_channels * 2)
        self.bn4 = nn.BatchNorm3d(num_channels * 2)
        self.bn5 = nn.BatchNorm3d(num_channels)

        flat_size = num_channels * (self.board_x - 4) * (self.board_y - 4) * (self.board_z - 4)
        self.fc1 = nn.Linear(flat_size, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.fc_bn3 = nn.BatchNorm1d(512)

        self.fc4 = nn.Linear(512, self.action_size)
        self.fc5 = nn.Linear(512, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, boards_and_valids: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        board_tensor, valid_mask = boards_and_valids
        num_channels = self.args.num_channels

        board_tensor = board_tensor.view(-1, 1, self.board_x, self.board_y, self.board_z)
        board_tensor = F.relu(self.bn1(self.conv1(board_tensor)))
        board_tensor = F.relu(self.bn2(self.conv2(board_tensor)))
        board_tensor = F.relu(self.bn3(self.conv3(board_tensor)))
        board_tensor = F.relu(self.bn4(self.conv4(board_tensor)))
        board_tensor = F.relu(self.bn5(self.conv5(board_tensor)))
        board_tensor = board_tensor.view(
            -1, num_channels * (self.board_x - 4) * (self.board_y - 4) * (self.board_z - 4)
        )
        board_tensor = F.dropout(
            F.relu(self.fc_bn1(self.fc1(board_tensor))),
            p=self.args.dropout,
            training=self.training,
        )
        board_tensor = F.dropout(
            F.relu(self.fc_bn2(self.fc2(board_tensor))),
            p=self.args.dropout,
            training=self.training,
        )
        board_tensor = F.dropout(
            F.relu(self.fc_bn3(self.fc3(board_tensor))),
            p=self.args.dropout,
            training=self.training,
        )

        policy_logits = self.fc4(board_tensor)
        value_logits = self.fc5(board_tensor)

        policy_logits -= (1 - valid_mask) * 1000
        return F.log_softmax(policy_logits, dim=1), torch.tanh(value_logits)
