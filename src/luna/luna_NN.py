import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import DictConfig # Import OmegaConf type
from .game.luna_game import ChessGame # Import for type hinting

class ResBlock(nn.Module):
    """Residual Block for ResNet"""
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class LunaNN(nn.Module):
    """ResNet architecture inspired by AlphaZero for Luna Chess - Updated for OmegaConf"""

    def __init__(self, game: ChessGame, cfg: DictConfig): # Accept game and cfg
        super(LunaNN, self).__init__()
        # Get board dimensions and action size from the game instance
        self.board_c, self.board_x, self.board_y = game.getBoardSize() # (C, H, W)
        self.action_size = game.getActionSize()
        self.cfg = cfg # Store cfg

        # Access model architecture parameters from cfg
        num_channels = self.cfg.model.num_channels
        num_res_blocks = self.cfg.model.num_res_blocks
        dropout = self.cfg.model.dropout

        # Initial Convolutional Block
        self.conv_in = nn.Conv2d(self.board_c, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_channels)

        # Residual Blocks
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy Head
        self.conv_pi = nn.Conv2d(num_channels, 2, kernel_size=1, stride=1, bias=False)
        self.bn_pi = nn.BatchNorm2d(2)
        self.fc_pi = nn.Linear(2 * self.board_x * self.board_y, self.action_size)

        # Value Head
        self.conv_v = nn.Conv2d(num_channels, 1, kernel_size=1, stride=1, bias=False)
        self.bn_v = nn.BatchNorm2d(1)
        self.fc_v1 = nn.Linear(1 * self.board_x * self.board_y, 256)
        self.fc_v2 = nn.Linear(256, 1)

    def forward(self, boardsAndValids):
        """Forward prop for ResNet"""
        x = boardsAndValids[0]

        # Initial block
        x = F.relu(self.bn_in(self.conv_in(x)))

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Policy Head
        pi = F.relu(self.bn_pi(self.conv_pi(x)))
        pi = pi.view(pi.size(0), -1)
        pi = self.fc_pi(pi)

        # Value Head
        v = F.relu(self.bn_v(self.conv_v(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_v1(v))
        v = torch.tanh(self.fc_v2(v))

        return pi, v