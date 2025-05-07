from typing import Dict
from pretrain.utils.board import INT_TO_UCI_MAP, uci_to_index, board_to_tensor, create_legal_moves_tensor

import chess
import abc
import torch


class PreprocessingLambda(metaclass=abc.ABCMeta):
    """
    Lambda takes a dictionary, which represents a single data point from the dataset
    and returns new dictionary representing the same point (possible with totally new data).
    Has to be carefully chained.
    """

    @abc.abstractmethod
    def __call__(self, sample: Dict) -> Dict:
        ...


class PreprocessFenDataset(PreprocessingLambda):
    """
    Prepares dataset for neural network model effectively encoding all the needed information.

    * State is tensor of shape (6, 8, 8) where each piece has its integer representing it.
    * Value is a single value tensor of the value for the current player's move
    (-1 - loss, 0 - draw, 1 - win)
    * Label is a tensor representing the action label (out of 4096). No probability distribution is needed
    as dataset provides infor about only one valid move.
    * Mask is an integer tensor marking allows moves like a boolean. Can be excluded by `use_mask=False`.
    """

    BAD_INDEX = -100

    def __init__(self, use_mask: bool = True):
        self.use_mask = use_mask

    @staticmethod
    def one_hot_encoding(board: chess.Board, color: bool) -> torch.Tensor:
        state = board_to_tensor(board).to(torch.float32)
        if color == chess.WHITE:
            current_base = 1
            opponent_base = 7
        else:
            current_base = 7
            opponent_base = 1
        current_pieces = torch.arange(current_base, current_base + 6, dtype=state.dtype, device=state.device)
        opponent_pieces = torch.arange(opponent_base, opponent_base + 6, dtype=state.dtype, device=state.device)
        new_state = torch.zeros((6, 8, 8,), dtype=torch.float32)
        new_state[state.unsqueeze(0) == current_pieces.view(-1, 1, 1)] = 1.0
        new_state[state.unsqueeze(0) == opponent_pieces.view(-1, 1, 1)] = -1.0
        return new_state

    def __call__(self, sample: Dict) -> Dict:
        board = chess.Board(sample["states"])
        uci = INT_TO_UCI_MAP[sample["actions"]]
        color = sample["move_index"] % 2 == 0
        if color == chess.BLACK:
            value = -sample["winner"]
        else:
            value = sample["winner"]
        sample = {
            "board": board,
            "uci": uci,
            "value": value,
            "color": color,
        }

        state = PreprocessFenDataset.one_hot_encoding(sample["board"], sample["color"])
        if sample["uci"] != "Terminal":
            from_row, from_col, to_row, to_col = uci_to_index(sample["uci"])
            from_pos = from_row * 8 + from_col
            to_pos = to_row * 8 + to_col
            label = from_pos * 64 + to_pos
        else:
            label = PreprocessFenDataset.BAD_INDEX
        if self.use_mask:
            mask = create_legal_moves_tensor(sample["board"], sample["color"]).to(torch.int32).flatten()
        else:
            mask = torch.ones((4096,), dtype=torch.int32)
        return {
            "state": state,
            "value": torch.tensor(sample["value"], dtype=torch.float32),
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long),
        }


class PreprocessTensorDataset(PreprocessingLambda):
    """
    Prepares dataset for neural network model effectively encoding all the needed information.
    Used for direct tensor dataset (where dataset is capable of being passed directly to a simple network or
    with little effort to the network).
    """

    @staticmethod
    def one_hot_encoding(sample: Dict) -> torch.Tensor:
        state = torch.zeros((19, 8, 8), dtype=torch.float32)
        state[12] = sample["clock"] % 2  # Who is to move
        state[13] = sample["repetitions"]  # For three repetitions rule
        state[14] = sample["castling_rights"][0]
        state[15] = sample["castling_rights"][1]
        state[16] = sample["castling_rights"][2]
        state[17] = sample["castling_rights"][3]
        state[18] = sample["clock"]  # for 50 moves rule
        figures = torch.arange(12)
        state[:12] = torch.eq(sample['state'], figures.reshape(12, 1, 1)).to(torch.float32)
        return state


    def __call__(self, sample: Dict) -> Dict:
        return {
            'mask': torch.ones(len(INT_TO_UCI_MAP), dtype=torch.int8),
            'state': PreprocessTensorDataset.one_hot_encoding(sample),
            'label': sample["action"],
            'value': sample['value'].to(torch.float32),
        }
