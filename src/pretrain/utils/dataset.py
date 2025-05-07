from typing import Any

import polars as pl
import pyarrow.parquet as pq
import os
import abc
import torch


class DatasetChunk(metaclass=abc.ABCMeta):
    """
    Interface for representing dataset chunk. Can be than used with ChessDataset
    to train the model.
    """

    @abc.abstractmethod
    def __init__(self, data_dir: str, **dataset_kwargs):
        ...

    @abc.abstractmethod
    def load(self):
        ...

    @abc.abstractmethod
    def delete(self):
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Any:
        ...


class FenDataset(DatasetChunk):
    """
    For reading and handling FEN datasets stored as parquet files.
    """

    Schema = pl.Schema({
        "states": pl.Utf8,
        "winner": pl.Int8,
        "game": pl.UInt64,
        "move_index": pl.Int16,
        "actions": pl.Int16,
    })

    def __init__(
            self,
            data_dir: str,
    ):
        self.data_dir = data_dir
        self.dataset_df = None
        self.data_len = pq.ParquetFile(data_dir).metadata.num_rows

    def load(self):
        self.dataset_df = pl.scan_parquet(self.data_dir, schema=self.Schema).collect(engine="in-memory")

    def delete(self):
        del self.dataset_df

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx: int):
        return self.dataset_df.row(idx, named=True)


class DirectTensorDataset(DatasetChunk):
    """
    Reads the dataset stored in PyTorch tensor dict format. Occupies more disk space and
    during loading RAM space, but is faster and its preprocessing sometimes can be vectorised.
    """

    def __init__(
            self,
            data_dir: str,
    ):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset not found under {data_dir}.")
        self.data_dir = data_dir
        self.dataset = None
        self.data_len = 0
        self.load()
        self.data_len = len(self.dataset['states'])
        self.delete()

    def load(self):
        self.dataset = torch.load(self.data_dir, map_location=torch.device("cpu"))

    def delete(self):
        del self.dataset

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx: int):
        return {
            'state': self.dataset['states'][idx],
            'action': self.dataset['actions'][idx],
            'value': self.dataset['values'][idx],
            'castling_rights': self.dataset['castling_rights'][idx],
            'clock': self.dataset['clocks'][idx],
            'repetitions': self.dataset['repetitions'][idx],
        }
