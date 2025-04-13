import math
from typing import Optional, List, Callable, Dict, Iterator, Any
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from pretrain.utils.preprocess import PreprocessingLambda
from pretrain.utils.transforms import TransformLambda

import pyarrow.parquet as pq
import polars as pl
import os


class ChessDataset(Dataset):
    """
    Handles chess dataset loading for various dataset types.
    The dataset is lazy loaded by default (initialisation does not load it)
    """

    def __init__(
        self,
        data_dir: str,
        schema: pl.Schema,
        lazy_load: bool = True,
        preprocessing: Optional[List[PreprocessingLambda]] = None,
        transforms: Optional[List[TransformLambda]] = None,
    ):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset not found under {data_dir}.")
        self.data_dir = data_dir
        self.schema = schema
        self.dataset_df = None
        self.data_len = pq.ParquetFile(data_dir).metadata.num_rows
        self.preprocessing = preprocessing
        self.transforms = transforms
        if not lazy_load:
            self.load()

    def __len__(self):
        return self.data_len

    def load(self):
        self.dataset_df = pl.scan_parquet(self.data_dir, schema=self.schema).collect(engine="in-memory")

    def delete(self):
        del self.dataset_df

    def __apply_preprocessing(self, sample: Dict):
        if self.preprocessing is None:
            return sample
        else:
            for preprocess in self.preprocessing:
                sample = preprocess(sample)
            return sample

    def __apply_transforms(self, sample: Dict):
        if self.transforms is None:
            return sample
        else:
            for transform in self.transforms:
                sample = transform(sample)
            return sample

    def __getitem__(self, idx: int):
        sample = self.dataset_df.row(idx, named=True)
        sample = self.__apply_preprocessing(sample)
        sample = self.__apply_transforms(sample)
        return sample


class DummyDataset(Dataset):

    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return None


class ChessChunkedDataLoader(DataLoader):
    """
    Handles data loading with multiple chunks.
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        chunks: List[str],
        num_workers: int = 0,
        **dataset_kwargs
    ):
        """
        :param root_dir: Directory where parquet datasets are located.
        :param batch_size: Batch size
        :param chunks: List of chunk files' names.
        :param num_workers: Number of parallel workers.
        :param dataset_kwargs: Keyword arguments passed to dataset.
        """
        self.root_dir = root_dir
        self.files = []
        self.total_size = 0
        for file in chunks:
            self.files.append(file)
            self.total_size += pq.ParquetFile(file).metadata.num_rows
        self.batch_size = batch_size
        self.batches = math.ceil(self.total_size / self.batch_size)
        self.datasets = [ChessDataset(file, **dataset_kwargs) for file in self.files]
        self.dataloaders = [DataLoader(dataset, batch_size=self.batch_size, num_workers=num_workers)
                            for dataset in self.datasets]
        super().__init__(DummyDataset(self.total_size))

    def __iter__(self) -> Iterator[Any]:
        for i, dataloader in enumerate(self.dataloaders):
            self.datasets[i].load()
            for batch in dataloader:
                yield batch
            self.datasets[i].delete()

    def __len__(self) -> int:
        return self.batches


class ChessDataModule(LightningDataModule):
    """
    Chess data module for loading static chess dataset.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1024,
        num_workers: int = 0,
        **dataset_specific_kwargs
    ):
        """
        :param data_dir: Directory where data chunks are located as parquet files.
        :param batch_size: Batch size used for data loaders.
        :param num_workers: Number of workers for data loader. Remember each worker copies the dataset!
        :param dataset_specific_kwargs: Keyword arguments passed to dataset.
        """
        super(ChessDataModule, self).__init__()
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset not found under {data_dir}.")
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"Data path should be a directory, got {data_dir}.")

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_specific_kwargs = dataset_specific_kwargs

        self.train_chunks = []
        for i, file in enumerate(list(sorted(os.listdir(self.data_dir)))):
            if file.endswith(".parquet"):
                self.train_chunks.append(os.path.join(data_dir, file))

    def get_chunks_num(self):
        return len(filter(lambda file: file.endswith(".parquet"), os.listdir(self.data_dir)))

    def train_dataloader(self):
        return ChessChunkedDataLoader(
            self.data_dir,
            self.batch_size,
            self.train_chunks,
            self.num_workers,
            **self.dataset_specific_kwargs
        )
