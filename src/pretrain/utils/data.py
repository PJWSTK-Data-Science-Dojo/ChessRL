from typing import Optional, List, Dict, Iterator, Any, Type
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from pretrain.utils.preprocess import PreprocessingLambda
from pretrain.utils.transforms import TransformLambda
from pretrain.utils.dataset import DatasetChunk

import math
import warnings
import os


class ChessDataset(Dataset):
    """
    Handles chess dataset loading for various dataset types.
    """

    def __init__(
        self,
        data_dir: str,
        dataset_class: Type[DatasetChunk],
        preprocessing: Optional[List[PreprocessingLambda]] = None,
        transforms: Optional[List[TransformLambda]] = None,
        **dataset_kwargs,
    ):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset not found under {data_dir}.")
        self.dataset = dataset_class(data_dir, **dataset_kwargs)
        self.preprocessing = preprocessing
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def load(self):
        self.dataset.load()

    def delete(self):
        self.dataset.delete()

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
        sample = self.dataset[idx]
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
        **dataset_init_kwargs
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
        for file in chunks:
            self.files.append(file)
        self.batch_size = batch_size
        self.datasets = [ChessDataset(file, **dataset_init_kwargs) for file in self.files]
        self.dataloaders = [DataLoader(dataset, batch_size=self.batch_size, num_workers=num_workers, pin_memory=True)
                            for dataset in self.datasets]
        self.total_size = sum(len(dataset) for dataset in self.datasets)
        self.num_of_batches = math.ceil(self.total_size / self.batch_size)
        super().__init__(DummyDataset(self.total_size))

    def __iter__(self) -> Iterator[Any]:
        for i, dataloader in enumerate(self.dataloaders):
            self.datasets[i].load()
            for batch in dataloader:
                yield batch
            self.datasets[i].delete()

    def __len__(self) -> int:
        return self.num_of_batches


class ChessDataModule(LightningDataModule):
    """
    Chess data module for loading static chess dataset.
    """

    SUPPORTED_FILE_FORMATS = [".parquet", ".pt", ".pth", ".csv"]

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1024,
        num_workers: int = 0,
        preprocessing: Optional[List[PreprocessingLambda]] = None,
        transforms: Optional[List[TransformLambda]] = None,
        dataset_class: Type[DatasetChunk] = None,
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
        self.preprocessing = preprocessing
        self.transforms = transforms
        self.dataset_class = dataset_class
        self.dataset_specific_kwargs = dataset_specific_kwargs

        self.train_chunks = []
        for i, file in enumerate(list(sorted(os.listdir(self.data_dir)))):
            if any(file.endswith(file_format) for file_format in ChessDataModule.SUPPORTED_FILE_FORMATS):
                self.train_chunks.append(os.path.join(data_dir, file))
            elif os.path.isfile(file):
                warnings.warn(f"In the dataset directory there is a file {file}, which is not supported. Only {ChessDataModule.SUPPORTED_FILE_FORMATS} are supported.")

    def train_dataloader(self):
        return ChessChunkedDataLoader(
            self.data_dir,
            self.batch_size,
            self.train_chunks,
            self.num_workers,
            preprocessing=self.preprocessing,
            transforms=self.transforms,
            dataset_class=self.dataset_class,
            **self.dataset_specific_kwargs
        )
