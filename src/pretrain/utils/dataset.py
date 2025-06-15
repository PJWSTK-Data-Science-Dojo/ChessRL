"""
Scalable dataset utilities for loading processed chess training data.
Designed to handle datasets with billions of transitions by loading only one batch at a time.
"""

import sys
import psutil
import json
import torch
import numpy as np
from typing import Dict, Optional, Iterator, Any, Callable
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import gc
import time

from pretrain.utils.tools import safe_delete_and_wait


class ChessDataset(Dataset):
    """
    Dataset class that operates on a single batch file.
    Loads and manages one batch at a time to handle massive datasets.
    """
    
    def __init__(self, batch_id: int, data_dir: str, transform: Optional[Callable] = None, total_transitions: int = 0, board_history: int = 1):
        """
        Initialize dataset for a specific batch.
        
        Args:
            batch_id: ID of the batch file to load
            data_dir: Directory containing batch files
            transform: Optional transform to apply to samples
        """
        self.batch_id = batch_id
        self.data_dir = Path(data_dir)
        self.batch_file = self.data_dir / f"batch_{batch_id:06d}.pt"
        self.transform = transform
        self.total_transitions = total_transitions
        self.board_history = board_history

        # Data containers
        self.boards = None
        self.actions = None
        self.values = None
        self.is_loaded = False
        
        # Verify batch file exists
        if not self.batch_file.exists():
            raise FileNotFoundError(f"Batch file not found: {self.batch_file}")
    
    def load(self) -> None:
        """Load the batch file into memory."""
        if self.is_loaded:
            return
            
        batch_data = torch.load(self.batch_file, map_location='cpu')
        self.boards = batch_data['boards'].share_memory_()
        self.actions = batch_data['actions'].share_memory_()
        self.values = batch_data['values'].share_memory_()
        self.history = batch_data['history'].share_memory_()
        self.is_loaded = True

        # Pad the boards with `history_len` on the beggining to ensure there is always some history to retrieve without
        # if conditioning during the __getitem__
        self.boards = torch.cat([torch.zeros((self.board_history - 1, *self.boards.shape[1:]),
                                             dtype=self.boards.dtype,
                                             device=self.boards.device),
                                             self.boards], dim=0)
        self.mask = torch.ones((self.__len__(), self.board_history), dtype=torch.bool)
    
    def delete(self) -> None:
        """Free memory by deleting loaded data."""
        if not self.is_loaded:
            return
            
        # For big tensor objects wait until garabge collector fully free the memory
        safe_delete_and_wait(self.boards, self.actions, self.values, self.history)
        
        self.boards = None
        self.actions = None  
        self.values = None
        self.history = None
        self.is_loaded = False
    
    def __len__(self) -> int:
        """Return number of transitions in this batch."""
        return self.total_transitions
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single transition from the batch."""
        if not self.is_loaded:
            return None

        mask = self.mask[idx]
        mask[:-((self.history[idx] + 1) if (self.history[idx] + 1) < self.board_history else -self.board_history)] = False
        board_idx = idx + self.board_history

        sample = {
            'board': self.boards[board_idx - self.board_history:board_idx].to(torch.float32),
            'padding_mask': mask,
            'action': self.actions[idx].to(torch.int32),
            'value': self.values[idx].to(torch.float32)
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class ChessDataLoader:
    """
    Data loader that loads one batch at a time from disk.
    Designed for massive datasets that don't fit in memory.
    """
    
    def __init__(self, data_dir: str, batch_size: int = 32, shuffle: bool = True, **kwargs):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing batch files and metadata
            batch_size: Training batch size
            shuffle: Whether to shuffle data
            resume_at_chunk: Chunk (batch file) index to resume from
            **kwargs: Additional arguments for DataLoader
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.step = 0
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.start_setup = {
            'start_chunk_idx': 0,
            'skip_steps': 0,
            'initialized': False
        }
        
        # Find all batch files
        self.batch_files = sorted(list(self.data_dir.glob("batch_*.pt")))
        if not self.batch_files:
            raise ValueError(f"No batch files found in {data_dir}")
        
        # Calculate transitions per batch (use avg_transitions_per_batch if available)
        if 'avg_transitions_per_batch' in self.metadata:
            self.transitions_per_batch = int(self.metadata['avg_transitions_per_batch'])
        else:
            self.transitions_per_batch = self.metadata['total_transitions'] // self.metadata['num_batches']
        
        # Create datasets
        self.datasets = []
        for i, batch_file in enumerate(self.batch_files):
            batch_id = int(batch_file.stem.split('_')[1])
            self.datasets.append(ChessDataset(batch_id, str(self.data_dir), total_transitions=self.transitions_per_batch))
        self.data_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs) for dataset in self.datasets]

    def resume(self, step: int):
        chunk_idx = step * self.batch_size // self.transitions_per_batch
        self.start_setup['start_chunk_idx'] = chunk_idx
        resume_transition = step * self.batch_size % self.transitions_per_batch
        resume_step = resume_transition // self.batch_size
        self.start_setup['skip_steps'] = resume_step
        self.step = step

    def state_dict(self) -> dict:
        return {
            'global_step': self.step
        }

    def load_state_dict(self, state_dict: dict):
        self.resume(state_dict['global_step'])

    def __iter__(self):
        # Start from the resume chunk
        self.step = 0 if self.start_setup['initialized'] else self.step  # If resuming or just starting use the original step. Else reset on epoch repetitions.
        for loader in self.data_loaders[self.start_setup['start_chunk_idx']:]:
            loader.dataset.load()

            # Skip until the given step
            if self.start_setup['skip_steps'] > 0 and not self.start_setup['initialized']:
                for idx, batch in enumerate(loader):
                    if idx == self.start_setup['skip_steps']:
                        break
                self.start_setup['initialized'] = True

            # Yield batches
            for batch in loader:
                self.step += 1
                yield batch
            loader.dataset.delete()

    def get_current_chunk(self) -> int:
        return self.current_chunk
    
    def set_current_chunk(self, chunk_idx: int):
        self.current_chunk = chunk_idx

    def __len__(self):
        return self.metadata['total_transitions'] // self.batch_size + (1 if self.metadata['total_transitions'] % self.batch_size != 0 else 0)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata file with dataset statistics."""
        metadata_file = self.data_dir / "metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return metadata


class ChessDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for chess training data.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        transform: Optional[Callable] = None,
        **dataloader_kwargs
    ):
        """
        Initialize the data module.
        
        Args:
            data_dir: Directory containing batch files
            batch_size: Training batch size
            num_workers: Number of worker processes for data loading
            transform: Optional transform to apply to samples
            **dataloader_kwargs: Additional arguments for DataLoader
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.dataloader_kwargs = dataloader_kwargs
        
        self.train_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training."""
        if stage == "fit" or stage is None:
            self.train_dataset = ChessDataLoader(
                data_dir=self.data_dir,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                **self.dataloader_kwargs
            )
    
    def train_dataloader(self):
        """Return training dataloader."""
        return self.train_dataset
