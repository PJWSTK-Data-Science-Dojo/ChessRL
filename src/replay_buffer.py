"""
Replay buffer and dataset for training the chess model
"""

from collections import deque
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os

class ReplayBuffer:
    """
    Buffer for storing and sampling experience
    """
    def __init__(self, max_size):
        """
        Initialize the replay buffer
        
        Args:
            max_size (int): Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        """
        Add experience to the buffer
        
        Args:
            experience (list): List of experience dictionaries
        """
        self.buffer.extend(experience)
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            list: Sampled experiences
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
            
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        """Get the current size of the buffer"""
        return len(self.buffer)
    
    def save(self, filename):
        """
        Save the replay buffer to a file
        
        Args:
            filename (str): File to save the buffer to
        """
        with open(filename, 'wb') as f:
            # Create a copy of the buffer with numpy arrays instead of tensors
            serializable_buffer = []
            for item in self.buffer:
                serializable_item = {}
                for key, value in item.items():
                    if isinstance(value, torch.Tensor):
                        serializable_item[key] = value.cpu().numpy()
                    else:
                        serializable_item[key] = value
                serializable_buffer.append(serializable_item)
            pickle.dump(serializable_buffer, f)
        print(f"Saved {len(self.buffer)} examples to {filename}")

    def load(self, filename, device=None):
        """
        Load the replay buffer from a file
        
        Args:
            filename (str): File to load the buffer from
            device (torch.device, optional): Device to load tensors to
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Replay buffer file {filename} not found")
        
        with open(filename, 'rb') as f:
            loaded_buffer = pickle.load(f)
        
        # Clear the current buffer and add the loaded items
        self.buffer.clear()
        
        # Convert numpy arrays back to tensors if needed
        for item in loaded_buffer:
            if 'board' in item and isinstance(item['board'], np.ndarray):
                tensor = torch.FloatTensor(item['board'])
                if device is not None:
                    tensor = tensor.to(device)
                item['board'] = tensor
            self.buffer.append(item)
        
        print(f"Loaded {len(self.buffer)} examples from {filename}")


class ChessDataset(Dataset):
    """
    Dataset for training the chess model
    """
    def __init__(self, examples):
        """
        Initialize the dataset
        
        Args:
            examples (list): List of experience dictionaries
        """
        self.examples = examples
    
    def __len__(self):
        """Get the size of the dataset"""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset
        
        Args:
            idx (int): Index of the item
            
        Returns:
            tuple: (board, policy, value)
        """
        example = self.examples[idx]
        return example['board'], torch.FloatTensor(example['policy']), torch.FloatTensor([example['value']])