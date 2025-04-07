"""
Optimized trainer for the chess reinforcement learning model
with better GPU and CPU utilization
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import threading
import pickle

from ChessRL.replay_buffer import ReplayBuffer, ChessDataset

# Dodana flaga do wyłączenia problematycznych modułów torch'a
os.environ["TORCH_COMPILE_DISABLE_CONTEXTS"] = "1"

class Trainer:
    """
    Optimized trainer for the chess reinforcement learning model
    """
    def __init__(self, network, config):
        """
        Initialize the trainer with performance optimizations
        
        Args:
            network (ChessNetwork): The neural network to train
            config (Config): Configuration parameters
        """
        self.network = network
        self.config = config
        self.device = next(network.parameters()).device
        
        # Optimizer with performance improvements - używamy AdamW bezpośrednio zamiast przez optim
        # co pozwala uniknąć problematycznych importów sympy
        self.optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-5,  # Improved numerical stability
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate * 10,
            total_steps=config.epochs * (config.replay_buffer_size // config.batch_size),
            pct_start=0.1,  # Warm up for 10% of training
            div_factor=10.0,
            final_div_factor=1000.0,
            anneal_strategy='cos'
        )
        
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        
        # Mixed precision training for better GPU utilization
        self.scaler = torch.amp.GradScaler() if config.use_mixed_precision else None
        
        # Set optimal thread settings
        if self.device.type == 'cuda':
            # Set CUDA specific optimizations
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True  # Use TensorFloat32 for faster computation
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            
            # Pre-allocate CUDA memory to avoid fragmentation
            torch.cuda.empty_cache()
            
            # Limit CPU threads when using GPU
            torch.set_num_threads(min(4, os.cpu_count()))
        else:
            # For CPU, use more threads
            torch.set_num_threads(min(16, os.cpu_count()))
    
    def train(self, examples):
        """
        Update the network using examples from self-play with performance optimizations
        
        Args:
            examples (list): List of experience dictionaries from self-play
        """
        # Add examples to replay buffer
        self.replay_buffer.add(examples)
        
        if len(self.replay_buffer) < self.config.batch_size:
            print(f"Not enough examples for training. Current: {len(self.replay_buffer)}, Required: {self.config.batch_size}")
            return
        
        # Sample batches and train
        start_time = time.time()
        total_samples = 0
        
        # Reset metrics for this training session
        policy_losses = []
        value_losses = []
        total_losses = []
        
        # Ensure model is in training mode
        self.network.train()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            dataloader = self._create_batches()
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_total_losses = []
            
            # Track throughput
            samples_in_epoch = 0
            batch_times = []
            
            for batch_idx, batch in enumerate(dataloader):
                batch_start = time.time()
                states, target_policies, target_values = [x.to(self.device, non_blocking=True) for x in batch]
                batch_size = states.size(0)
                total_samples += batch_size
                samples_in_epoch += batch_size
                
                # Use mixed precision training if available
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        # Forward pass
                        out_policies, out_values = self.network(states)
                        
                        # Calculate loss
                        policy_loss = F.cross_entropy(out_policies, target_policies)
                        value_loss = F.mse_loss(out_values, target_values)
                        total_loss = policy_loss + value_loss
                    
                    # Backward pass with gradient scaling
                    self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                    self.scaler.scale(total_loss).backward()
                    
                    # Apply gradient clipping to prevent exploding gradients
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular training
                    # Forward pass
                    out_policies, out_values = self.network(states)
                    
                    # Calculate loss
                    policy_loss = F.cross_entropy(out_policies, target_policies)
                    value_loss = F.mse_loss(out_values, target_values)
                    total_loss = policy_loss + value_loss
                    
                    # Backward pass and optimization
                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                # Step the learning rate scheduler
                self.scheduler.step()
                
                # Record metrics
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_total_losses.append(total_loss.item())
                
                # Track batch time
                batch_times.append(time.time() - batch_start)
                
                # Print progress periodically
                if (batch_idx + 1) % 20 == 0 or batch_idx == len(dataloader) - 1:
                    avg_batch_time = np.mean(batch_times[-20:])
                    samples_per_sec = batch_size / avg_batch_time
                    
                    print(f"Epoch {epoch+1}/{self.config.epochs} - "
                          f"Batch {batch_idx+1}/{len(dataloader)} - "
                          f"Loss: {total_loss.item():.4f} - "
                          f"Samples/sec: {samples_per_sec:.1f}")
            
            # Log epoch metrics
            policy_losses.extend(epoch_policy_losses)
            value_losses.extend(epoch_value_losses)
            total_losses.extend(epoch_total_losses)
            
            # Calculate epoch statistics
            avg_policy_loss = np.mean(epoch_policy_losses)
            avg_value_loss = np.mean(epoch_value_losses)
            avg_total_loss = np.mean(epoch_total_losses)
            epoch_time = time.time() - epoch_start
            
            # Report epoch metrics
            print(f"Epoch {epoch+1}/{self.config.epochs} completed in {epoch_time:.1f}s - "
                  f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, "
                  f"Total Loss: {avg_total_loss:.4f}, "
                  f"Samples/sec: {samples_in_epoch / epoch_time:.1f}")
        
        # Log overall training metrics
        total_time = time.time() - start_time
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_total_loss = np.mean(total_losses)
        
        print(f"Training completed in {total_time:.1f}s - "
              f"Average Policy Loss: {avg_policy_loss:.4f}, "
              f"Average Value Loss: {avg_value_loss:.4f}, "
              f"Average Total Loss: {avg_total_loss:.4f}, "
              f"Samples/sec: {total_samples / total_time:.1f}")
        
        # Ensure model is back in eval mode for self-play
        self.network.eval()
    
    def _create_batches(self):
        """
        Create optimized batches for training
        
        Returns:
            DataLoader: Batches of training data
        """
        samples = self.replay_buffer.sample(min(len(self.replay_buffer), self.config.batch_size * 100))
        dataset = ChessDataset(samples)
        
        # Configure dataloader with optimal performance settings
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_data_workers,
            drop_last=False,
            prefetch_factor=2 if self.config.num_data_workers > 0 else None,
            persistent_workers=True if self.config.num_data_workers > 0 else False
        )
        return dataloader
    
    def save_model(self, filename):
        """
        Save the model to disk with additional optimization info
        
        Args:
            filename (str): Filename to save the model to
        """
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)
        
        # Use a more efficient saving approach
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': {
                'n_features': self.config.n_features,
                'n_residual_blocks': self.config.n_residual_blocks,
            }
        }, os.path.join(self.config.model_dir, filename), _use_new_zipfile_serialization=True)
    
    def load_model(self, filename):
        """
        Load the model from disk
        
        Args:
            filename (str): Filename to load the model from
        """
        # First clear GPU memory if needed
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        checkpoint = torch.load(os.path.join(self.config.model_dir, filename), map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        
        # Only load optimizer state if it matches the current optimizer
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            print("Warning: Could not load optimizer state. Starting with fresh optimizer.")
            
    def save_replay_buffer(self, filename):
        """
        Save the replay buffer to a file
        
        Args:
            filename (str): Filename to save the replay buffer to
        """
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)
        
        buffer_path = os.path.join(self.config.model_dir, filename)
        
        # Stwórz kopię bufora z tablicami NumPy zamiast tensorów
        serializable_buffer = []
        for item in self.replay_buffer.buffer:
            serializable_item = {}
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    serializable_item[key] = value.cpu().numpy()
                else:
                    serializable_item[key] = value
            serializable_buffer.append(serializable_item)
            
        # Zapisz do pliku
        with open(buffer_path, 'wb') as f:
            pickle.dump(serializable_buffer, f)
            
        print(f"Replay buffer saved to {buffer_path} with {len(self.replay_buffer)} examples")

    def load_replay_buffer(self, filename):
        """
        Load the replay buffer from a file
        
        Args:
            filename (str): Filename to load the replay buffer from
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        buffer_path = os.path.join(self.config.model_dir, filename)
        if not os.path.exists(buffer_path):
            print(f"Replay buffer file {buffer_path} not found")
            return False
            
        try:
            # Wczytaj dane
            with open(buffer_path, 'rb') as f:
                loaded_buffer = pickle.load(f)
            
            # Wyczyść obecny bufor
            self.replay_buffer.buffer.clear()
            
            # Konwertuj tablice NumPy z powrotem na tensory
            for item in loaded_buffer:
                if 'board' in item and isinstance(item['board'], np.ndarray):
                    tensor = torch.FloatTensor(item['board'])
                    if self.device.type == 'cuda':
                        tensor = tensor.to(self.device)
                    item['board'] = tensor
                self.replay_buffer.buffer.append(item)
            
            print(f"Replay buffer loaded from {buffer_path} with {len(self.replay_buffer)} examples")
            return True
        except Exception as e:
            print(f"Error loading replay buffer: {e}")
            return False