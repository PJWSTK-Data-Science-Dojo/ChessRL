import os
import time
import numpy as np
import logging
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from .luna_NN import LunaNN as net
from .utils import AverageMeter # Keep AverageMeter
from .game.luna_game import ChessGame
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR

log = logging.getLogger(__name__)

class Luna_Network(object):
    """Neural Network Wrapper - Updated for Inference-only Init and separate Training Setup"""

    # Change __init__ signature
    # It takes game and model_cfg (containing model params and cuda flag)
    def __init__(self, game: ChessGame, model_cfg: DictConfig) -> None:
        super(Luna_Network, self).__init__()

        # Store model_cfg and game
        self.model_cfg = model_cfg
        self.game = game

        # Initialize the underlying neural network
        self.nnet = net(game, model_cfg) # Pass game and model_cfg to the NN constructor
        self.board_c, self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.wandb_enabled = wandb.run is not None # Check if wandb is active

        # Access cuda from model_cfg
        if self.model_cfg.cuda:
            log.info("Moving model to GPU.")
            self.nnet.cuda()

        # Optimizer and Scheduler are NOT initialized here for inference instances
        self.optimizer = None
        self.scheduler = None
        # Store a reference to the full training config if set by Coach
        self.training_cfg = None
        self.optimizer_cfg = None


    def setup_optimizer_scheduler(self, optimizer_cfg: DictConfig, training_cfg: DictConfig):
        """Sets up optimizer and scheduler for training. Called by Coach."""
        if self.optimizer is not None or self.scheduler is not None:
             log.warning("Optimizer and scheduler already set up for this NNet instance.")
             return

        self.optimizer_cfg = optimizer_cfg
        self.training_cfg = training_cfg

        initial_lr = self.optimizer_cfg.lr
        weight_decay = self.optimizer_cfg.weight_decay
        log.info(f"Setting up Adam optimizer with LR: {initial_lr}, Weight Decay: {weight_decay}")
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=initial_lr, weight_decay=weight_decay)

        step_size = self.training_cfg.get('lr_decay_step_size', 2) # Default step_size
        gamma = self.training_cfg.get('lr_decay_gamma', 0.69)       # Default gamma
        log.info(f"Using StepLR with step_size={step_size}, gamma={gamma}")
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)


    def train(self, examples) -> None:
        """Train on examples with WandB logging. Requires optimizer/scheduler setup."""
        if self.optimizer is None or self.scheduler is None or self.training_cfg is None or self.optimizer_cfg is None:
             log.error("Attempted to call train() on a Luna_Network instance without full training setup.")
             return # Cannot train without setup

        # Access training parameters from stored config
        epochs = self.training_cfg.epochs
        batch_size = self.optimizer_cfg.batch_size

        for epoch in range(epochs):
            log.info(f'Starting Epoch {epoch+1}/{epochs}')
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            end = time.time()

            num_batches = int(len(examples) / batch_size)

            if num_batches == 0:
                log.warning(f"Not enough examples ({len(examples)}) for a single batch (size {batch_size}). Skipping training epoch.")
                continue

            tqdm_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1} Training")

            for batch_idx in tqdm_bar:
                sample_ids = np.random.randint(len(examples), size=batch_size)
                boards, pis, vs, valids = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float32))
                target_pis = torch.FloatTensor(np.array(pis).astype(np.float32))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))
                target_valids = torch.FloatTensor(np.array(valids).astype(np.float32))

                if self.model_cfg.cuda:
                    boards, target_pis, target_vs, target_valids = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda(), target_valids.contiguous().cuda()

                data_time.update(time.time() - end)

                out_pi, out_v = self.nnet((boards, target_valids))

                l_pi = self.loss_pi(target_pis, out_pi, target_valids)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                batch_time.update(time.time() - end)
                end = time.time()

                tqdm_bar.set_postfix(Loss_pi=f'{pi_losses.avg:.4f}', Loss_v=f'{v_losses.avg:.4f}', LR=f'{self.optimizer.param_groups[0]["lr"]:.2e}')

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            log.info(f"Epoch {epoch+1} finished. Avg Pi Loss: {pi_losses.avg:.4f}, Avg V Loss: {pi_losses.avg:.4f}, LR: {current_lr:.2e}")

            if self.wandb_enabled:
                wandb.log({
                    'epoch': epoch + 1,
                    'avg_epoch_loss_pi': pi_losses.avg,
                    'avg_epoch_loss_v': v_losses.avg,
                    'avg_epoch_total_loss': pi_losses.avg + v_losses.avg,
                    'learning_rate': current_lr
                })

    def predict(self, boardAndValid):
        """Predict policy and value for a given board state."""
        board_array, valid_moves_mask = boardAndValid

        board_tensor = torch.FloatTensor(board_array.astype(np.float32))
        valid_moves_tensor = torch.FloatTensor(valid_moves_mask.astype(np.float32))
        if self.model_cfg.cuda:
            board_tensor = board_tensor.contiguous().cuda()
            valid_moves_tensor = valid_moves_tensor.contiguous().cuda()

        board_tensor = board_tensor.view(1, self.board_c, self.board_x, self.board_y)
        valid_moves_tensor = valid_moves_tensor.view(1, -1)

        self.nnet.eval()
        with torch.no_grad():
            pi_logits, v = self.nnet((board_tensor, valid_moves_tensor))

        pi_masked = pi_logits * valid_moves_tensor
        pi_masked[valid_moves_tensor == 0] = -1e9

        pi_probs = torch.softmax(pi_masked, dim=1)

        return pi_probs.data.cpu().numpy()[0], v.data.cpu().numpy()[0][0]


    def loss_pi(self, targets, outputs, masks):
        """Policy loss: Cross-entropy on valid moves."""
        log_probs = F.log_softmax(outputs, dim=1)
        loss = -torch.sum(targets * log_probs) / targets.size()[0]
        return loss


    def loss_v(self, targets, outputs):
        """Value loss: Mean Squared Error."""
        return torch.sum((targets - outputs.view(-1))**2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar') -> None:
        """Save weights checkpoint, ensuring directory exists."""
        filepath = os.path.join(folder, filename)
        try:
            os.makedirs(folder, exist_ok=True)
            log.info(f"Saving checkpoint to: {filepath}")
            save_data = {
                'state_dict': self.nnet.state_dict(),
                # Save optimizer/scheduler state only if they exist (training instance)
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                # Save model_cfg (architecture/device)
                'model_cfg': OmegaConf.to_container(self.model_cfg, resolve=True),
            }
            # Add optimizer/training config if they exist (Coach instance)
            if self.optimizer_cfg is not None:
                 save_data['optimizer_cfg'] = OmegaConf.to_container(self.optimizer_cfg, resolve=True)
            if self.training_cfg is not None:
                 save_data['training_cfg'] = OmegaConf.to_container(self.training_cfg, resolve=True)


            torch.save(save_data, filepath)

            if self.wandb_enabled:
                 try:
                     log.info(f"Logging checkpoint artifact to WandB: {filename}")
                     # Start with metadata from the configs managed directly by this NNet instance
                     artifact_metadata = OmegaConf.to_container(self.model_cfg, resolve=True)
                     if self.optimizer_cfg is not None:
                         artifact_metadata['optimizer_cfg'] = OmegaConf.to_container(self.optimizer_cfg, resolve=True)
                     if self.training_cfg is not None:
                         artifact_metadata['training_cfg'] = OmegaConf.to_container(self.training_cfg, resolve=True)

                     # If this instance is the main Coach's NNet (has the full cfg), add more context
                     if hasattr(self, 'cfg') and isinstance(self.cfg, DictConfig):
                          # Add a subset of the main cfg fields for context
                          main_cfg_subset_keys = ['run_name', 'project_name', 'game', 'mcts', 'arena', 'loading', 'inference'] # Include relevant top-level keys
                          main_cfg_subset_data = {k: self.cfg.get(k) for k in main_cfg_subset_keys if self.cfg.get(k) is not None}
                          # Only update if the subset is not empty
                          if main_cfg_subset_data:
                               artifact_metadata.update(OmegaConf.to_container(OmegaConf.create(main_cfg_subset_data), resolve=True))
                     else:
                          log.debug("NNet instance does not have full cfg or it's not DictConfig. Logging limited metadata.")


                     artifact = wandb.Artifact(f'model-{wandb.run.id}', type='model', metadata=artifact_metadata)
                     artifact.add_file(filepath)
                     wandb.log_artifact(artifact)
                 except Exception as e:
                     log.error(f"Failed to log checkpoint artifact to WandB: {e}")

        except Exception as e:
            log.error(f"Error saving checkpoint to {filepath}: {e}")

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar') -> None:
        """Load Weights, Optimizer, and Scheduler state."""
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            log.error(f"No model checkpoint found at: {filepath}")
            raise FileNotFoundError(f"No model checkpoint found at: {filepath}")

        log.info(f"Loading checkpoint from: {filepath}")
        # Device mapping needs access to the cuda setting, which is in self.model_cfg
        map_location = torch.device('cuda') if self.model_cfg.cuda and torch.cuda.is_available() else torch.device('cpu')

        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

        # Check if loaded model_cfg matches current model_cfg (optional but good practice)
        if 'model_cfg' in checkpoint:
            loaded_model_cfg = OmegaConf.create(checkpoint['model_cfg'])
            # Simple check for architecture compatibility
            if loaded_model_cfg.model.num_channels != self.model_cfg.model.num_channels or \
               loaded_model_cfg.model.num_res_blocks != self.model_cfg.model.num_res_blocks:
                log.warning(f"Loaded model architecture ({OmegaConf.to_yaml(loaded_model_cfg.model).strip()}) differs from current architecture ({OmegaConf.to_yaml(self.model_cfg.model).strip()}). Loading might fail or behave unexpectedly.")
        else:
            log.warning("Checkpoint does not contain 'model_cfg'. Cannot verify architecture.")


        try:
            self.nnet.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("Model state_dict loaded.")
        except KeyError:
            log.error("Checkpoint does not contain 'state_dict'. Cannot load model weights.")
        except Exception as e:
            log.error(f"Error loading model state_dict: {e}")


        # Load optimizer state only if this instance *has* an optimizer and the checkpoint *contains* it
        # This instance has optimizer only if setup_optimizer_scheduler was called (Coach instance)
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            try:
                # Verify optimizer config compatibility if needed
                if 'optimizer_cfg' in checkpoint:
                     loaded_optimizer_cfg = OmegaConf.create(checkpoint['optimizer_cfg'])
                     # Compare lr, weight_decay, etc.
                     if loaded_optimizer_cfg.lr != self.optimizer_cfg.lr or \
                        loaded_optimizer_cfg.weight_decay != self.optimizer_cfg.weight_decay:
                         log.warning("Loaded optimizer config differs from current optimizer config. Loading state might cause issues.")
                else:
                    log.warning("Checkpoint does not contain 'optimizer_cfg'. Cannot verify optimizer compatibility.")

                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                log.info("Optimizer state_dict loaded.")
            except Exception as e:
                log.warning(f"Could not load optimizer state_dict (might be due to parameter group changes or mismatch): {e}")

        # Load scheduler state only if this instance *has* a scheduler and the checkpoint *contains* it
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            try:
                 # Verify training config compatibility if needed for scheduler
                 if 'training_cfg' in checkpoint:
                      loaded_training_cfg = OmegaConf.create(checkpoint['training_cfg'])
                      # Compare epochs, lr_scheduler_step, lr_scheduler_gamma
                      if loaded_training_cfg.epochs != self.training_cfg.epochs:
                           log.warning("Loaded training epochs differ from current training epochs. Scheduler state might be inconsistent.")
                 else:
                     log.warning("Checkpoint does not contain 'training_cfg'. Cannot verify scheduler compatibility.")

                 self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 log.info("Scheduler state_dict loaded.")
            except Exception as e:
                log.warning(f"Could not load scheduler state_dict (might be due to missing scheduler or mismatch): {e}")


    def print(self) -> None:
        """Print current self object state"""
        device = 'cuda' if self.model_cfg.cuda else 'cpu'
        print(self.nnet.to(device))