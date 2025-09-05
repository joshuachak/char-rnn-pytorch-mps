"""
Training pipeline implementation for character-level RNNs.

This module provides the main training loop with modern PyTorch practices
while maintaining compatibility with the original Lua/Torch implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import math
from pathlib import Path
from typing import Dict, Optional, List, Any
from tqdm import tqdm

from training.checkpoints import CheckpointManager
from utils.device import DeviceManager
from utils.config import Config


class Trainer:
    """
    Main training class for character-level RNNs.
    
    This class provides:
    - Modern PyTorch training loop
    - Automatic mixed precision support
    - TensorBoard logging
    - Checkpoint management
    - Validation and early stopping
    - MPS/CUDA/CPU support
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        config: Config,
        device_manager: DeviceManager,
        resume_from: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The RNN model to train
            dataloaders: Dictionary of train/val/test dataloaders
            config: Training configuration
            device_manager: Device management utility
            resume_from: Path to checkpoint to resume from
        """
        self.config = config
        self.device = device_manager.device
        self.device_manager = device_manager
        
        # Model setup
        self.model = model.to(self.device)
        self.vocab_size = model.vocab_size
        
        # Data
        self.dataloaders = dataloaders
        self.train_dataset = dataloaders['train'].dataset
        
        # Optimization
        self._setup_optimization()
        
        # Logging and checkpointing
        self._setup_logging()
        self._setup_checkpointing()
        
        # Training state
        self.current_epoch = 0
        self.current_iteration = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Resume from checkpoint if specified
        if resume_from:
            self.resume_training(resume_from)
        
        print(f"Trainer initialized:")
        print(f"  Model: {model.__class__.__name__}")
        print(f"  Parameters: {model.get_num_params():,}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Sequence length: {config.training.seq_length}")
    
    def _setup_optimization(self):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer (using AdamW instead of RMSprop for better generalization)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.max_epochs,
            eta_min=self.config.training.learning_rate * 0.01
        )
        
        # Loss function
        self.criterion = nn.NLLLoss()
        
        # Mixed precision scaler (if supported)
        self.use_amp = (
            self.device_manager.supports_mixed_precision() and 
            self.config.device.mixed_precision
        )
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision training enabled")
    
    def _setup_logging(self):
        """Setup TensorBoard logging."""
        if self.config.logging.enable_tensorboard:
            log_dir = Path(self.config.logging.log_dir) / f"run_{int(time.time())}"
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging to: {log_dir}")
        else:
            self.writer = None
    
    def _setup_checkpointing(self):
        """Setup checkpoint manager."""
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.logging.checkpoint_dir,
            model_name=self.config.model.type,
            max_checkpoints=5
        )
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config.training.max_epochs} epochs...")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config.training.max_epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_loss = self._train_epoch()
                self.train_losses.append(train_loss)
                
                # Validation phase - run validation every epoch for now, or based on iterations
                validation_frequency = max(1, self.config.training.eval_interval // len(self.dataloaders['train']))
                if (epoch + 1) % validation_frequency == 0:
                    val_loss = self._validate()
                    self.val_losses.append(val_loss)
                    
                    # Save checkpoint  
                    save_frequency = max(1, self.config.training.save_interval // len(self.dataloaders['train']))
                    if (epoch + 1) % save_frequency == 0:
                        self._save_checkpoint(val_loss)
                    
                    # Early stopping check
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter = getattr(self, 'patience_counter', 0) + 1
                    
                    self.patience_counter = patience_counter
                
                # Update learning rate
                self.scheduler.step()
                
                # Log epoch summary
                if self.writer:
                    self._log_epoch_summary(epoch, train_loss)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            total_time = time.time() - start_time
            print(f"\nTraining completed in {total_time:.2f} seconds")
            
            # Save final checkpoint
            final_val_loss = self._validate()
            self._save_checkpoint(final_val_loss)
            
            if self.writer:
                self.writer.close()
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.dataloaders['train'])
        
        # Progress bar
        pbar = tqdm(
            self.dataloaders['train'],
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.max_epochs}",
            leave=False
        )
        
        for batch_idx, (data, target) in enumerate(pbar):
            self.current_iteration += 1
            
            # Move data to device
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            # Initialize hidden state
            hidden = self.model.init_hidden(data.size(0), self.device)
            
            # Forward pass with optional mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output, hidden = self.model(data, hidden)
                    loss = self.criterion(
                        output.view(-1, self.vocab_size),
                        target.view(-1)
                    )
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                # Standard precision training
                output, hidden = self.model(data, hidden)
                loss = self.criterion(
                    output.view(-1, self.vocab_size),
                    target.view(-1)
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip
                )
                
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Logging
            if self.writer and self.current_iteration % self.config.logging.log_interval == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.current_iteration)
                self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.current_iteration)
                
                # Log gradient norms
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                self.writer.add_scalar('Train/Gradient_Norm', total_norm, self.current_iteration)
        
        return epoch_loss / num_batches
    
    def _validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.dataloaders['val'])
        
        with torch.no_grad():
            for data, target in tqdm(self.dataloaders['val'], desc="Validation", leave=False):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # Initialize hidden state
                hidden = self.model.init_hidden(data.size(0), self.device)
                
                # Forward pass
                output, hidden = self.model(data, hidden)
                loss = self.criterion(
                    output.view(-1, self.vocab_size),
                    target.view(-1)
                )
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / num_batches
        
        # Calculate perplexity
        perplexity = math.exp(avg_val_loss)
        
        print(f"Validation - Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        if self.writer:
            self.writer.add_scalar('Val/Loss', avg_val_loss, self.current_epoch)
            self.writer.add_scalar('Val/Perplexity', perplexity, self.current_epoch)
        
        return avg_val_loss
    
    def _save_checkpoint(self, val_loss: float):
        """Save a training checkpoint."""
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            iteration=self.current_iteration,
            val_loss=val_loss,
            config=self.config,
            vocab_mapping=self.train_dataset.get_vocab_mapping(),
            train_losses=self.train_losses,
            additional_info={
                'val_losses': self.val_losses,
                'device': str(self.device),
                'model_class': self.model.__class__.__name__
            }
        )
    
    def _log_epoch_summary(self, epoch: int, train_loss: float):
        """Log epoch summary to TensorBoard."""
        if self.writer:
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            
            # Log memory usage if available
            memory_info = self.device_manager.get_memory_info()
            if isinstance(memory_info.get('allocated'), (int, float)):
                self.writer.add_scalar('System/Memory_Allocated_GB', memory_info['allocated'], epoch)
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        checkpoint_data = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        # Restore training state
        self.current_epoch = checkpoint_data.get('epoch', 0)
        self.current_iteration = checkpoint_data.get('iteration', 0)
        self.train_losses = checkpoint_data.get('train_losses', [])
        self.val_losses = checkpoint_data.get('val_losses', [])
        self.best_val_loss = checkpoint_data.get('val_loss', float('inf'))
        
        print(f"Resumed training from epoch {self.current_epoch}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the training progress."""
        return {
            'current_epoch': self.current_epoch,
            'current_iteration': self.current_iteration,
            'best_val_loss': self.best_val_loss,
            'num_parameters': self.model.get_num_params(),
            'device': str(self.device),
            'train_losses': self.train_losses[-10:],  # Last 10 training losses
            'val_losses': self.val_losses[-10:],      # Last 10 validation losses
        }
