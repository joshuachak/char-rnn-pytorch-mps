"""
Checkpoint management utilities for training.

This module provides utilities for saving, loading, and managing model
checkpoints during training with compatibility for the original format.
"""

import torch
import os
import glob
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoints during training.
    
    This class provides:
    - Automatic checkpoint saving with validation loss in filename
    - Best model tracking
    - Checkpoint loading and resuming
    - Cleanup of old checkpoints
    - Compatibility with original Lua/Torch checkpoint format
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str = "char_rnn",
        max_checkpoints: int = 5,
        save_best_only: bool = False
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir (str): Directory to save checkpoints
            model_name (str): Name prefix for checkpoint files
            max_checkpoints (int): Maximum number of checkpoints to keep
            save_best_only (bool): If True, only save checkpoints that improve validation loss
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best validation loss
        self.best_val_loss = float('inf')
        self.best_checkpoint_path = None
        
        # Track all checkpoints
        self.checkpoints = []
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        iteration: int,
        val_loss: float,
        config: Any,
        vocab_mapping: Dict[str, int],
        train_losses: List[float],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a model checkpoint.
        
        Args:
            model: The model to save
            optimizer: The optimizer state
            scheduler: The learning rate scheduler
            epoch: Current epoch number
            iteration: Current iteration number
            val_loss: Current validation loss
            config: Training configuration
            vocab_mapping: Character to index mapping
            train_losses: List of training losses
            additional_info: Additional information to save
        
        Returns:
            str: Path to the saved checkpoint
        """
        # Check if we should save this checkpoint
        is_best = val_loss < self.best_val_loss
        
        if self.save_best_only and not is_best:
            return None
        
        # Create checkpoint filename similar to original format
        # Format: lm_<model_type>_epoch<epoch>_<val_loss>.pt
        filename = f"lm_{self.model_name}_epoch{epoch:.2f}_{val_loss:.4f}.pt"
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': config,
            'vocab_mapping': vocab_mapping,
            'train_losses': train_losses,
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__
        }
        
        # Add scheduler state if available
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add additional info if provided
        if additional_info:
            checkpoint_data.update(additional_info)
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update tracking
        self.checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'val_loss': val_loss,
            'is_best': is_best
        })
        
        # Update best checkpoint
        if is_best:
            self.best_val_loss = val_loss
            self.best_checkpoint_path = checkpoint_path
            print(f"New best checkpoint saved: {filename} (val_loss: {val_loss:.4f})")
        else:
            print(f"Checkpoint saved: {filename} (val_loss: {val_loss:.4f})")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore model state.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load tensors on
        
        Returns:
            dict: Checkpoint information and metadata
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint (use weights_only=False for our trusted checkpoints)
        if device:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        else:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded")
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        
        # Load scheduler state if provided
        if scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Scheduler state loaded")
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}")
        
        # Update best loss tracking
        if 'val_loss' in checkpoint:
            self.best_val_loss = checkpoint['val_loss']
        
        print(f"Checkpoint loaded: epoch {checkpoint.get('epoch', 'unknown')}, "
              f"val_loss {checkpoint.get('val_loss', 'unknown')}")
        
        return checkpoint
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get the path to the best checkpoint."""
        return str(self.best_checkpoint_path) if self.best_checkpoint_path else None
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the most recent checkpoint."""
        if not self.checkpoints:
            # Look for existing checkpoints in directory
            pattern = str(self.checkpoint_dir / f"lm_{self.model_name}_*.pt")
            existing = glob.glob(pattern)
            if existing:
                # Return the most recent one
                return max(existing, key=os.path.getctime)
        
        if self.checkpoints:
            return str(self.checkpoints[-1]['path'])
        
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Get list of all tracked checkpoints."""
        return self.checkpoints.copy()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints if we exceed the maximum number."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by validation loss (keep best) and recency
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: (not x['is_best'], x['val_loss'], -x['epoch'])
        )
        
        # Remove excess checkpoints
        checkpoints_to_remove = sorted_checkpoints[self.max_checkpoints:]
        
        for checkpoint_info in checkpoints_to_remove:
            checkpoint_path = checkpoint_info['path']
            if checkpoint_path.exists() and not checkpoint_info['is_best']:
                try:
                    checkpoint_path.unlink()
                    print(f"Removed old checkpoint: {checkpoint_path.name}")
                except Exception as e:
                    print(f"Warning: Could not remove checkpoint {checkpoint_path}: {e}")
            
            # Remove from tracking
            self.checkpoints.remove(checkpoint_info)
    
    def cleanup_all_checkpoints(self):
        """Remove all checkpoints (use with caution)."""
        for checkpoint_info in self.checkpoints:
            checkpoint_path = checkpoint_info['path']
            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                    print(f"Removed checkpoint: {checkpoint_path.name}")
                except Exception as e:
                    print(f"Warning: Could not remove checkpoint {checkpoint_path}: {e}")
        
        self.checkpoints.clear()
        self.best_val_loss = float('inf')
        self.best_checkpoint_path = None
    
    def convert_from_torch_checkpoint(self, torch_checkpoint_path: str, output_path: str):
        """
        Convert a Lua/Torch checkpoint to PyTorch format.
        
        Note: This is a placeholder for future implementation.
        The actual conversion would require lua and torch packages.
        
        Args:
            torch_checkpoint_path: Path to the original Lua/Torch checkpoint
            output_path: Path for the converted PyTorch checkpoint
        """
        raise NotImplementedError(
            "Torch checkpoint conversion not yet implemented. "
            "This would require lua and torch packages for reading the original format."
        )
