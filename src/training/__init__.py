"""
Training pipeline and utilities.

This module contains the main training loop, checkpoint management,
and optimization utilities for training character-level RNNs.
"""

from .trainer import Trainer
from .checkpoints import CheckpointManager

__all__ = ["Trainer", "CheckpointManager"]
