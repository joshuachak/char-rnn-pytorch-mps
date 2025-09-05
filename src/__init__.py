"""
Char-RNN PyTorch: Modern character-level RNN for text generation
===============================================================

A modernized PyTorch implementation of Andrej Karpathy's char-rnn with
support for Apple Silicon MPS acceleration.

Key Features:
- LSTM, GRU, and vanilla RNN models
- Apple Silicon MPS support
- Modern PyTorch training pipeline
- Flexible text generation with multiple sampling strategies
- TensorBoard logging and monitoring
"""

__version__ = "2.0.0"
__author__ = "Modernized from Andrej Karpathy's char-rnn"

from .models import CharLSTM, CharGRU, CharRNN
from .data import CharDataset
from .training import Trainer
from .generation import TextGenerator
from .utils import DeviceManager, load_config

__all__ = [
    "CharLSTM",
    "CharGRU", 
    "CharRNN",
    "CharDataset",
    "Trainer",
    "TextGenerator",
    "DeviceManager",
    "load_config",
]
