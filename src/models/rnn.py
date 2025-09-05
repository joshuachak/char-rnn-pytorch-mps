"""
Vanilla RNN model implementation for character-level language modeling.

This module provides a PyTorch implementation of multi-layer vanilla RNN
that closely matches the behavior of the original Lua/Torch implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharRNN(nn.Module):
    """
    Multi-layer vanilla RNN for character-level language modeling.
    
    This implementation provides:
    - Multi-layer RNN with configurable hidden size and number of layers
    - Dropout regularization between layers
    - One-hot input encoding via embedding
    - Log-softmax output for character prediction
    - Tanh activation function
    
    Args:
        vocab_size (int): Size of the character vocabulary
        hidden_size (int): Size of RNN hidden state
        num_layers (int): Number of RNN layers
        dropout (float): Dropout probability between layers (default: 0.0)
    """
    
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.0):
        super(CharRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout
        
        # Embedding layer to replace one-hot encoding for efficiency
        # We use vocab_size as embedding dimension to match one-hot behavior
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        
        # Initialize embedding as identity matrix (one-hot equivalent)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(vocab_size))
        
        # Multi-layer vanilla RNN with tanh activation
        self.rnn = nn.RNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            nonlinearity='tanh'  # Explicit tanh activation
        )
        
        # Output projection layer
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        
        # Log softmax for probability output
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        # Initialize weights similar to original implementation
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to match original Lua/Torch implementation."""
        # Initialize RNN weights with uniform distribution [-0.08, 0.08]
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -0.08, 0.08)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize output projection
        nn.init.uniform_(self.output_proj.weight, -0.08, 0.08)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the RNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length)
                             containing character indices
            hidden (torch.Tensor, optional): Hidden state from previous timestep
        
        Returns:
            tuple: (output, hidden) where:
                - output: Log probabilities of shape (batch_size, seq_length, vocab_size)
                - hidden: Final hidden state of shape (num_layers, batch_size, hidden_size)
        """
        batch_size, seq_length = x.size()
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # Embed input characters (equivalent to one-hot encoding)
        embedded = self.embedding(x)  # (batch_size, seq_length, vocab_size)
        
        # Pass through RNN
        rnn_out, hidden = self.rnn(embedded, hidden)  # (batch_size, seq_length, hidden_size)
        
        # Project to vocabulary size
        output = self.output_proj(rnn_out)  # (batch_size, seq_length, vocab_size)
        
        # Apply log softmax
        output = self.log_softmax(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state for the RNN.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to place tensors on
        
        Returns:
            torch.Tensor: Initial hidden state of shape (num_layers, batch_size, hidden_size)
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
    
    def get_num_params(self):
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_embedding(self):
        """Freeze the embedding layer to maintain one-hot behavior."""
        self.embedding.weight.requires_grad = False
    
    def unfreeze_embedding(self):
        """Unfreeze the embedding layer to allow learning."""
        self.embedding.weight.requires_grad = True
