"""
LSTM model implementation for character-level language modeling.

This module provides a PyTorch implementation of multi-layer LSTM
that closely matches the behavior of the original Lua/Torch implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharLSTM(nn.Module):
    """
    Multi-layer LSTM for character-level language modeling.
    
    This implementation provides:
    - Multi-layer LSTM with configurable hidden size and number of layers
    - Dropout regularization between layers
    - One-hot input encoding via embedding
    - Log-softmax output for character prediction
    
    Args:
        vocab_size (int): Size of the character vocabulary
        hidden_size (int): Size of LSTM hidden state
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout probability between layers (default: 0.0)
    """
    
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.0):
        super(CharLSTM, self).__init__()
        
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
        
        # Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection layer
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        
        # Log softmax for probability output
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        # Initialize weights similar to original implementation
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to match original Lua/Torch implementation."""
        # Initialize LSTM weights with uniform distribution [-0.08, 0.08]
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -0.08, 0.08)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better initial performance
                # This matches the forget gate bias initialization in the original
                if 'bias_ih' in name:
                    # bias_ih contains [input_gate, forget_gate, output_gate, cell_gate]
                    hidden_size = param.size(0) // 4
                    param.data[hidden_size:2*hidden_size].fill_(1.0)
        
        # Initialize output projection
        nn.init.uniform_(self.output_proj.weight, -0.08, 0.08)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the LSTM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length)
                             containing character indices
            hidden (tuple, optional): Hidden state tuple (h0, c0) from previous timestep
        
        Returns:
            tuple: (output, hidden) where:
                - output: Log probabilities of shape (batch_size, seq_length, vocab_size)
                - hidden: Tuple (h_n, c_n) of final hidden states
        """
        batch_size, seq_length = x.size()
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # Embed input characters (equivalent to one-hot encoding)
        embedded = self.embedding(x)  # (batch_size, seq_length, vocab_size)
        
        # Pass through LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch_size, seq_length, hidden_size)
        
        # Project to vocabulary size
        output = self.output_proj(lstm_out)  # (batch_size, seq_length, vocab_size)
        
        # Apply log softmax
        output = self.log_softmax(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state for the LSTM.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to place tensors on
        
        Returns:
            tuple: (h0, c0) initial hidden states
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)
    
    def get_num_params(self):
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_embedding(self):
        """Freeze the embedding layer to maintain one-hot behavior."""
        self.embedding.weight.requires_grad = False
    
    def unfreeze_embedding(self):
        """Unfreeze the embedding layer to allow learning."""
        self.embedding.weight.requires_grad = True
