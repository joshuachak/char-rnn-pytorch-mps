# Technical Specification: Char-RNN PyTorch Migration

## Model Architecture Specifications

### 1. LSTM Implementation

#### Original Lua/Torch Structure
```lua
-- Multi-layer LSTM with:
-- - One-hot input encoding
-- - Linear transformations for input-to-hidden and hidden-to-hidden
-- - Sigmoid gates (input, forget, output)
-- - Tanh activation for candidate values
-- - LogSoftmax output layer
```

#### PyTorch Implementation Specification
```python
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer replaces one-hot encoding for efficiency
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        
        # Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x, hidden=None):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.output_proj(lstm_out)
        return self.log_softmax(output), hidden
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)
```

#### Key Differences from Original
1. **Embedding Layer**: More efficient than one-hot encoding
2. **Batch First**: Modern convention for better performance
3. **Built-in LSTM**: PyTorch's optimized implementation
4. **Device Awareness**: Proper tensor placement for MPS/CUDA/CPU

### 2. GRU Implementation

#### PyTorch Implementation Specification
```python
class CharGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        
        self.gru = nn.GRU(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded, hidden)
        output = self.output_proj(gru_out)
        return self.log_softmax(output), hidden
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
```

### 3. Vanilla RNN Implementation

#### PyTorch Implementation Specification
```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        
        self.rnn = nn.RNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            nonlinearity='tanh'
        )
        
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded, hidden)
        output = self.output_proj(rnn_out)
        return self.log_softmax(output), hidden
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
```

## Data Pipeline Specification

### Character-Level Dataset Implementation

```python
class CharDataset(torch.utils.data.Dataset):
    def __init__(self, text_file, seq_length, train_frac=0.95, val_frac=0.05):
        # Load and preprocess text
        with open(text_file, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Create character vocabulary
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Convert text to indices
        self.data = [self.char_to_idx[ch] for ch in self.text]
        
        # Split data
        total_len = len(self.data)
        train_len = int(total_len * train_frac)
        val_len = int(total_len * val_frac)
        
        self.splits = {
            'train': self.data[:train_len],
            'val': self.data[train_len:train_len + val_len],
            'test': self.data[train_len + val_len:]
        }
        
        self.seq_length = seq_length
        self.current_split = 'train'
    
    def set_split(self, split):
        self.current_split = split
    
    def __len__(self):
        return len(self.splits[self.current_split]) - self.seq_length
    
    def __getitem__(self, idx):
        data = self.splits[self.current_split]
        x = torch.tensor(data[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y
```

### DataLoader Configuration

```python
def create_dataloaders(dataset, batch_size, num_workers=4):
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset.set_split(split)
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,  # For faster GPU transfer
            drop_last=(split == 'train')
        )
    
    return dataloaders
```

## Training Pipeline Specification

### Trainer Class Implementation

```python
class CharRNNTrainer:
    def __init__(self, model, dataloaders, config, device):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.config = config
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs
        )
        
        # Loss function
        self.criterion = nn.NLLLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # TensorBoard logging
        self.writer = SummaryWriter(log_dir=config.log_dir)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.dataloaders['train']):
            data, target = data.to(self.device), target.to(self.device)
            
            # Initialize hidden state
            hidden = self.model.init_hidden(data.size(0), self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output, hidden = self.model(data, hidden)
            loss = self.criterion(output.view(-1, self.model.vocab_size), target.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), 
                                     self.current_epoch * len(self.dataloaders['train']) + batch_idx)
        
        return total_loss / len(self.dataloaders['train'])
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in self.dataloaders['val']:
                data, target = data.to(self.device), target.to(self.device)
                hidden = self.model.init_hidden(data.size(0), self.device)
                
                output, hidden = self.model(data, hidden)
                loss = self.criterion(output.view(-1, self.model.vocab_size), target.view(-1))
                total_loss += loss.item()
        
        return total_loss / len(self.dataloaders['val'])
    
    def save_checkpoint(self, epoch, val_loss, filepath):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'vocab': self.dataloaders['train'].dataset.char_to_idx
        }
        torch.save(checkpoint, filepath)
```

## Text Generation Specification

### Sampling Implementation

```python
class TextGenerator:
    def __init__(self, model, char_to_idx, idx_to_char, device):
        self.model = model.to(device)
        self.model.eval()
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.device = device
    
    def generate(self, prime_text="", length=1000, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text using various sampling strategies.
        
        Args:
            prime_text: Initial text to prime the model
            length: Number of characters to generate
            temperature: Sampling temperature (lower = more conservative)
            top_k: Top-k sampling (only consider top k tokens)
            top_p: Nucleus sampling (consider tokens with cumulative prob <= p)
        """
        self.model.eval()
        
        # Initialize hidden state
        hidden = self.model.init_hidden(1, self.device)
        
        # Process prime text
        generated_text = prime_text
        input_char = prime_text[-1] if prime_text else random.choice(list(self.char_to_idx.keys()))
        
        with torch.no_grad():
            for _ in range(length):
                # Convert character to tensor
                input_tensor = torch.tensor([[self.char_to_idx[input_char]]], device=self.device)
                
                # Forward pass
                output, hidden = self.model(input_tensor, hidden)
                output = output.squeeze(0).squeeze(0)  # Remove batch and sequence dimensions
                
                # Apply temperature
                output = output / temperature
                
                # Apply sampling strategy
                if top_k is not None:
                    values, indices = torch.topk(output, top_k)
                    output = torch.full_like(output, float('-inf'))
                    output[indices] = values
                
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(output, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    output[indices_to_remove] = float('-inf')
                
                # Sample from the distribution
                probabilities = torch.softmax(output, dim=-1)
                next_char_idx = torch.multinomial(probabilities, 1).item()
                
                # Convert back to character
                input_char = self.idx_to_char[next_char_idx]
                generated_text += input_char
        
        return generated_text
```

## Device Management Specification

### MPS Integration

```python
class DeviceManager:
    def __init__(self):
        self.device = self._detect_device()
        self._log_device_info()
    
    def _detect_device(self):
        """Detect and return the best available device."""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _log_device_info(self):
        """Log information about the selected device."""
        if self.device.type == "mps":
            print(f"Using Metal Performance Shaders (MPS) on Apple Silicon")
        elif self.device.type == "cuda":
            print(f"Using CUDA: {torch.cuda.get_device_name()}")
        else:
            print("Using CPU")
    
    def get_optimal_batch_size(self, model_size, seq_length):
        """Get optimal batch size based on device and model constraints."""
        if self.device.type == "mps":
            # Conservative batch sizes for Apple Silicon
            if model_size < 1e6:  # Small model
                return min(64, 2048 // seq_length)
            elif model_size < 5e6:  # Medium model
                return min(32, 1024 // seq_length)
            else:  # Large model
                return min(16, 512 // seq_length)
        elif self.device.type == "cuda":
            # More aggressive batch sizes for CUDA
            return min(128, 4096 // seq_length)
        else:
            # Conservative for CPU
            return min(32, 512 // seq_length)
```

## Configuration Management

### YAML Configuration Schema

```yaml
# config/default.yaml
model:
  type: "lstm"  # lstm, gru, rnn
  hidden_size: 128
  num_layers: 2
  dropout: 0.2

training:
  batch_size: 50
  seq_length: 50
  learning_rate: 0.002
  weight_decay: 1e-5
  max_epochs: 50
  grad_clip: 5.0
  eval_interval: 1000

data:
  data_dir: "data/tinyshakespeare"
  train_frac: 0.95
  val_frac: 0.05

generation:
  temperature: 1.0
  top_k: null
  top_p: 0.9
  length: 1000

logging:
  log_dir: "logs"
  checkpoint_dir: "checkpoints"
  log_interval: 100
  save_interval: 1000

device:
  auto_detect: true
  preferred: "mps"  # mps, cuda, cpu
```

### Configuration Loading

```python
import yaml
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    type: str = "lstm"
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2

@dataclass
class TrainingConfig:
    batch_size: int = 50
    seq_length: int = 50
    learning_rate: float = 0.002
    weight_decay: float = 1e-5
    max_epochs: int = 50
    grad_clip: float = 5.0
    eval_interval: int = 1000

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # ... other config sections

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert nested dict to nested dataclasses
    return Config(**config_dict)
```

This technical specification provides the detailed implementation guidance needed to execute the migration plan effectively, ensuring mathematical equivalence with the original implementation while leveraging modern PyTorch capabilities and Apple Silicon optimization.
