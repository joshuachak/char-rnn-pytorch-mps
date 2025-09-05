# Char-RNN Migration Plan: Lua/Torch → Python/PyTorch with MPS Support

## Overview

This document outlines the complete migration strategy for modernizing the 2015 char-rnn codebase from Lua/Torch to Python/PyTorch with Apple Silicon MPS acceleration support.

## Current Architecture Analysis

### Existing Components
1. **Training (`train.lua`)**: Multi-layer RNN training with RMSprop optimization
2. **Sampling (`sample.lua`)**: Character-level text generation with temperature control
3. **Models**: LSTM, GRU, and vanilla RNN implementations using nngraph
4. **Data Processing**: Character-level text preprocessing and batch loading
5. **Utilities**: Model cloning, parameter management, one-hot encoding

### Key Features to Preserve
- Multi-layer RNN support (LSTM, GRU, vanilla RNN)
- Character-level language modeling
- Checkpointing and resume capability  
- Temperature-controlled sampling
- Gradient clipping and dropout regularization
- Flexible data splits (train/val/test)
- Text priming for generation

## Migration Strategy

### 1. Modern Python Project Structure
```
char-rnn-pytorch/
├── requirements.txt           # Python dependencies
├── setup.py                  # Package setup
├── README.md                 # Updated documentation
├── config/
│   └── default.yaml          # Configuration files
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm.py          # PyTorch LSTM implementation
│   │   ├── gru.py           # PyTorch GRU implementation
│   │   └── rnn.py           # PyTorch vanilla RNN implementation
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataloader.py    # Character-level data loading
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py       # Training logic
│   │   └── checkpoints.py   # Checkpoint management
│   ├── generation/
│   │   ├── __init__.py
│   │   └── sampler.py       # Text generation
│   └── utils/
│       ├── __init__.py
│       ├── config.py        # Configuration handling
│       └── device.py        # Device management (CPU/CUDA/MPS)
├── scripts/
│   ├── train.py             # Training script
│   ├── sample.py            # Sampling script
│   └── convert_checkpoint.py # Checkpoint conversion utility
├── data/
│   └── tinyshakespeare/     # Existing data
└── checkpoints/             # Model checkpoints
```

### 2. Technology Stack Modernization

#### Core Dependencies
- **PyTorch 2.0+**: Modern deep learning framework
- **NumPy**: Numerical computing
- **PyYAML**: Configuration management
- **Tensorboard**: Training visualization
- **Matplotlib**: Plotting and visualization
- **Rich**: Enhanced CLI output
- **Click**: Command-line interface

#### Apple Silicon Optimization
- **MPS Backend**: Metal Performance Shaders for GPU acceleration
- **Optimized Data Loading**: Efficient preprocessing with multiprocessing
- **Memory Management**: Smart batching for Apple Silicon memory constraints

### 3. Model Architecture Updates

#### Enhanced Model Features
- **Attention Mechanisms**: Optional attention layers for better long-range dependencies
- **Layer Normalization**: Improved training stability
- **Modern Initialization**: Xavier/He initialization schemes
- **Flexible Architecture**: Easy switching between model types
- **Mixed Precision**: FP16 training support for faster computation

#### Backward Compatibility
- **Checkpoint Conversion**: Utility to convert Torch checkpoints to PyTorch
- **Parameter Mapping**: Preserve original model behavior
- **Evaluation Modes**: Ensure identical sampling behavior

### 4. Training Improvements

#### Modern Optimization
- **AdamW Optimizer**: Better generalization than RMSprop
- **Learning Rate Scheduling**: Cosine annealing, warm-up schedules
- **Gradient Accumulation**: Support for larger effective batch sizes
- **Early Stopping**: Automatic stopping based on validation metrics

#### Monitoring and Logging
- **TensorBoard Integration**: Real-time training visualization
- **Metrics Tracking**: Loss curves, perplexity, gradient norms
- **Checkpoint Management**: Automatic best model saving
- **Progress Bars**: Rich CLI progress indication

### 5. MPS Integration Strategy

#### Device Detection and Fallback
```python
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
```

#### Memory Management
- **Efficient Batching**: Optimal batch sizes for Apple Silicon
- **Memory Monitoring**: Track GPU memory usage
- **Fallback Mechanisms**: Graceful degradation to CPU when needed

#### Performance Optimization
- **Model Compilation**: PyTorch 2.0 compile for speed improvements
- **Data Loading**: Optimized preprocessing pipelines
- **Mixed Precision**: FP16 training where supported

## Implementation Phases

### Phase 1: Core Infrastructure (Days 1-2)
1. Set up Python project structure
2. Create configuration management system
3. Implement device detection and MPS support
4. Set up development environment and dependencies

### Phase 2: Data Pipeline (Day 3)
1. Convert character-level data loader to PyTorch
2. Implement efficient preprocessing
3. Add data validation and caching
4. Test with existing Shakespeare dataset

### Phase 3: Model Implementation (Days 4-5)
1. Convert LSTM model to PyTorch
2. Convert GRU model to PyTorch  
3. Convert vanilla RNN model to PyTorch
4. Add modern architectural improvements
5. Implement checkpoint conversion utility

### Phase 4: Training Pipeline (Days 6-7)
1. Convert training loop to PyTorch
2. Implement modern optimization techniques
3. Add TensorBoard logging and monitoring
4. Add checkpoint management and resume functionality

### Phase 5: Text Generation (Day 8)
1. Convert sampling script to PyTorch
2. Implement temperature control and nucleus sampling
3. Add text priming functionality
4. Create interactive generation interface

### Phase 6: Testing and Validation (Days 9-10)
1. Test all components on Apple Silicon
2. Validate output consistency with original
3. Performance benchmarking and optimization
4. Create comprehensive test suite

### Phase 7: Documentation and Polish (Day 11)
1. Update README with modern instructions
2. Create usage examples and tutorials
3. Add API documentation
4. Create migration guide from original

## Key Technical Considerations

### Model Equivalence
- Ensure mathematical equivalence between Lua/Torch and PyTorch implementations
- Validate gradient computations and weight updates
- Test with identical initialization for reproducible results

### Performance Optimization
- Leverage PyTorch's optimized kernels
- Use efficient data loading with DataLoader
- Implement model compilation for additional speedups
- Optimize for Apple Silicon's unified memory architecture

### Backward Compatibility
- Provide tools to convert existing Torch checkpoints
- Maintain CLI compatibility where possible
- Support legacy configuration formats

### Error Handling and Robustness
- Comprehensive error handling for device issues
- Graceful fallbacks when MPS is unavailable
- Informative error messages for common issues

## Success Metrics

1. **Functional Parity**: All original features working in PyTorch
2. **Performance**: Equal or better training speed on Apple Silicon
3. **Output Quality**: Identical text generation quality
4. **Usability**: Improved user experience with modern tooling
5. **Maintainability**: Clean, documented, testable code

## Risk Mitigation

### Technical Risks
- **MPS Compatibility**: Test extensively on different Apple Silicon variants
- **Memory Issues**: Implement smart batching and memory monitoring
- **Model Differences**: Rigorous validation against original implementation

### User Experience Risks
- **Migration Complexity**: Provide clear migration documentation
- **Breaking Changes**: Maintain backward compatibility where possible
- **Performance Regression**: Benchmark and optimize continuously

This migration plan ensures a systematic approach to modernizing the char-rnn codebase while preserving its educational value and extending its capabilities for modern hardware.
