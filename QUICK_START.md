# 🚀 Quick Start Guide - Modernized Char-RNN

## Get Started in 3 Commands!

Your modernized char-rnn is ready to use with Apple Silicon MPS acceleration. Here's how to get started immediately:

### 1. ⚡ Train a Model (5-10 minutes on Apple Silicon)

```bash
# Train a small LSTM model on Shakespeare data
cd /Users/admin/Documents/char-rnn
python scripts/train.py --model-type lstm --hidden-size 128 --max-epochs 10
```

This will:
- Use your Apple Silicon GPU (MPS) automatically
- Train on the included Shakespeare dataset
- Save checkpoints to `checkpoints/`
- Show beautiful progress bars and training info

### 2. 🎨 Generate Text

```bash
# Generate 500 characters using your trained model
python scripts/sample.py checkpoints/lm_lstm_epoch*.pt --length 500
```

Example output:
```
ROMEO:
What shall I do? I am not well; I have
A thousand times more than I can bear.

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
```

### 3. 🎮 Interactive Mode

```bash
# Start interactive generation session
python scripts/sample.py checkpoints/lm_lstm_epoch*.pt --interactive
```

Then try commands like:
- `temp 0.8` - Set creativity level
- `length 200` - Set generation length  
- Type "To be or not to be" and press Enter
- `quit` to exit

## 🎯 Quick Examples

### Training Variations

```bash
# Fast training (small model)
python scripts/train.py --hidden-size 64 --max-epochs 5

# High quality (larger model)
python scripts/train.py --hidden-size 256 --num-layers 3 --max-epochs 25

# Different model types
python scripts/train.py --model-type gru    # Use GRU instead of LSTM
python scripts/train.py --model-type rnn    # Use vanilla RNN

# Check your data
python scripts/train.py --dataset-stats     # See dataset information
```

### Generation Variations

```bash
# Conservative generation (low creativity)
python scripts/sample.py model.pt --temperature 0.5

# Creative generation (high creativity)
python scripts/sample.py model.pt --temperature 1.5

# Primed generation
python scripts/sample.py model.pt --prime-text "HAMLET:" --length 300

# Modern sampling (higher quality)
python scripts/sample.py model.pt --top-p 0.9 --temperature 0.8

# Save to file
python scripts/sample.py model.pt --output my_generated_text.txt
```

## 📊 Monitor Training

### TensorBoard (Real-time Visualization)
```bash
# In another terminal, start TensorBoard
tensorboard --logdir logs

# Open http://localhost:6006 in your browser
```

### Rich CLI Output
The training script shows beautiful tables with:
- Model configuration (parameters, architecture)
- Training progress (loss curves, speed)
- Device information (MPS status, memory)
- Dataset statistics (vocabulary, splits)

## 🍎 Apple Silicon Specific

### Verify MPS is Working
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Optimal Settings for Apple Silicon
```bash
# Recommended for M1/M2 MacBooks
python scripts/train.py \
  --batch-size 32 \
  --hidden-size 256 \
  --num-layers 2 \
  --max-epochs 20
```

## 🔧 Use Your Own Data

1. **Prepare your text file:**
   ```bash
   mkdir data/my_data
   # Put your text in data/my_data/input.txt
   ```

2. **Train on your data:**
   ```bash
   python scripts/train.py --data-dir data/my_data
   ```

## 🐛 Troubleshooting

### Common Issues & Quick Fixes

**"MPS not available"**
```bash
# Use CPU instead
python scripts/train.py --device cpu
```

**"Out of memory"**
```bash
# Reduce batch size
python scripts/train.py --batch-size 16
```

**"Training too slow"**
```bash
# Use smaller model
python scripts/train.py --hidden-size 64 --num-layers 1
```

**"Poor text quality"**
```bash
# Train longer or use larger model
python scripts/train.py --hidden-size 256 --max-epochs 50
```

## 📚 What's Different from Original?

### ✅ Better
- **Apple Silicon**: Native MPS GPU acceleration
- **UI**: Beautiful terminal output with progress bars
- **Sampling**: Modern techniques (top-k, nucleus)
- **Config**: YAML files instead of command-line only
- **Monitoring**: TensorBoard integration

### 🔄 Same
- **Models**: Identical LSTM/GRU/RNN architectures
- **Training**: Same mathematical operations
- **Quality**: Equivalent text generation quality
- **Data**: Uses same Shakespeare dataset

## 🎓 Next Steps

1. **Experiment with hyperparameters**
2. **Try your own datasets**  
3. **Compare different model types (LSTM vs GRU vs RNN)**
4. **Explore advanced sampling techniques**
5. **Use TensorBoard to understand training dynamics**

---

**Happy text generation! 🎉**

Your modernized char-rnn is ready to create amazing text with the power of Apple Silicon!
