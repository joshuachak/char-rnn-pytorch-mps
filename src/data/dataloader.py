"""
Character-level dataset implementation for text data.

This module provides PyTorch Dataset and DataLoader implementations for
character-level text processing that matches the original Lua/Torch behavior.
"""

import torch
import torch.utils.data as data
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter


class CharDataset(data.Dataset):
    """
    Character-level dataset for text data.
    
    This dataset:
    - Loads text data and creates character-level vocabulary
    - Splits data into train/validation/test sets
    - Provides character sequences for training
    - Caches preprocessed data for efficiency
    
    Args:
        data_dir (str): Directory containing input.txt file
        seq_length (int): Length of character sequences
        train_frac (float): Fraction of data for training (default: 0.95)
        val_frac (float): Fraction of data for validation (default: 0.05)
        split (str): Which split to use ('train', 'val', 'test')
    """
    
    def __init__(
        self, 
        data_dir: str,
        seq_length: int,
        train_frac: float = 0.95,
        val_frac: float = 0.05,
        split: str = 'train'
    ):
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = 1.0 - train_frac - val_frac
        self.current_split = split
        
        # File paths
        self.input_file = self.data_dir / "input.txt"
        self.vocab_file = self.data_dir / "vocab.pkl"
        self.data_file = self.data_dir / "data.pkl"
        
        # Check if preprocessing is needed
        if self._needs_preprocessing():
            print("Preprocessing data...")
            self._preprocess_data()
        
        # Load preprocessed data
        self._load_data()
        
        # Set current split
        self.set_split(split)
    
    def _needs_preprocessing(self) -> bool:
        """Check if data preprocessing is needed."""
        if not self.vocab_file.exists() or not self.data_file.exists():
            return True
        
        # Check if input file is newer than processed files
        if self.input_file.exists():
            input_mtime = self.input_file.stat().st_mtime
            vocab_mtime = self.vocab_file.stat().st_mtime
            data_mtime = self.data_file.stat().st_mtime
            
            if input_mtime > vocab_mtime or input_mtime > data_mtime:
                return True
        
        return False
    
    def _preprocess_data(self):
        """Preprocess the input text file."""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        print(f"Loading text from {self.input_file}...")
        
        # Read the entire text file
        with open(self.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Text length: {len(text):,} characters")
        
        # Create character vocabulary
        print("Creating character vocabulary...")
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        
        print(f"Vocabulary size: {vocab_size} characters")
        print(f"Vocabulary: {''.join(chars[:50])}{'...' if vocab_size > 50 else ''}")
        
        # Create mappings
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Convert text to indices
        print("Converting text to indices...")
        data_indices = [char_to_idx[ch] for ch in text]
        
        # Save vocabulary
        vocab_data = {
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'chars': chars,
            'vocab_size': vocab_size
        }
        
        with open(self.vocab_file, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        # Save data
        with open(self.data_file, 'wb') as f:
            pickle.dump(data_indices, f)
        
        print(f"Saved vocabulary to {self.vocab_file}")
        print(f"Saved data to {self.data_file}")
    
    def _load_data(self):
        """Load preprocessed data and vocabulary."""
        # Load vocabulary
        with open(self.vocab_file, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = vocab_data['idx_to_char']
        self.chars = vocab_data['chars']
        self.vocab_size = vocab_data['vocab_size']
        
        # Load data
        with open(self.data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        # Create splits
        self._create_splits()
    
    def _create_splits(self):
        """Create train/validation/test splits."""
        total_len = len(self.data)
        
        # Calculate split sizes
        train_size = int(total_len * self.train_frac)
        val_size = int(total_len * self.val_frac)
        
        # Create splits
        self.splits = {
            'train': self.data[:train_size],
            'val': self.data[train_size:train_size + val_size],
            'test': self.data[train_size + val_size:]
        }
        
        # Print split information
        print(f"Data splits:")
        print(f"  Train: {len(self.splits['train']):,} characters")
        print(f"  Validation: {len(self.splits['val']):,} characters")
        print(f"  Test: {len(self.splits['test']):,} characters")
        
        # Calculate number of sequences in each split
        for split_name, split_data in self.splits.items():
            num_sequences = max(0, len(split_data) - self.seq_length)
            print(f"  {split_name.capitalize()} sequences: {num_sequences:,}")
    
    def set_split(self, split: str):
        """Set the current data split."""
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of: train, val, test")
        
        self.current_split = split
        self.current_data = self.splits[split]
    
    def __len__(self) -> int:
        """Return the number of sequences in the current split."""
        return max(0, len(self.current_data) - self.seq_length)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence pair (input, target) at the given index.
        
        Args:
            idx (int): Index of the sequence
        
        Returns:
            tuple: (input_sequence, target_sequence) as torch.Tensors
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for split '{self.current_split}' with {len(self)} sequences")
        
        # Get input sequence
        input_seq = self.current_data[idx:idx + self.seq_length]
        # Target sequence is shifted by one character
        target_seq = self.current_data[idx + 1:idx + self.seq_length + 1]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)
    
    def get_vocab_mapping(self) -> Dict[str, int]:
        """Get the character to index mapping."""
        return self.char_to_idx.copy()
    
    def get_inverse_vocab_mapping(self) -> Dict[int, str]:
        """Get the index to character mapping."""
        return self.idx_to_char.copy()
    
    def decode_sequence(self, indices: List[int]) -> str:
        """
        Decode a sequence of indices back to text.
        
        Args:
            indices (List[int]): List of character indices
        
        Returns:
            str: Decoded text
        """
        return ''.join(self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char)
    
    def encode_text(self, text: str) -> List[int]:
        """
        Encode text to a sequence of indices.
        
        Args:
            text (str): Text to encode
        
        Returns:
            List[int]: Encoded character indices
        """
        return [self.char_to_idx.get(ch, 0) for ch in text]  # Use 0 for unknown characters
    
    def get_random_sample(self, length: int = 100) -> str:
        """
        Get a random sample of text from the current split.
        
        Args:
            length (int): Length of the sample
        
        Returns:
            str: Random text sample
        """
        if len(self.current_data) < length:
            return self.decode_sequence(self.current_data)
        
        import random
        start_idx = random.randint(0, len(self.current_data) - length)
        sample_indices = self.current_data[start_idx:start_idx + length]
        return self.decode_sequence(sample_indices)


def create_dataloaders(
    data_dir: str,
    seq_length: int,
    batch_size: int,
    train_frac: float = 0.95,
    val_frac: float = 0.05,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Dict[str, data.DataLoader]:
    """
    Create DataLoaders for train, validation, and test splits.
    
    Args:
        data_dir (str): Directory containing input.txt
        seq_length (int): Length of character sequences
        batch_size (int): Batch size for training
        train_frac (float): Fraction of data for training
        val_frac (float): Fraction of data for validation
        num_workers (int): Number of worker processes
        pin_memory (bool): Whether to pin memory for faster GPU transfer
    
    Returns:
        dict: Dictionary of DataLoaders for each split
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = CharDataset(
            data_dir=data_dir,
            seq_length=seq_length,
            train_frac=train_frac,
            val_frac=val_frac,
            split=split
        )
        
        # Only shuffle training data
        shuffle = (split == 'train')
        
        # Create DataLoader
        dataloaders[split] = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train')  # Drop last incomplete batch for training
        )
        
        print(f"Created {split} DataLoader: {len(dataset)} sequences, {len(dataloaders[split])} batches")
    
    return dataloaders


def get_dataset_stats(data_dir: str) -> Dict[str, any]:
    """
    Get statistics about the dataset.
    
    Args:
        data_dir (str): Directory containing the data
    
    Returns:
        dict: Dataset statistics
    """
    # Create a temporary dataset to get stats
    dataset = CharDataset(data_dir, seq_length=50, split='train')
    
    # Character frequency analysis
    all_data = []
    for split_data in dataset.splits.values():
        all_data.extend(split_data)
    
    char_counts = Counter(all_data)
    most_common = char_counts.most_common(10)
    
    return {
        'vocab_size': dataset.vocab_size,
        'total_characters': len(all_data),
        'train_chars': len(dataset.splits['train']),
        'val_chars': len(dataset.splits['val']),
        'test_chars': len(dataset.splits['test']),
        'most_common_chars': [(dataset.idx_to_char[idx], count) for idx, count in most_common],
        'character_set': ''.join(dataset.chars)
    }
