#!/usr/bin/env python3
"""
Training script for character-level RNN models.

This script provides a command-line interface for training LSTM, GRU, or vanilla RNN
models on character-level text data with modern PyTorch features and MPS support.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from rich.console import Console
from rich.table import Table
import yaml

from models import CharLSTM, CharGRU, CharRNN
from data import create_dataloaders, get_dataset_stats
from training import Trainer
from utils import DeviceManager, load_config, validate_paths


def create_model(model_type: str, vocab_size: int, config) -> torch.nn.Module:
    """Create a model based on the specified type."""
    model_args = {
        'vocab_size': vocab_size,
        'hidden_size': config.model.hidden_size,
        'num_layers': config.model.num_layers,
        'dropout': config.model.dropout
    }
    
    if model_type.lower() == 'lstm':
        return CharLSTM(**model_args)
    elif model_type.lower() == 'gru':
        return CharGRU(**model_args)
    elif model_type.lower() == 'rnn':
        return CharRNN(**model_args)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def print_training_info(config, device_manager, model, dataloaders):
    """Print training information using Rich formatting."""
    console = Console()
    
    # Model information table
    model_table = Table(title="Model Configuration")
    model_table.add_column("Parameter", style="cyan")
    model_table.add_column("Value", style="white")
    
    model_table.add_row("Model Type", config.model.type.upper())
    model_table.add_row("Hidden Size", str(config.model.hidden_size))
    model_table.add_row("Number of Layers", str(config.model.num_layers))
    model_table.add_row("Dropout", str(config.model.dropout))
    model_table.add_row("Total Parameters", f"{model.get_num_params():,}")
    
    console.print(model_table)
    
    # Training configuration table
    train_table = Table(title="Training Configuration")
    train_table.add_column("Parameter", style="cyan")
    train_table.add_column("Value", style="white")
    
    train_table.add_row("Batch Size", str(config.training.batch_size))
    train_table.add_row("Sequence Length", str(config.training.seq_length))
    train_table.add_row("Learning Rate", str(config.training.learning_rate))
    train_table.add_row("Max Epochs", str(config.training.max_epochs))
    train_table.add_row("Gradient Clipping", str(config.training.grad_clip))
    
    console.print(train_table)
    
    # Device information table
    device_table = Table(title="Device Configuration")
    device_table.add_column("Parameter", style="cyan")
    device_table.add_column("Value", style="white")
    
    device_table.add_row("Device", str(device_manager.device))
    device_table.add_row("Mixed Precision", str(device_manager.supports_mixed_precision()))
    
    # Add memory info if available
    memory_info = device_manager.get_memory_info()
    if isinstance(memory_info.get('total'), (int, float)):
        device_table.add_row("Total Memory", f"{memory_info['total']:.1f} GB")
    
    console.print(device_table)
    
    # Dataset information table
    dataset_table = Table(title="Dataset Information")
    dataset_table.add_column("Split", style="cyan")
    dataset_table.add_column("Sequences", style="white")
    dataset_table.add_column("Batches", style="white")
    
    for split, dataloader in dataloaders.items():
        dataset_table.add_row(
            split.capitalize(),
            f"{len(dataloader.dataset):,}",
            f"{len(dataloader):,}"
        )
    
    console.print(dataset_table)


def main():
    parser = argparse.ArgumentParser(description="Train a character-level RNN")
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file'
    )
    
    # Data
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Data directory (overrides config)'
    )
    
    # Model
    parser.add_argument(
        '--model-type',
        choices=['lstm', 'gru', 'rnn'],
        help='Model type (overrides config)'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        help='Hidden size (overrides config)'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        help='Number of layers (overrides config)'
    )
    
    # Training
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        help='Sequence length (overrides config)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        help='Maximum epochs (overrides config)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='Device to use for training'
    )
    
    # Resuming
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    
    # Output
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Checkpoint directory (overrides config)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        help='Log directory (overrides config)'
    )
    
    # Utilities
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit without training'
    )
    parser.add_argument(
        '--dataset-stats',
        action='store_true',
        help='Print dataset statistics and exit'
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    try:
        # Load configuration
        console.print(f"[bold blue]Loading configuration from {args.config}[/bold blue]")
        config = load_config(args.config)
        
        # Apply command line overrides
        if args.data_dir:
            config.data.data_dir = args.data_dir
        if args.model_type:
            config.model.type = args.model_type
        if args.hidden_size:
            config.model.hidden_size = args.hidden_size
        if args.num_layers:
            config.model.num_layers = args.num_layers
        if args.batch_size:
            config.training.batch_size = args.batch_size
        if args.seq_length:
            config.training.seq_length = args.seq_length
        if args.learning_rate:
            config.training.learning_rate = args.learning_rate
        if args.max_epochs:
            config.training.max_epochs = args.max_epochs
        if args.checkpoint_dir:
            config.logging.checkpoint_dir = args.checkpoint_dir
        if args.log_dir:
            config.logging.log_dir = args.log_dir
        if args.device != 'auto':
            config.device.preferred = args.device
            config.device.auto_detect = False
        
        # Validate paths
        console.print("[bold blue]Validating paths...[/bold blue]")
        if not validate_paths(config, create_dirs=True):
            console.print("[bold red]Path validation failed![/bold red]")
            sys.exit(1)
        
        # Print dataset statistics if requested
        if args.dataset_stats:
            console.print("[bold blue]Generating dataset statistics...[/bold blue]")
            stats = get_dataset_stats(config.data.data_dir)
            
            stats_table = Table(title="Dataset Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            
            stats_table.add_row("Vocabulary Size", str(stats['vocab_size']))
            stats_table.add_row("Total Characters", f"{stats['total_characters']:,}")
            stats_table.add_row("Train Characters", f"{stats['train_chars']:,}")
            stats_table.add_row("Validation Characters", f"{stats['val_chars']:,}")
            stats_table.add_row("Test Characters", f"{stats['test_chars']:,}")
            
            console.print(stats_table)
            
            console.print(f"\\n[bold]Most common characters:[/bold]")
            for char, count in stats['most_common_chars']:
                if char == ' ':
                    char_display = '<space>'
                elif char == '\\n':
                    char_display = '<newline>'
                elif char == '\\t':
                    char_display = '<tab>'
                else:
                    char_display = char
                console.print(f"  '{char_display}': {count:,} times")
            
            return
        
        # Setup device manager
        preferred_device = config.device.preferred if not config.device.auto_detect else None
        device_manager = DeviceManager(preferred_device=preferred_device)
        
        # Adjust batch size based on device capabilities
        console.print("[bold blue]Optimizing batch size for device...[/bold blue]")
        estimated_model_size = (
            config.model.hidden_size * config.model.num_layers * 4 * 1000  # Rough estimate
        )
        optimal_batch_size = device_manager.get_optimal_batch_size(
            estimated_model_size, config.training.seq_length, config.training.batch_size
        )
        
        if optimal_batch_size != config.training.batch_size:
            console.print(f"[yellow]Adjusting batch size from {config.training.batch_size} to {optimal_batch_size} for optimal performance[/yellow]")
            config.training.batch_size = optimal_batch_size
        
        # Create data loaders
        console.print("[bold blue]Creating data loaders...[/bold blue]")
        dataloaders = create_dataloaders(
            data_dir=config.data.data_dir,
            seq_length=config.training.seq_length,
            batch_size=config.training.batch_size,
            train_frac=config.data.train_frac,
            val_frac=config.data.val_frac,
            num_workers=config.data.num_workers,
            pin_memory=device_manager.get_recommended_settings()['pin_memory']
        )
        
        # Get vocabulary size from dataset
        vocab_size = dataloaders['train'].dataset.vocab_size
        
        # Create model
        console.print(f"[bold blue]Creating {config.model.type.upper()} model...[/bold blue]")
        model = create_model(config.model.type, vocab_size, config)
        
        # Print training information
        print_training_info(config, device_manager, model, dataloaders)
        
        if args.dry_run:
            console.print("[bold yellow]Dry run mode - exiting without training[/bold yellow]")
            return
        
        # Create trainer
        console.print("[bold blue]Initializing trainer...[/bold blue]")
        trainer = Trainer(
            model=model,
            dataloaders=dataloaders,
            config=config,
            device_manager=device_manager,
            resume_from=args.resume
        )
        
        # Start training
        console.print("[bold green]Starting training...[/bold green]")
        trainer.train()
        
        # Print training summary
        summary = trainer.get_training_summary()
        
        summary_table = Table(title="Training Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Final Epoch", str(summary['current_epoch']))
        summary_table.add_row("Total Iterations", f"{summary['current_iteration']:,}")
        summary_table.add_row("Best Validation Loss", f"{summary['best_val_loss']:.4f}")
        summary_table.add_row("Model Parameters", f"{summary['num_parameters']:,}")
        
        console.print(summary_table)
        
        # Print best checkpoint path
        best_checkpoint = trainer.checkpoint_manager.get_best_checkpoint()
        if best_checkpoint:
            console.print(f"[bold green]Best checkpoint saved at: {best_checkpoint}[/bold green]")
        
        console.print("[bold green]Training completed successfully![/bold green]")
    
    except KeyboardInterrupt:
        console.print("[bold yellow]Training interrupted by user[/bold yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Training failed with error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
