#!/usr/bin/env python3
"""
Sampling script for character-level RNN models.

This script provides a command-line interface for generating text using
trained LSTM, GRU, or vanilla RNN models with various sampling strategies.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
import time

from models import CharLSTM, CharGRU, CharRNN
from generation import TextGenerator
from utils import DeviceManager


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    console = Console()
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    console.print(f"[bold blue]Loading checkpoint from {checkpoint_path}[/bold blue]")
    
    # Load checkpoint (use weights_only=False for our trusted checkpoints)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model configuration
    config = checkpoint['config']
    vocab_mapping = checkpoint['vocab_mapping']
    vocab_size = len(vocab_mapping)
    
    # Create inverse vocabulary mapping
    idx_to_char = {i: char for char, i in vocab_mapping.items()}
    
    # Create model based on type
    model_type = config.model.type.lower()
    model_args = {
        'vocab_size': vocab_size,
        'hidden_size': config.model.hidden_size,
        'num_layers': config.model.num_layers,
        'dropout': config.model.dropout
    }
    
    if model_type == 'lstm':
        model = CharLSTM(**model_args)
    elif model_type == 'gru':
        model = CharGRU(**model_args)
    elif model_type == 'rnn':
        model = CharRNN(**model_args)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    console.print(f"[green]Model loaded successfully:[/green]")
    console.print(f"  Type: {model_type.upper()}")
    console.print(f"  Parameters: {model.get_num_params():,}")
    console.print(f"  Vocabulary size: {vocab_size}")
    console.print(f"  Training epoch: {checkpoint.get('epoch', 'unknown')}")
    console.print(f"  Validation loss: {checkpoint.get('val_loss', 'unknown')}")
    
    return model, vocab_mapping, idx_to_char, config


def format_generated_text(text: str, console: Console):
    """Format and display generated text beautifully."""
    # Create a panel with the generated text
    panel = Panel(
        text,
        title="Generated Text",
        title_align="left",
        border_style="green",
        padding=(1, 2)
    )
    console.print(panel)


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained character-level RNN")
    
    # Required arguments
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to the trained model checkpoint'
    )
    
    # Generation parameters
    parser.add_argument(
        '--length', '-l',
        type=int,
        default=1000,
        help='Number of characters to generate (default: 1000)'
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=1.0,
        help='Sampling temperature (default: 1.0)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        help='Nucleus (top-p) sampling parameter'
    )
    parser.add_argument(
        '--prime-text', '-p',
        type=str,
        default="",
        help='Text to prime the model with'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible generation'
    )
    
    # Generation mode
    parser.add_argument(
        '--sample',
        action='store_true',
        default=True,
        help='Use sampling (default)'
    )
    parser.add_argument(
        '--argmax',
        action='store_true',
        help='Use argmax (deterministic) instead of sampling'
    )
    
    # Device
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='Device to use for generation'
    )
    
    # Interactive mode
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive generation session'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save generated text to file'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Only output generated text, no other messages'
    )
    
    # Analysis options
    parser.add_argument(
        '--perplexity',
        type=str,
        help='Calculate perplexity on the given text file'
    )
    
    args = parser.parse_args()
    
    console = Console() if not args.quiet else Console(file=open(os.devnull, 'w'))
    
    try:
        # Setup device manager
        if args.device == 'auto':
            device_manager = DeviceManager()
        else:
            device_manager = DeviceManager(preferred_device=args.device)
        
        # Load model
        model, char_to_idx, idx_to_char, config = load_model_from_checkpoint(
            args.checkpoint, device_manager.device
        )
        
        # Create text generator
        generator = TextGenerator(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            device=device_manager.device
        )
        
        # Handle perplexity calculation
        if args.perplexity:
            if not os.path.exists(args.perplexity):
                console.print(f"[bold red]Text file not found: {args.perplexity}[/bold red]")
                sys.exit(1)
            
            with open(args.perplexity, 'r', encoding='utf-8') as f:
                text = f.read()
            
            console.print(f"[bold blue]Calculating perplexity on {args.perplexity}...[/bold blue]")
            perplexity = generator.calculate_perplexity(text)
            console.print(f"[bold green]Perplexity: {perplexity:.2f}[/bold green]")
            return
        
        # Handle interactive mode
        if args.interactive:
            generator.interactive_generation(
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            return
        
        # Single generation
        sample_mode = not args.argmax
        
        if not args.quiet:
            console.print(f"[bold blue]Generating {args.length} characters...[/bold blue]")
            console.print(f"  Temperature: {args.temperature}")
            console.print(f"  Top-k: {args.top_k}")
            console.print(f"  Top-p: {args.top_p}")
            console.print(f"  Prime text: '{args.prime_text}'")
            console.print(f"  Sampling: {sample_mode}")
            console.print(f"  Seed: {args.seed}")
            console.print()
        
        # Generate text
        start_time = time.time()
        
        generated_text = generator.generate(
            length=args.length,
            prime_text=args.prime_text,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample=sample_mode,
            seed=args.seed
        )
        
        generation_time = time.time() - start_time
        
        if not args.quiet:
            console.print(f"[green]Generated {len(generated_text)} characters in {generation_time:.2f} seconds[/green]")
            console.print(f"[green]Generation speed: {len(generated_text) / generation_time:.1f} chars/sec[/green]")
            console.print()
        
        # Display or save generated text
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            console.print(f"[bold green]Generated text saved to {args.output}[/bold green]")
        else:
            if args.quiet:
                # Just print the generated text
                print(generated_text)
            else:
                # Format and display nicely
                format_generated_text(generated_text, console)
    
    except KeyboardInterrupt:
        console.print("[bold yellow]Generation interrupted by user[/bold yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Generation failed with error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
