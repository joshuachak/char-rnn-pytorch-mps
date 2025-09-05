"""
Text generation and sampling utilities.

This module provides text generation capabilities with various sampling
strategies including temperature control, top-k, and nucleus sampling.
"""

import torch
import torch.nn.functional as F
import random
import math
from typing import Dict, Optional, List
import numpy as np


class TextGenerator:
    """
    Text generator for character-level language models.
    
    This class provides:
    - Temperature-controlled sampling
    - Top-k sampling
    - Nucleus (top-p) sampling
    - Deterministic (argmax) generation
    - Text priming support
    - Compatible with original Lua/Torch sampling behavior
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        char_to_idx: Dict[str, int],
        idx_to_char: Dict[int, str],
        device: torch.device
    ):
        """
        Initialize the text generator.
        
        Args:
            model: Trained character-level RNN model
            char_to_idx: Character to index mapping
            idx_to_char: Index to character mapping
            device: Device to run generation on
        """
        self.model = model.to(device)
        self.model.eval()
        
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = len(char_to_idx)
        self.device = device
        
        print(f"Text generator initialized:")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Device: {device}")
    
    def generate(
        self,
        length: int = 1000,
        prime_text: str = "",
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        sample: bool = True,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate text using the trained model.
        
        Args:
            length: Number of characters to generate
            prime_text: Initial text to prime the model
            temperature: Sampling temperature (lower = more conservative)
            top_k: Top-k sampling (only consider top k tokens)
            top_p: Nucleus sampling (consider tokens with cumulative prob <= p)
            sample: If False, use argmax instead of sampling
            seed: Random seed for reproducible generation
        
        Returns:
            str: Generated text
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        
        self.model.eval()
        
        with torch.no_grad():
            # Initialize hidden state
            hidden = self.model.init_hidden(1, self.device)
            
            # Process prime text if provided
            generated_text = prime_text
            
            if prime_text:
                print(f"Priming with: '{prime_text}'")
                
                # Process each character in prime text
                for char in prime_text:
                    if char in self.char_to_idx:
                        char_idx = torch.tensor([[self.char_to_idx[char]]], device=self.device)
                        output, hidden = self.model(char_idx, hidden)
                
                # Use the last character as starting point
                if prime_text[-1] in self.char_to_idx:
                    current_input = torch.tensor([[self.char_to_idx[prime_text[-1]]]], device=self.device)
                else:
                    # Fallback to random character if last char not in vocab
                    current_input = torch.tensor([[random.randint(0, self.vocab_size - 1)]], device=self.device)
            else:
                # Start with random character if no prime text
                current_input = torch.tensor([[random.randint(0, self.vocab_size - 1)]], device=self.device)
            
            # Generate characters
            for _ in range(length):
                # Forward pass
                output, hidden = self.model(current_input, hidden)
                
                # Get predictions for the last (and only) time step
                predictions = output.squeeze(0).squeeze(0)  # Remove batch and sequence dimensions
                
                # Sample next character
                if sample:
                    next_char_idx = self._sample_next_char(
                        predictions, temperature, top_k, top_p
                    )
                else:
                    # Use argmax (deterministic)
                    next_char_idx = torch.argmax(predictions).item()
                
                # Convert to character and add to generated text
                if next_char_idx in self.idx_to_char:
                    next_char = self.idx_to_char[next_char_idx]
                    generated_text += next_char
                
                # Update input for next iteration
                current_input = torch.tensor([[next_char_idx]], device=self.device)
            
            return generated_text
    
    def _sample_next_char(
        self,
        predictions: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> int:
        """
        Sample the next character using various strategies.
        
        Args:
            predictions: Log probabilities from the model
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
        
        Returns:
            int: Index of the sampled character
        """
        # Apply temperature
        if temperature != 1.0:
            predictions = predictions / temperature
        
        # Convert log probabilities to probabilities
        probabilities = F.softmax(predictions, dim=-1)
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, probabilities.size(-1))  # Ensure top_k doesn't exceed vocab size
            
            # Get top-k values and indices
            values, indices = torch.topk(probabilities, top_k)
            
            # Create a mask for top-k tokens
            probabilities_filtered = torch.zeros_like(probabilities)
            probabilities_filtered[indices] = values
            probabilities = probabilities_filtered
        
        # Apply nucleus (top-p) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            
            # Compute cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Set probabilities to 0 for tokens to remove
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probabilities[indices_to_remove] = 0.0
        
        # Renormalize probabilities
        probabilities = probabilities / probabilities.sum()
        
        # Handle edge case where all probabilities are 0
        if probabilities.sum() == 0:
            # Fallback to uniform distribution
            probabilities = torch.ones_like(probabilities) / probabilities.size(-1)
        
        # Sample from the distribution
        try:
            next_char_idx = torch.multinomial(probabilities, 1).item()
        except RuntimeError:
            # Fallback to argmax if sampling fails
            next_char_idx = torch.argmax(probabilities).item()
        
        return next_char_idx
    
    def interactive_generation(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ):
        """
        Interactive text generation session.
        
        Args:
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
        """
        print("Interactive Text Generation")
        print("Commands:")
        print("  'quit' or 'exit' - Exit the session")
        print("  'temp <value>' - Set temperature")
        print("  'length <value>' - Set generation length")
        print("  'sample' or 'argmax' - Set sampling mode")
        print("  Otherwise, enter text to use as primer")
        print()
        
        length = 200
        sample_mode = True
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                if user_input.startswith('temp '):
                    try:
                        temperature = float(user_input.split()[1])
                        print(f"Temperature set to {temperature}")
                        continue
                    except (IndexError, ValueError):
                        print("Invalid temperature value")
                        continue
                
                if user_input.startswith('length '):
                    try:
                        length = int(user_input.split()[1])
                        print(f"Length set to {length}")
                        continue
                    except (IndexError, ValueError):
                        print("Invalid length value")
                        continue
                
                if user_input.lower() == 'sample':
                    sample_mode = True
                    print("Sampling mode enabled")
                    continue
                
                if user_input.lower() == 'argmax':
                    sample_mode = False
                    print("Argmax mode enabled")
                    continue
                
                # Generate text with user input as primer
                print(f"\nGenerating {length} characters...")
                print("-" * 50)
                
                generated = self.generate(
                    length=length,
                    prime_text=user_input,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sample=sample_mode
                )
                
                print(generated)
                print("-" * 50)
                print()
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of the model on given text.
        
        Args:
            text: Text to calculate perplexity on
        
        Returns:
            float: Perplexity value
        """
        self.model.eval()
        
        # Convert text to indices
        try:
            indices = [self.char_to_idx[char] for char in text if char in self.char_to_idx]
        except KeyError as e:
            print(f"Warning: Character {e} not in vocabulary")
            return float('inf')
        
        if len(indices) < 2:
            return float('inf')
        
        total_log_prob = 0.0
        count = 0
        
        with torch.no_grad():
            hidden = self.model.init_hidden(1, self.device)
            
            for i in range(len(indices) - 1):
                current_char = torch.tensor([[indices[i]]], device=self.device)
                target_char = indices[i + 1]
                
                output, hidden = self.model(current_char, hidden)
                log_probs = output.squeeze(0).squeeze(0)  # Remove batch and sequence dims
                
                total_log_prob += log_probs[target_char].item()
                count += 1
        
        if count == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / count
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    def generate_with_constraints(
        self,
        length: int,
        prime_text: str = "",
        forbidden_chars: Optional[List[str]] = None,
        required_chars: Optional[List[str]] = None,
        temperature: float = 1.0
    ) -> str:
        """
        Generate text with character constraints.
        
        Args:
            length: Number of characters to generate
            prime_text: Initial text to prime the model
            forbidden_chars: Characters to avoid in generation
            required_chars: Characters that must appear in generation
            temperature: Sampling temperature
        
        Returns:
            str: Generated text respecting constraints
        """
        # Convert character constraints to indices
        forbidden_indices = set()
        if forbidden_chars:
            for char in forbidden_chars:
                if char in self.char_to_idx:
                    forbidden_indices.add(self.char_to_idx[char])
        
        required_indices = set()
        if required_chars:
            for char in required_chars:
                if char in self.char_to_idx:
                    required_indices.add(self.char_to_idx[char])
        
        self.model.eval()
        
        with torch.no_grad():
            hidden = self.model.init_hidden(1, self.device)
            generated_text = prime_text
            generated_indices = set()
            
            # Process prime text
            if prime_text:
                for char in prime_text:
                    if char in self.char_to_idx:
                        char_idx = torch.tensor([[self.char_to_idx[char]]], device=self.device)
                        output, hidden = self.model(char_idx, hidden)
                        generated_indices.add(self.char_to_idx[char])
                
                current_input = torch.tensor([[self.char_to_idx[prime_text[-1]]]], device=self.device)
            else:
                current_input = torch.tensor([[random.randint(0, self.vocab_size - 1)]], device=self.device)
            
            for i in range(length):
                output, hidden = self.model(current_input, hidden)
                predictions = output.squeeze(0).squeeze(0)
                
                # Apply temperature
                predictions = predictions / temperature
                probabilities = F.softmax(predictions, dim=-1)
                
                # Apply forbidden character constraint
                if forbidden_indices:
                    for idx in forbidden_indices:
                        probabilities[idx] = 0.0
                
                # Boost required characters if not yet seen
                remaining_required = required_indices - generated_indices
                if remaining_required and i > length * 0.8:  # Boost in later part of generation
                    for idx in remaining_required:
                        probabilities[idx] *= 2.0  # Boost probability
                
                # Renormalize
                probabilities = probabilities / probabilities.sum()
                
                # Sample
                next_char_idx = torch.multinomial(probabilities, 1).item()
                generated_indices.add(next_char_idx)
                
                if next_char_idx in self.idx_to_char:
                    generated_text += self.idx_to_char[next_char_idx]
                
                current_input = torch.tensor([[next_char_idx]], device=self.device)
            
            return generated_text
