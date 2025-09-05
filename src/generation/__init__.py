"""
Text generation and sampling utilities.

This module provides text generation capabilities with various sampling
strategies including temperature control, top-k, and nucleus sampling.
"""

from .sampler import TextGenerator

__all__ = ["TextGenerator"]
