"""
Device management utilities for cross-platform support.

This module provides utilities for detecting and managing different compute
devices (CPU, CUDA, MPS) with special focus on Apple Silicon MPS support.
"""

import torch
import platform
from typing import Optional


class DeviceManager:
    """
    Manages device detection and optimization for different hardware configurations.
    
    This class provides:
    - Automatic device detection (MPS > CUDA > CPU)
    - Device-specific optimization recommendations
    - Memory management utilities
    - Performance monitoring
    """
    
    def __init__(self, preferred_device: Optional[str] = None):
        """
        Initialize the device manager.
        
        Args:
            preferred_device (str, optional): Preferred device type ('mps', 'cuda', 'cpu')
                                            If None, auto-detect best available device
        """
        self.preferred_device = preferred_device
        self.device = self._detect_device()
        self._log_device_info()
        
    def _detect_device(self) -> torch.device:
        """
        Detect and return the best available device.
        
        Priority order: MPS > CUDA > CPU
        
        Returns:
            torch.device: The selected device
        """
        # If user specified a preferred device, try to use it
        if self.preferred_device:
            if self.preferred_device.lower() == "mps" and self._is_mps_available():
                return torch.device("mps")
            elif self.preferred_device.lower() == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif self.preferred_device.lower() == "cpu":
                return torch.device("cpu")
            else:
                print(f"Warning: Preferred device '{self.preferred_device}' not available. Auto-detecting...")
        
        # Auto-detect best available device
        if self._is_mps_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _is_mps_available(self) -> bool:
        """
        Check if MPS (Metal Performance Shaders) is available.
        
        Returns:
            bool: True if MPS is available and working
        """
        try:
            # Check if we're on macOS
            if platform.system() != "Darwin":
                return False
            
            # Check if MPS backend is available
            if not torch.backends.mps.is_available():
                return False
            
            # Check if MPS is built
            if not torch.backends.mps.is_built():
                return False
            
            # Try to create a tensor on MPS to verify it works
            test_tensor = torch.tensor([1.0], device="mps")
            return True
            
        except Exception as e:
            print(f"MPS check failed: {e}")
            return False
    
    def _log_device_info(self):
        """Log information about the selected device."""
        device_info = self._get_device_info()
        print(f"Device: {device_info}")
    
    def _get_device_info(self) -> str:
        """Get detailed information about the current device."""
        if self.device.type == "mps":
            return f"Metal Performance Shaders (MPS) on {platform.processor()} - Apple Silicon"
        elif self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(self.device.index or 0)
            gpu_memory = torch.cuda.get_device_properties(self.device.index or 0).total_memory / 1e9
            return f"CUDA: {gpu_name} ({gpu_memory:.1f} GB)"
        else:
            return f"CPU: {platform.processor()}"
    
    def get_optimal_batch_size(self, model_size: int, seq_length: int, base_batch_size: int = 50) -> int:
        """
        Get optimal batch size based on device capabilities and model size.
        
        Args:
            model_size (int): Number of parameters in the model
            seq_length (int): Sequence length
            base_batch_size (int): Base batch size from configuration
        
        Returns:
            int: Recommended batch size
        """
        if self.device.type == "mps":
            # Conservative batch sizes for Apple Silicon
            # MPS has unified memory but can be sensitive to memory pressure
            if model_size < 1e6:  # Small model (<1M parameters)
                max_batch = min(128, 4096 // seq_length)
            elif model_size < 5e6:  # Medium model (<5M parameters)
                max_batch = min(64, 2048 // seq_length)
            else:  # Large model
                max_batch = min(32, 1024 // seq_length)
                
        elif self.device.type == "cuda":
            # More aggressive batch sizes for dedicated CUDA GPUs
            try:
                # Get available GPU memory
                gpu_memory = torch.cuda.get_device_properties(self.device.index or 0).total_memory
                if gpu_memory > 8e9:  # > 8GB
                    max_batch = min(256, 8192 // seq_length)
                elif gpu_memory > 4e9:  # > 4GB
                    max_batch = min(128, 4096 // seq_length)
                else:  # <= 4GB
                    max_batch = min(64, 2048 // seq_length)
            except:
                max_batch = min(128, 4096 // seq_length)
                
        else:  # CPU
            # Conservative for CPU to avoid memory issues
            max_batch = min(32, 1024 // seq_length)
        
        # Return the minimum of configured batch size and device-optimal batch size
        return min(base_batch_size, max(1, max_batch))
    
    def get_memory_info(self) -> dict:
        """
        Get current memory usage information.
        
        Returns:
            dict: Memory information for the current device
        """
        if self.device.type == "cuda":
            return {
                'allocated': torch.cuda.memory_allocated(self.device) / 1e9,
                'cached': torch.cuda.memory_reserved(self.device) / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1e9,
                'total': torch.cuda.get_device_properties(self.device).total_memory / 1e9
            }
        elif self.device.type == "mps":
            # MPS doesn't have memory tracking APIs yet
            return {
                'allocated': 'N/A (MPS)',
                'cached': 'N/A (MPS)', 
                'max_allocated': 'N/A (MPS)',
                'total': 'Unified Memory'
            }
        else:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'allocated': 'N/A (CPU)',
                'cached': 'N/A (CPU)',
                'max_allocated': 'N/A (CPU)',
                'total': memory.total / 1e9,
                'available': memory.available / 1e9
            }
    
    def clear_cache(self):
        """Clear device memory cache if supported."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            print("CUDA cache cleared")
        elif self.device.type == "mps":
            # MPS doesn't have explicit cache clearing yet
            print("MPS: Memory management is handled automatically")
        else:
            print("CPU: No cache to clear")
    
    def supports_mixed_precision(self) -> bool:
        """
        Check if the device supports mixed precision training.
        
        Returns:
            bool: True if mixed precision is supported
        """
        if self.device.type == "cuda":
            # Check for tensor cores (modern NVIDIA GPUs)
            return torch.cuda.get_device_capability(self.device.index or 0)[0] >= 7
        elif self.device.type == "mps":
            # MPS support for FP16 is limited and experimental
            return False
        else:
            return False
    
    def get_recommended_settings(self) -> dict:
        """
        Get recommended training settings for the current device.
        
        Returns:
            dict: Recommended settings
        """
        return {
            'device': str(self.device),
            'mixed_precision': self.supports_mixed_precision(),
            'num_workers': 4 if self.device.type in ['cuda', 'mps'] else 2,
            'pin_memory': self.device.type == 'cuda',  # Only CUDA supports pin_memory
            'non_blocking': self.device.type in ['cuda', 'mps'],
        }
