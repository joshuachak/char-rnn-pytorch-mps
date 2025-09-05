"""
Configuration management utilities.

This module provides utilities for loading, validating, and managing
configuration files for the char-rnn training and generation pipeline.
"""

import yaml
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Union
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    type: str = "lstm"  # lstm, gru, rnn
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2

    def __post_init__(self):
        """Validate model configuration."""
        if self.type not in ["lstm", "gru", "rnn"]:
            raise ValueError(f"Invalid model type: {self.type}. Must be one of: lstm, gru, rnn")
        if self.hidden_size <= 0:
            raise ValueError(f"Hidden size must be positive, got: {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"Number of layers must be positive, got: {self.num_layers}")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"Dropout must be between 0 and 1, got: {self.dropout}")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 50
    seq_length: int = 50
    learning_rate: float = 0.002
    weight_decay: float = 1e-5
    max_epochs: int = 50
    grad_clip: float = 5.0
    eval_interval: int = 1000
    save_interval: int = 1000

    def __post_init__(self):
        """Validate training configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got: {self.batch_size}")
        if self.seq_length <= 0:
            raise ValueError(f"Sequence length must be positive, got: {self.seq_length}")
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got: {self.learning_rate}")
        if self.max_epochs <= 0:
            raise ValueError(f"Max epochs must be positive, got: {self.max_epochs}")
        if self.grad_clip <= 0:
            raise ValueError(f"Gradient clipping must be positive, got: {self.grad_clip}")


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_dir: str = "data/tinyshakespeare"
    train_frac: float = 0.9
    val_frac: float = 0.05
    num_workers: int = 4

    def __post_init__(self):
        """Validate data configuration."""
        if not 0 < self.train_frac < 1:
            raise ValueError(f"Train fraction must be between 0 and 1, got: {self.train_frac}")
        if not 0 < self.val_frac < 1:
            raise ValueError(f"Validation fraction must be between 0 and 1, got: {self.val_frac}")
        if self.train_frac + self.val_frac > 1:
            raise ValueError(f"Train + validation fractions must be <= 1, got: {self.train_frac + self.val_frac}")
        if self.num_workers < 0:
            raise ValueError(f"Number of workers must be non-negative, got: {self.num_workers}")

    @property
    def test_frac(self) -> float:
        """Calculate test fraction."""
        return max(0.0, 1.0 - self.train_frac - self.val_frac)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.9
    length: int = 1000
    prime_text: str = ""

    def __post_init__(self):
        """Validate generation configuration."""
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive, got: {self.temperature}")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"Top-k must be positive or None, got: {self.top_k}")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError(f"Top-p must be between 0 and 1 or None, got: {self.top_p}")
        if self.length <= 0:
            raise ValueError(f"Generation length must be positive, got: {self.length}")


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100
    enable_tensorboard: bool = True

    def __post_init__(self):
        """Validate logging configuration."""
        if self.log_interval <= 0:
            raise ValueError(f"Log interval must be positive, got: {self.log_interval}")


@dataclass
class DeviceConfig:
    """Configuration for device management."""
    auto_detect: bool = True
    preferred: str = "mps"  # mps, cuda, cpu
    mixed_precision: bool = False

    def __post_init__(self):
        """Validate device configuration."""
        if self.preferred not in ["mps", "cuda", "cpu"]:
            raise ValueError(f"Invalid preferred device: {self.preferred}. Must be one of: mps, cuda, cpu")


@dataclass
class Config:
    """Main configuration class containing all sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)

    def save(self, path: str):
        """Save configuration to YAML file."""
        config_dict = asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def update_from_dict(self, update_dict: Dict[str, Any]):
        """Update configuration from a dictionary."""
        for section, values in update_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                if isinstance(values, dict):
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            # Convert value to proper type based on the field annotation
                            converted_value = self._convert_value_type(section_obj, key, value)
                            setattr(section_obj, key, converted_value)
                        else:
                            print(f"Warning: Unknown config key {section}.{key}")
                else:
                    print(f"Warning: Config section {section} should be a dictionary")
            else:
                print(f"Warning: Unknown config section: {section}")
    
    def _convert_value_type(self, section_obj, key: str, value: Any) -> Any:
        """Convert value to the correct type based on field annotation."""
        if hasattr(section_obj.__class__, '__annotations__'):
            field_type = section_obj.__class__.__annotations__.get(key)
            if field_type is not None:
                try:
                    # Handle special cases
                    if field_type == float and isinstance(value, str):
                        return float(value)
                    elif field_type == int and isinstance(value, str):
                        return int(value)
                    elif field_type == bool and isinstance(value, str):
                        return value.lower() in ('true', '1', 'yes', 'on')
                    elif hasattr(field_type, '__origin__') and (
                        field_type.__origin__ is Union or 
                        (hasattr(field_type, '__args__') and type(None) in field_type.__args__)
                    ):
                        # Handle Optional types (e.g., Optional[int] which is Union[int, None])
                        if value is None:
                            return None
                        # Get the non-None type from the Union
                        inner_types = [t for t in field_type.__args__ if t is not type(None)]
                        if inner_types:
                            inner_type = inner_types[0]
                            if inner_type == float and isinstance(value, str):
                                return float(value)
                            elif inner_type == int and isinstance(value, str):
                                return int(value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert {key}={value} to {field_type}: {e}")
                    return value
        return value


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path (str): Path to the YAML configuration file
        overrides (dict, optional): Dictionary of configuration overrides
    
    Returns:
        Config: Loaded and validated configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML file
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        config_dict = {}
    
    # Create config object with nested dataclasses
    config = Config()
    
    # Update from loaded configuration
    if config_dict:
        config.update_from_dict(config_dict)
    
    # Apply overrides if provided
    if overrides:
        config.update_from_dict(overrides)
    
    return config


def create_default_config(save_path: str):
    """
    Create and save a default configuration file.
    
    Args:
        save_path (str): Path where to save the default configuration
    """
    config = Config()
    config.save(save_path)
    print(f"Default configuration saved to: {save_path}")


def validate_paths(config: Config, create_dirs: bool = True) -> bool:
    """
    Validate that all required paths exist and are accessible.
    
    Args:
        config (Config): Configuration to validate
        create_dirs (bool): Whether to create missing directories
    
    Returns:
        bool: True if all paths are valid
    """
    paths_to_check = [
        config.data.data_dir,
        config.logging.log_dir,
        config.logging.checkpoint_dir
    ]
    
    all_valid = True
    
    for path in paths_to_check:
        path_obj = Path(path)
        
        if not path_obj.exists():
            if create_dirs:
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    print(f"Created directory: {path}")
                except Exception as e:
                    print(f"Error creating directory {path}: {e}")
                    all_valid = False
            else:
                print(f"Path does not exist: {path}")
                all_valid = False
        elif not path_obj.is_dir():
            print(f"Path is not a directory: {path}")
            all_valid = False
    
    # Check if input.txt exists in data directory
    input_file = Path(config.data.data_dir) / "input.txt"
    if not input_file.exists():
        print(f"Warning: input.txt not found in {config.data.data_dir}")
        print("Make sure to place your training data in this file")
    
    return all_valid
