"""Utility functions for multi-modal domain adaptation."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    logger.info(f"Set random seed to {seed}")


def get_device(device_preference: str = "auto") -> torch.device:
    """Get the best available device.
    
    Args:
        device_preference: Device preference ("auto", "cuda", "mps", "cpu")
        
    Returns:
        PyTorch device
    """
    if device_preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        device = torch.device(device_preference)
        logger.info(f"Using specified device: {device}")
    
    return device


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging setup complete (level: {log_level})")


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    return config


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    OmegaConf.save(config, output_path)
    logger.info(f"Saved configuration to {output_path}")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> Dict[str, int]:
    """Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    result = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
    }
    
    if trainable_only:
        logger.info(f"Trainable parameters: {trainable_params:,}")
    else:
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {result['frozen_parameters']:,}")
    
    return result


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_directory_structure(base_dir: Union[str, Path]) -> None:
    """Create standard directory structure for the project.
    
    Args:
        base_dir: Base directory path
    """
    base_dir = Path(base_dir)
    
    directories = [
        "data/images/general",
        "data/images/medical", 
        "data/audio",
        "data/video",
        "data/text",
        "configs/model",
        "configs/data",
        "configs/training",
        "configs/evaluation",
        "src/data",
        "src/models",
        "src/losses",
        "src/eval",
        "src/viz",
        "src/utils",
        "src/scripts",
        "tests",
        "demo",
        "assets",
        "notebooks",
        "outputs/checkpoints",
        "outputs/logs",
        "outputs/assets",
    ]
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directory structure in {base_dir}")


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def enable_mixed_precision() -> bool:
    """Check if mixed precision training is available.
    
    Returns:
        True if mixed precision is available
    """
    return torch.cuda.is_available() and hasattr(torch.cuda, "amp")


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information.
    
    Returns:
        Dictionary with GPU memory info in MB
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    memory_info = {
        "available": True,
        "total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
        "allocated_memory_mb": torch.cuda.memory_allocated(0) / 1024 / 1024,
        "cached_memory_mb": torch.cuda.memory_reserved(0) / 1024 / 1024,
    }
    
    memory_info["free_memory_mb"] = (
        memory_info["total_memory_mb"] - memory_info["allocated_memory_mb"]
    )
    
    return memory_info


def print_system_info() -> None:
    """Print system information for debugging."""
    logger.info("System Information:")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if hasattr(torch.backends, "mps"):
        logger.info(f"  MPS available: {torch.backends.mps.is_available()}")
    
    # Memory info
    memory_info = get_gpu_memory_info()
    if memory_info["available"]:
        logger.info(f"  GPU memory: {memory_info['free_memory_mb']:.1f}MB free / {memory_info['total_memory_mb']:.1f}MB total")
