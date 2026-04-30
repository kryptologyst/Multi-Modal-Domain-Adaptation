"""Main training script for multi-modal domain adaptation."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.models.clip_domain_adaptation import CLIPDomainAdaptationModel
from src.data.medical_dataset import create_data_loaders
from src.training.trainer import DomainAdaptationTrainer
from src.utils import (
    set_seed, get_device, setup_logging, load_config, 
    count_parameters, print_system_info, create_directory_structure
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multi-modal domain adaptation model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Path to output directory"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_logging()
    print_system_info()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.device = args.device
    config.seed = args.seed
    
    # Create output directory structure
    create_directory_structure(args.output_dir)
    
    # Get device
    device = get_device(args.device)
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        config=config.data,
        processor=None,  # Will be created with model
        batch_size=config.data.get("batch_size", 32),
        num_workers=config.data.get("num_workers", 4),
    )
    
    # Create model
    logger.info("Creating model...")
    model = CLIPDomainAdaptationModel(
        base_model_name=config.model.base_model_name,
        config=config.model
    )
    
    # Set processor for data loaders
    train_loader.dataset.processor = model.processor
    val_loader.dataset.processor = model.processor
    test_loader.dataset.processor = model.processor
    
    # Print model info
    count_parameters(model, trainable_only=False)
    
    # Create trainer
    trainer = DomainAdaptationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        device=device,
        output_dir=Path(args.output_dir)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))
    
    # Train model
    training_history = trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate_model(test_loader)
    
    # Save training history
    import json
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
