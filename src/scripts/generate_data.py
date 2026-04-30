"""Data generation script for synthetic medical dataset."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

from src.data.medical_dataset import MedicalDomainDataset
from src.utils import setup_logging, load_config, create_directory_structure

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic medical dataset")
    
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
        "--force",
        action="store_true",
        help="Force regeneration even if data exists"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main data generation function."""
    args = parse_args()
    
    # Setup
    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directory structure
    create_directory_structure(args.data_dir)
    
    # Generate datasets for all splits
    splits = ["train", "val", "test"]
    
    for split in splits:
        logger.info(f"Generating {split} dataset...")
        
        dataset = MedicalDomainDataset(
            data_dir=args.data_dir,
            split=split,
            config=config.data,
            generate_synthetic=True
        )
        
        logger.info(f"Generated {len(dataset)} samples for {split} split")
    
    logger.info("Data generation completed successfully!")


if __name__ == "__main__":
    main()
