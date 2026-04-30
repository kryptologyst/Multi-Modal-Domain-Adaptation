"""Evaluation script for multi-modal domain adaptation."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.models.clip_domain_adaptation import CLIPDomainAdaptationModel
from src.data.medical_dataset import create_data_loaders
from src.eval.domain_adaptation_evaluator import DomainAdaptationEvaluator
from src.utils import (
    set_seed, get_device, setup_logging, load_config, 
    print_system_info
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate multi-modal domain adaptation model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    args = parse_args()
    
    # Setup
    set_seed(42)
    setup_logging()
    print_system_info()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    device = get_device(args.device)
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        config=config.data,
        processor=None,  # Will be created with model
        batch_size=config.evaluation.get("batch_size", 64),
        num_workers=config.evaluation.get("num_workers", 4),
    )
    
    # Select data loader based on split
    if args.split == "train":
        data_loader = train_loader
    elif args.split == "val":
        data_loader = val_loader
    else:
        data_loader = test_loader
    
    # Create model
    logger.info("Creating model...")
    model = CLIPDomainAdaptationModel(
        base_model_name=config.model.base_model_name,
        config=config.model
    )
    
    # Set processor for data loader
    data_loader.dataset.processor = model.processor
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # Create evaluator
    evaluator = DomainAdaptationEvaluator(config.evaluation)
    
    # Run evaluation
    logger.info(f"Evaluating on {args.split} split...")
    metrics = evaluator.evaluate(
        model=model,
        data_loader=data_loader,
        device=device,
        compute_embeddings=True
    )
    
    # Print results
    logger.info(f"Evaluation Results ({args.split} split):")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualizations...")
        
        # Collect embeddings for visualization
        model.eval()
        image_embeddings = []
        text_embeddings = []
        domain_ids = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(
                    input_ids=batch["text"],
                    pixel_values=batch["image"],
                    attention_mask=batch.get("attention_mask"),
                    domain_ids=batch.get("domain"),
                    return_embeddings=True
                )
                
                image_embeddings.append(outputs["image_embeds"].cpu())
                text_embeddings.append(outputs["text_embeds"].cpu())
                domain_ids.append(batch["domain"].cpu())
        
        # Concatenate embeddings
        image_embeddings = torch.cat(image_embeddings, dim=0)
        text_embeddings = torch.cat(text_embeddings, dim=0)
        domain_ids = torch.cat(domain_ids, dim=0)
        
        # Create visualization
        output_dir = Path(args.output_dir)
        viz_path = output_dir / f"embeddings_visualization_{args.split}.png"
        
        evaluator.visualize_embeddings(
            image_embeds=image_embeddings,
            text_embeds=text_embeddings,
            domain_ids=domain_ids,
            save_path=str(viz_path)
        )
        
        logger.info(f"Visualization saved to {viz_path}")
    
    # Save metrics
    import json
    metrics_path = Path(args.output_dir) / f"evaluation_metrics_{args.split}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
