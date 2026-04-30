"""Training utilities and trainer for domain adaptation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np

from src.models.clip_domain_adaptation import CLIPDomainAdaptationModel
from src.losses.domain_adaptation_losses import MultiModalDomainAdaptationLoss
from src.eval.domain_adaptation_evaluator import DomainAdaptationEvaluator

logger = logging.getLogger(__name__)


class DomainAdaptationTrainer:
    """Trainer for multi-modal domain adaptation models.
    
    Handles training loop, validation, checkpointing, and logging
    for domain adaptation experiments.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        output_dir: Directory to save outputs
    """

    def __init__(
        self,
        model: CLIPDomainAdaptationModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        output_dir: Path,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Create output directories
        self.checkpoint_dir = output_dir / "checkpoints"
        self.log_dir = output_dir / "logs"
        self.asset_dir = output_dir / "assets"
        
        for dir_path in [self.checkpoint_dir, self.log_dir, self.asset_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize loss function
        loss_config = config.get("loss_config", {})
        self.criterion = MultiModalDomainAdaptationLoss(**loss_config)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize evaluator
        self.evaluator = DomainAdaptationEvaluator()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Move model to device
        self.model.to(device)
        
        logger.info(f"Initialized trainer with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_config = self.config.get("optimizer", {})
        
        return AdamW(
            self.model.parameters(),
            lr=optimizer_config.get("lr", 1e-4),
            weight_decay=optimizer_config.get("weight_decay", 0.01),
            betas=optimizer_config.get("betas", [0.9, 0.999]),
            eps=optimizer_config.get("eps", 1e-8),
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        scheduler_config = self.config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "cosine_with_warmup")
        
        if scheduler_type == "cosine_with_warmup":
            warmup_steps = scheduler_config.get("warmup_steps", 1000)
            total_steps = len(self.train_loader) * self.config.get("num_epochs", 50)
            
            # Warmup scheduler
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
            # Cosine scheduler
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=1e-6
            )
            
            # Combined scheduler
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        
        elif scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get("num_epochs", 50),
                eta_min=1e-6
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def train(self) -> Dict[str, Any]:
        """Run training loop."""
        logger.info("Starting training...")
        
        training_history = {
            "train_losses": [],
            "val_losses": [],
            "val_metrics": [],
            "learning_rates": [],
        }
        
        num_epochs = self.config.get("num_epochs", 50)
        validation_frequency = self.config.get("validation", {}).get("frequency", 1)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch()
            training_history["train_losses"].append(train_loss)
            
            # Validation phase
            if epoch % validation_frequency == 0:
                val_metrics = self._validate_epoch()
                training_history["val_losses"].append(val_metrics.get("total_loss", 0))
                training_history["val_metrics"].append(val_metrics)
                
                # Check for improvement
                if val_metrics.get("total_loss", float('inf')) < self.best_val_loss:
                    self.best_val_loss = val_metrics.get("total_loss", float('inf'))
                    self.patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                early_stopping_config = self.config.get("early_stopping", {})
                if early_stopping_config.get("patience", 10) <= self.patience_counter:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Learning rate tracking
            current_lr = self.optimizer.param_groups[0]["lr"]
            training_history["learning_rates"].append(current_lr)
            
            # Logging
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics.get('total_loss', 0):.4f}, "
                f"LR: {current_lr:.6f}"
            )
        
        # Save final checkpoint
        self._save_checkpoint(is_final=True)
        
        logger.info("Training completed!")
        return training_history

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch["text"],
                pixel_values=batch["image"],
                attention_mask=batch.get("attention_mask"),
                domain_ids=batch.get("domain"),
                return_embeddings=True
            )
            
            # Compute loss
            loss_outputs = self.criterion(
                image_embeds=outputs["image_embeds"],
                text_embeds=outputs["text_embeds"],
                domain_ids=batch.get("domain"),
                model=self.model
            )
            
            loss = loss_outputs["total_loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            gradient_clip_norm = self.config.get("gradient_clip_norm", 1.0)
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "LR": f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        return total_loss / num_batches

    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        with torch.no_grad():
            val_metrics = self.evaluator.evaluate(
                model=self.model,
                data_loader=self.val_loader,
                device=self.device,
                compute_embeddings=True
            )
        
        return val_metrics

    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {self.current_epoch}")
        
        # Save final checkpoint
        if is_final:
            final_path = self.checkpoint_dir / "final_model.pt"
            torch.save(checkpoint, final_path)
            logger.info(f"Saved final model at epoch {self.current_epoch}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set."""
        logger.info("Evaluating model on test set...")
        
        test_metrics = self.evaluator.evaluate(
            model=self.model,
            data_loader=test_loader,
            device=self.device,
            compute_embeddings=True
        )
        
        # Log results
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return test_metrics
