"""Loss functions for multi-modal domain adaptation."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for multi-modal learning.
    
    Implements InfoNCE loss for aligning image and text embeddings
    with support for hard negative mining and temperature scaling.
    
    Args:
        temperature: Temperature parameter for scaling similarities
        use_hard_negatives: Whether to use hard negative mining
        hard_negative_ratio: Ratio of hard negatives to use
        margin: Margin for hard negative mining
    """

    def __init__(
        self,
        temperature: float = 0.07,
        use_hard_negatives: bool = True,
        hard_negative_ratio: float = 0.5,
        margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_ratio = hard_negative_ratio
        self.margin = margin

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute contrastive loss.
        
        Args:
            image_embeds: Image embeddings [batch_size, embed_dim]
            text_embeds: Text embeddings [batch_size, embed_dim]
            domain_ids: Domain IDs for each sample [batch_size]
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = image_embeds.size(0)
        device = image_embeds.device
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_embeds, text_embeds.T) / self.temperature
        
        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=device)
        
        # Compute cross-entropy loss
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.T, labels)
        
        # Total contrastive loss
        contrastive_loss = (loss_i2t + loss_t2i) / 2
        
        # Hard negative mining if enabled
        hard_negative_loss = torch.tensor(0.0, device=device)
        if self.use_hard_negatives and batch_size > 1:
            hard_negative_loss = self._compute_hard_negative_loss(
                image_embeds, text_embeds, domain_ids
            )
        
        return {
            "contrastive_loss": contrastive_loss,
            "hard_negative_loss": hard_negative_loss,
            "total_loss": contrastive_loss + hard_negative_loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
        }

    def _compute_hard_negative_loss(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute hard negative loss for better domain separation."""
        batch_size = image_embeds.size(0)
        device = image_embeds.device
        
        # Compute pairwise similarities
        similarities = torch.matmul(image_embeds, text_embeds.T)
        
        # Create mask for hard negatives (non-diagonal elements)
        mask = ~torch.eye(batch_size, device=device, dtype=torch.bool)
        
        # Get hard negatives (highest similarity non-positive pairs)
        hard_negatives = similarities[mask]
        num_hard_negatives = int(len(hard_negatives) * self.hard_negative_ratio)
        
        if num_hard_negatives > 0:
            # Select top hard negatives
            _, top_indices = torch.topk(hard_negatives, num_hard_negatives)
            hard_negative_loss = F.relu(hard_negatives[top_indices] - self.margin).mean()
        else:
            hard_negative_loss = torch.tensor(0.0, device=device)
        
        return hard_negative_loss


class DomainAlignmentLoss(nn.Module):
    """Domain alignment loss for domain adaptation.
    
    Encourages similar representations for samples from the same domain
    and different representations for samples from different domains.
    
    Args:
        alignment_weight: Weight for alignment loss
        separation_weight: Weight for domain separation loss
    """

    def __init__(
        self,
        alignment_weight: float = 1.0,
        separation_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.alignment_weight = alignment_weight
        self.separation_weight = separation_weight

    def forward(
        self,
        embeddings: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute domain alignment loss.
        
        Args:
            embeddings: Feature embeddings [batch_size, embed_dim]
            domain_ids: Domain IDs for each sample [batch_size]
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # Compute pairwise similarities
        similarities = torch.matmul(embeddings, embeddings.T)
        
        # Create domain masks
        domain_mask = domain_ids.unsqueeze(0) == domain_ids.unsqueeze(1)
        
        # Intra-domain alignment loss (encourage high similarity within domains)
        intra_domain_similarities = similarities[domain_mask]
        alignment_loss = -intra_domain_similarities.mean()
        
        # Inter-domain separation loss (encourage low similarity across domains)
        inter_domain_mask = ~domain_mask
        inter_domain_similarities = similarities[inter_domain_mask]
        separation_loss = inter_domain_similarities.mean()
        
        # Total domain alignment loss
        total_loss = (
            self.alignment_weight * alignment_loss +
            self.separation_weight * separation_loss
        )
        
        return {
            "domain_alignment_loss": total_loss,
            "alignment_loss": alignment_loss,
            "separation_loss": separation_loss,
        }


class UniformityLoss(nn.Module):
    """Uniformity loss for representation learning.
    
    Encourages uniform distribution of embeddings in the feature space
    to prevent collapse and improve representation quality.
    
    Args:
        temperature: Temperature parameter for uniformity computation
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute uniformity loss.
        
        Args:
            embeddings: Feature embeddings [batch_size, embed_dim]
            
        Returns:
            Uniformity loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Compute pairwise distances
        pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Compute uniformity loss (encourage uniform distribution)
        uniformity_loss = torch.logsumexp(-pairwise_distances / self.temperature, dim=-1).mean()
        
        return uniformity_loss


class MultiModalDomainAdaptationLoss(nn.Module):
    """Combined loss function for multi-modal domain adaptation.
    
    Combines contrastive loss, domain alignment loss, and uniformity loss
    for comprehensive domain adaptation training.
    
    Args:
        contrastive_weight: Weight for contrastive loss
        domain_adaptation_weight: Weight for domain alignment loss
        uniformity_weight: Weight for uniformity loss
        regularization_weight: Weight for regularization
        contrastive_config: Configuration for contrastive loss
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        domain_adaptation_weight: float = 0.5,
        uniformity_weight: float = 0.1,
        regularization_weight: float = 0.01,
        contrastive_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.domain_adaptation_weight = domain_adaptation_weight
        self.uniformity_weight = uniformity_weight
        self.regularization_weight = regularization_weight
        
        # Initialize loss components
        contrastive_config = contrastive_config or {}
        self.contrastive_loss = ContrastiveLoss(**contrastive_config)
        self.domain_alignment_loss = DomainAlignmentLoss()
        self.uniformity_loss = UniformityLoss()

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.
        
        Args:
            image_embeds: Image embeddings [batch_size, embed_dim]
            text_embeds: Text embeddings [batch_size, embed_dim]
            domain_ids: Domain IDs for each sample [batch_size]
            model: Model for computing regularization loss
            
        Returns:
            Dictionary containing all loss components
        """
        losses = {}
        
        # Contrastive loss
        contrastive_outputs = self.contrastive_loss(image_embeds, text_embeds, domain_ids)
        losses.update(contrastive_outputs)
        
        # Domain alignment loss
        if domain_ids is not None:
            # Combine image and text embeddings for domain alignment
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=0)
            combined_domain_ids = torch.cat([domain_ids, domain_ids], dim=0)
            
            domain_outputs = self.domain_alignment_loss(combined_embeds, combined_domain_ids)
            losses.update(domain_outputs)
        
        # Uniformity loss
        uniformity_loss_img = self.uniformity_loss(image_embeds)
        uniformity_loss_txt = self.uniformity_loss(text_embeds)
        uniformity_loss = (uniformity_loss_img + uniformity_loss_txt) / 2
        losses["uniformity_loss"] = uniformity_loss
        
        # Regularization loss
        regularization_loss = torch.tensor(0.0, device=image_embeds.device)
        if model is not None:
            regularization_loss = self._compute_regularization_loss(model)
        losses["regularization_loss"] = regularization_loss
        
        # Total loss
        total_loss = (
            self.contrastive_weight * losses["contrastive_loss"] +
            self.domain_adaptation_weight * losses.get("domain_alignment_loss", 0.0) +
            self.uniformity_weight * uniformity_loss +
            self.regularization_weight * regularization_loss
        )
        losses["total_loss"] = total_loss
        
        return losses

    def _compute_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute L2 regularization loss for model parameters."""
        l2_reg = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for param in model.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, p=2)
        
        return l2_reg
