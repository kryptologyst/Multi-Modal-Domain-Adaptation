"""Evaluation metrics and utilities for domain adaptation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class DomainAdaptationEvaluator:
    """Evaluator for multi-modal domain adaptation models.
    
    Provides comprehensive evaluation metrics including retrieval performance,
    domain alignment, and transfer learning capabilities.
    
    Args:
        config: Evaluation configuration dictionary
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.metrics = self.config.get("metrics", {})
        
    def evaluate(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        compute_embeddings: bool = True,
    ) -> Dict[str, float]:
        """Evaluate model on given data loader.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            device: Device to run evaluation on
            compute_embeddings: Whether to compute and store embeddings
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_metrics = {}
        
        # Storage for embeddings and metadata
        image_embeddings = []
        text_embeddings = []
        domain_ids = []
        sample_ids = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    input_ids=batch["text"],
                    pixel_values=batch["image"],
                    attention_mask=batch.get("attention_mask"),
                    domain_ids=batch.get("domain"),
                    return_embeddings=True
                )
                
                # Store embeddings if requested
                if compute_embeddings:
                    image_embeddings.append(outputs["image_embeds"].cpu())
                    text_embeddings.append(outputs["text_embeds"].cpu())
                    domain_ids.append(batch["domain"].cpu())
                    sample_ids.extend(batch["sample_id"])
        
        # Concatenate all embeddings
        if compute_embeddings:
            image_embeddings = torch.cat(image_embeddings, dim=0)
            text_embeddings = torch.cat(text_embeddings, dim=0)
            domain_ids = torch.cat(domain_ids, dim=0)
            
            # Compute retrieval metrics
            retrieval_metrics = self._compute_retrieval_metrics(
                image_embeddings, text_embeddings
            )
            all_metrics.update(retrieval_metrics)
            
            # Compute domain adaptation metrics
            domain_metrics = self._compute_domain_metrics(
                image_embeddings, text_embeddings, domain_ids
            )
            all_metrics.update(domain_metrics)
            
            # Compute contrastive metrics
            contrastive_metrics = self._compute_contrastive_metrics(
                image_embeddings, text_embeds
            )
            all_metrics.update(contrastive_metrics)
        
        return all_metrics

    def _compute_retrieval_metrics(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute retrieval performance metrics."""
        batch_size = image_embeds.size(0)
        device = image_embeds.device
        
        # Compute similarity matrix
        similarities = torch.matmul(image_embeds, text_embeds.T)
        
        # Image-to-text retrieval
        i2t_ranks = self._compute_ranks(similarities)
        i2t_recall_at_1 = self._compute_recall_at_k(i2t_ranks, k=1)
        i2t_recall_at_5 = self._compute_recall_at_k(i2t_ranks, k=5)
        i2t_recall_at_10 = self._compute_recall_at_k(i2t_ranks, k=10)
        i2t_median_rank = torch.median(i2t_ranks.float()).item()
        i2t_mean_rank = i2t_ranks.float().mean().item()
        
        # Text-to-image retrieval
        t2i_ranks = self._compute_ranks(similarities.T)
        t2i_recall_at_1 = self._compute_recall_at_k(t2i_ranks, k=1)
        t2i_recall_at_5 = self._compute_recall_at_k(t2i_ranks, k=5)
        t2i_recall_at_10 = self._compute_recall_at_k(t2i_ranks, k=10)
        t2i_median_rank = torch.median(t2i_ranks.float()).item()
        t2i_mean_rank = t2i_ranks.float().mean().item()
        
        return {
            "i2t_recall_at_1": i2t_recall_at_1,
            "i2t_recall_at_5": i2t_recall_at_5,
            "i2t_recall_at_10": i2t_recall_at_10,
            "i2t_median_rank": i2t_median_rank,
            "i2t_mean_rank": i2t_mean_rank,
            "t2i_recall_at_1": t2i_recall_at_1,
            "t2i_recall_at_5": t2i_recall_at_5,
            "t2i_recall_at_10": t2i_recall_at_10,
            "t2i_median_rank": t2i_median_rank,
            "t2i_mean_rank": t2i_mean_rank,
            "avg_recall_at_1": (i2t_recall_at_1 + t2i_recall_at_1) / 2,
            "avg_recall_at_5": (i2t_recall_at_5 + t2i_recall_at_5) / 2,
            "avg_recall_at_10": (i2t_recall_at_10 + t2i_recall_at_10) / 2,
        }

    def _compute_ranks(self, similarities: torch.Tensor) -> torch.Tensor:
        """Compute ranks for each query."""
        batch_size = similarities.size(0)
        device = similarities.device
        
        # Get ranks (1-indexed)
        _, indices = torch.sort(similarities, dim=1, descending=True)
        ranks = torch.zeros(batch_size, device=device, dtype=torch.long)
        
        for i in range(batch_size):
            rank = torch.where(indices[i] == i)[0]
            if len(rank) > 0:
                ranks[i] = rank[0] + 1
            else:
                ranks[i] = batch_size + 1  # Not found
        
        return ranks

    def _compute_recall_at_k(self, ranks: torch.Tensor, k: int) -> float:
        """Compute recall at k."""
        return (ranks <= k).float().mean().item()

    def _compute_domain_metrics(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute domain adaptation metrics."""
        # Combine embeddings
        combined_embeds = torch.cat([image_embeds, text_embeds], dim=0)
        combined_domain_ids = torch.cat([domain_ids, domain_ids], dim=0)
        
        # Compute domain alignment score (intra-domain similarity)
        domain_alignment_score = self._compute_domain_alignment_score(
            combined_embeds, combined_domain_ids
        )
        
        # Compute domain separation score (inter-domain dissimilarity)
        domain_separation_score = self._compute_domain_separation_score(
            combined_embeds, combined_domain_ids
        )
        
        # Compute transfer accuracy (domain classification)
        transfer_accuracy = self._compute_transfer_accuracy(
            combined_embeds, combined_domain_ids
        )
        
        return {
            "domain_alignment_score": domain_alignment_score,
            "domain_separation_score": domain_separation_score,
            "transfer_accuracy": transfer_accuracy,
        }

    def _compute_domain_alignment_score(
        self,
        embeddings: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> float:
        """Compute domain alignment score."""
        similarities = torch.matmul(embeddings, embeddings.T)
        domain_mask = domain_ids.unsqueeze(0) == domain_ids.unsqueeze(1)
        
        # Exclude diagonal elements
        mask = domain_mask & ~torch.eye(domain_mask.size(0), device=domain_mask.device, dtype=torch.bool)
        intra_domain_similarities = similarities[mask]
        
        if len(intra_domain_similarities) > 0:
            return intra_domain_similarities.mean().item()
        else:
            return 0.0

    def _compute_domain_separation_score(
        self,
        embeddings: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> float:
        """Compute domain separation score."""
        similarities = torch.matmul(embeddings, embeddings.T)
        domain_mask = domain_ids.unsqueeze(0) == domain_ids.unsqueeze(1)
        
        # Inter-domain similarities
        inter_domain_mask = ~domain_mask
        inter_domain_similarities = similarities[inter_domain_mask]
        
        if len(inter_domain_similarities) > 0:
            return -inter_domain_similarities.mean().item()  # Negative because we want separation
        else:
            return 0.0

    def _compute_transfer_accuracy(
        self,
        embeddings: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> float:
        """Compute domain classification accuracy."""
        # Simple nearest neighbor classification
        similarities = torch.matmul(embeddings, embeddings.T)
        
        # For each sample, find most similar samples
        _, indices = torch.topk(similarities, k=min(5, similarities.size(0)), dim=1)
        
        # Predict domain based on majority vote of neighbors
        predictions = []
        for i in range(embeddings.size(0)):
            neighbor_domains = domain_ids[indices[i]]
            # Exclude self
            neighbor_domains = neighbor_domains[neighbor_domains != domain_ids[i]]
            if len(neighbor_domains) > 0:
                pred = torch.mode(neighbor_domains)[0]
                predictions.append(pred.item())
            else:
                predictions.append(domain_ids[i].item())
        
        predictions = torch.tensor(predictions, device=domain_ids.device)
        accuracy = (predictions == domain_ids).float().mean().item()
        
        return accuracy

    def _compute_contrastive_metrics(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute contrastive learning metrics."""
        # Alignment loss (similarity between positive pairs)
        batch_size = image_embeds.size(0)
        positive_similarities = torch.sum(image_embeds * text_embeds, dim=1)
        alignment_loss = -positive_similarities.mean().item()
        
        # Uniformity loss (uniform distribution of embeddings)
        all_embeds = torch.cat([image_embeds, text_embeds], dim=0)
        pairwise_distances = torch.cdist(all_embeds, all_embeds, p=2)
        uniformity_loss = torch.logsumexp(-pairwise_distances / 0.1, dim=-1).mean().item()
        
        return {
            "alignment_loss": alignment_loss,
            "uniformity_loss": uniformity_loss,
        }

    def visualize_embeddings(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        domain_ids: torch.Tensor,
        save_path: Optional[str] = None,
    ) -> None:
        """Visualize embeddings using t-SNE."""
        # Combine embeddings
        combined_embeds = torch.cat([image_embeds, text_embeds], dim=0)
        combined_domain_ids = torch.cat([domain_ids, domain_ids], dim=0)
        
        # Convert to numpy
        embeddings_np = combined_embeds.cpu().numpy()
        domain_ids_np = combined_domain_ids.cpu().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings_np)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot by domain
        domains = np.unique(domain_ids_np)
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, domain in enumerate(domains):
            mask = domain_ids_np == domain
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=colors[i % len(colors)],
                label=f'Domain {domain}',
                alpha=0.7,
                s=50
            )
        
        plt.title('t-SNE Visualization of Domain-Adapted Embeddings')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def create_leaderboard(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> str:
        """Create a leaderboard from evaluation results."""
        # Create leaderboard table
        leaderboard = []
        
        for model_name, metrics in results.items():
            row = {
                'Model': model_name,
                'Avg Recall@1': f"{metrics.get('avg_recall_at_1', 0):.3f}",
                'Avg Recall@5': f"{metrics.get('avg_recall_at_5', 0):.3f}",
                'Avg Recall@10': f"{metrics.get('avg_recall_at_10', 0):.3f}",
                'Domain Alignment': f"{metrics.get('domain_alignment_score', 0):.3f}",
                'Domain Separation': f"{metrics.get('domain_separation_score', 0):.3f}",
                'Transfer Accuracy': f"{metrics.get('transfer_accuracy', 0):.3f}",
            }
            leaderboard.append(row)
        
        # Sort by average recall@1
        leaderboard.sort(key=lambda x: float(x['Avg Recall@1']), reverse=True)
        
        # Create markdown table
        if leaderboard:
            headers = list(leaderboard[0].keys())
            table_lines = ['| ' + ' | '.join(headers) + ' |']
            table_lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
            
            for row in leaderboard:
                table_lines.append('| ' + ' | '.join(str(row[h]) for h in headers) + ' |')
            
            table_md = '\n'.join(table_lines)
        else:
            table_md = "No results available."
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("# Domain Adaptation Leaderboard\n\n")
                f.write(table_md)
        
        return table_md
