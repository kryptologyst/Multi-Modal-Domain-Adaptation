"""Unit tests for multi-modal domain adaptation."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.clip_domain_adaptation import CLIPDomainAdaptationModel, AdapterLayer, DomainEmbedding
from src.losses.domain_adaptation_losses import ContrastiveLoss, DomainAlignmentLoss, UniformityLoss
from src.data.medical_dataset import MedicalDomainDataset
from src.eval.domain_adaptation_evaluator import DomainAdaptationEvaluator
from src.utils import set_seed, get_device, count_parameters


class TestAdapterLayer:
    """Test adapter layer functionality."""
    
    def test_adapter_initialization(self):
        """Test adapter layer initialization."""
        adapter = AdapterLayer(hidden_size=512, adapter_size=64)
        
        assert adapter.hidden_size == 512
        assert adapter.adapter_size == 64
        assert adapter.down_proj.in_features == 512
        assert adapter.down_proj.out_features == 64
        assert adapter.up_proj.in_features == 64
        assert adapter.up_proj.out_features == 512
    
    def test_adapter_forward(self):
        """Test adapter forward pass."""
        adapter = AdapterLayer(hidden_size=512, adapter_size=64)
        x = torch.randn(2, 10, 512)
        
        output = adapter(x)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, x)  # Should modify input


class TestDomainEmbedding:
    """Test domain embedding functionality."""
    
    def test_domain_embedding_initialization(self):
        """Test domain embedding initialization."""
        domain_emb = DomainEmbedding(num_domains=3, embedding_size=128)
        
        assert domain_emb.num_domains == 3
        assert domain_emb.embedding_size == 128
        assert domain_emb.domain_embeddings.num_embeddings == 3
        assert domain_emb.domain_embeddings.embedding_dim == 128
    
    def test_domain_embedding_forward(self):
        """Test domain embedding forward pass."""
        domain_emb = DomainEmbedding(num_domains=3, embedding_size=128)
        domain_ids = torch.tensor([0, 1, 2])
        
        embeddings = domain_emb(domain_ids)
        
        assert embeddings.shape == (3, 128)
        assert embeddings.requires_grad


class TestCLIPDomainAdaptationModel:
    """Test CLIP domain adaptation model."""
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        config = {
            "adaptation_method": "adapter",
            "freeze_base_model": False,
            "adapter_config": {
                "hidden_size": 512,
                "adapter_size": 64,
                "adapter_layers": [6, 8, 10],
                "dropout": 0.1
            },
            "domain_embedding_size": 128
        }
        return CLIPDomainAdaptationModel(
            base_model_name="openai/clip-vit-base-patch32",
            config=config
        )
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.base_model_name == "openai/clip-vit-base-patch32"
        assert model.adaptation_method == "adapter"
        assert hasattr(model, "domain_embedding")
        assert hasattr(model, "image_domain_proj")
        assert hasattr(model, "text_domain_proj")
    
    def test_model_forward(self, model):
        """Test model forward pass."""
        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, 77))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        attention_mask = torch.ones(batch_size, 77)
        domain_ids = torch.tensor([0, 1])
        
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            domain_ids=domain_ids
        )
        
        assert "image_embeds" in outputs
        assert "text_embeds" in outputs
        assert outputs["image_embeds"].shape == (batch_size, 128)
        assert outputs["text_embeds"].shape == (batch_size, 128)
    
    def test_model_encoding(self, model):
        """Test model encoding methods."""
        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, 77))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        
        image_embeds = model.encode_image(pixel_values)
        text_embeds = model.encode_text(input_ids)
        
        assert image_embeds.shape == (batch_size, 128)
        assert text_embeds.shape == (batch_size, 128)
        
        # Check normalization
        assert torch.allclose(torch.norm(image_embeds, dim=1), torch.ones(batch_size), atol=1e-6)
        assert torch.allclose(torch.norm(text_embeds, dim=1), torch.ones(batch_size), atol=1e-6)


class TestLossFunctions:
    """Test loss functions."""
    
    def test_contrastive_loss(self):
        """Test contrastive loss."""
        loss_fn = ContrastiveLoss(temperature=0.07)
        
        batch_size = 4
        image_embeds = torch.randn(batch_size, 128)
        text_embeds = torch.randn(batch_size, 128)
        domain_ids = torch.tensor([0, 0, 1, 1])
        
        # Normalize embeddings
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)
        
        loss_outputs = loss_fn(image_embeds, text_embeds, domain_ids)
        
        assert "contrastive_loss" in loss_outputs
        assert "hard_negative_loss" in loss_outputs
        assert "total_loss" in loss_outputs
        assert loss_outputs["contrastive_loss"] > 0
    
    def test_domain_alignment_loss(self):
        """Test domain alignment loss."""
        loss_fn = DomainAlignmentLoss()
        
        batch_size = 4
        embeddings = torch.randn(batch_size, 128)
        domain_ids = torch.tensor([0, 0, 1, 1])
        
        loss_outputs = loss_fn(embeddings, domain_ids)
        
        assert "domain_alignment_loss" in loss_outputs
        assert "alignment_loss" in loss_outputs
        assert "separation_loss" in loss_outputs
        assert loss_outputs["domain_alignment_loss"] > 0
    
    def test_uniformity_loss(self):
        """Test uniformity loss."""
        loss_fn = UniformityLoss()
        
        batch_size = 4
        embeddings = torch.randn(batch_size, 128)
        
        loss = loss_fn(embeddings)
        
        assert isinstance(loss, torch.Tensor)
        assert loss > 0


class TestMedicalDataset:
    """Test medical dataset functionality."""
    
    def test_dataset_initialization(self, tmp_path):
        """Test dataset initialization."""
        dataset = MedicalDomainDataset(
            data_dir=tmp_path,
            split="train",
            generate_synthetic=True
        )
        
        assert len(dataset) > 0
        assert hasattr(dataset, "samples")
    
    def test_dataset_getitem(self, tmp_path):
        """Test dataset item retrieval."""
        dataset = MedicalDomainDataset(
            data_dir=tmp_path,
            split="train",
            generate_synthetic=True
        )
        
        sample = dataset[0]
        
        assert "image" in sample
        assert "text" in sample
        assert "domain" in sample
        assert "sample_id" in sample


class TestEvaluator:
    """Test evaluator functionality."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        config = {
            "metrics": {
                "retrieval": ["recall_at_1", "recall_at_5"],
                "domain_adaptation": ["domain_alignment_score"]
            }
        }
        evaluator = DomainAdaptationEvaluator(config)
        
        assert evaluator.config == config
        assert evaluator.metrics == config["metrics"]
    
    def test_retrieval_metrics(self):
        """Test retrieval metrics computation."""
        evaluator = DomainAdaptationEvaluator()
        
        batch_size = 4
        image_embeds = torch.randn(batch_size, 128)
        text_embeds = torch.randn(batch_size, 128)
        
        # Normalize embeddings
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)
        
        metrics = evaluator._compute_retrieval_metrics(image_embeds, text_embeds)
        
        assert "i2t_recall_at_1" in metrics
        assert "t2i_recall_at_1" in metrics
        assert "avg_recall_at_1" in metrics
        assert 0 <= metrics["avg_recall_at_1"] <= 1


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seeds are set
        assert torch.initial_seed() is not None
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("auto")
        
        assert isinstance(device, torch.device)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        
        counts = count_parameters(model, trainable_only=False)
        
        assert counts["total_parameters"] == 55  # 10*5 + 5
        assert counts["trainable_parameters"] == 55
        assert counts["frozen_parameters"] == 0


# Integration tests
class TestIntegration:
    """Integration tests."""
    
    def test_training_loop(self, tmp_path):
        """Test basic training loop."""
        # Create synthetic dataset
        dataset = MedicalDomainDataset(
            data_dir=tmp_path,
            split="train",
            generate_synthetic=True
        )
        
        # Create model
        config = {
            "adaptation_method": "adapter",
            "freeze_base_model": False,
            "adapter_config": {
                "hidden_size": 512,
                "adapter_size": 64,
                "adapter_layers": [6, 8, 10],
                "dropout": 0.1
            },
            "domain_embedding_size": 128
        }
        model = CLIPDomainAdaptationModel(
            base_model_name="openai/clip-vit-base-patch32",
            config=config
        )
        
        # Test forward pass
        sample = dataset[0]
        if isinstance(sample["image"], torch.Tensor):
            pixel_values = sample["image"].unsqueeze(0)
        else:
            # Handle PIL image
            pixel_values = torch.randn(1, 3, 224, 224)
        
        input_ids = torch.randint(0, 1000, (1, 77))
        attention_mask = torch.ones(1, 77)
        domain_ids = torch.tensor([0])
        
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            domain_ids=domain_ids
        )
        
        assert "image_embeds" in outputs
        assert "text_embeds" in outputs
        assert outputs["image_embeds"].shape[0] == 1
        assert outputs["text_embeds"].shape[0] == 1


if __name__ == "__main__":
    pytest.main([__file__])
