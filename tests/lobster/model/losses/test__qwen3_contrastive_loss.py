import pytest
import torch
import torch.nn.functional as F

from lobster.model.losses import Qwen3ContrastiveLoss


class TestQwen3ContrastiveLossBasic:
    """Basic functionality tests."""

    def test_initialization_and_forward(self):
        """Test loss initialization and forward pass returns scalar."""
        torch.manual_seed(42)
        loss_fn = Qwen3ContrastiveLoss(temperature=0.02)

        batch_size = 8
        embed_dim = 128
        num_negatives = 5

        query = torch.randn(batch_size, embed_dim)
        positive = torch.randn(batch_size, embed_dim)
        negative = torch.randn(batch_size, num_negatives, embed_dim)

        loss = loss_fn(query, positive, negative)

        # Loss should be a positive scalar
        assert loss.ndim == 0, "Loss should be scalar"
        assert loss > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_gradient_flow(self):
        """Test that gradients flow through all components."""
        torch.manual_seed(42)
        loss_fn = Qwen3ContrastiveLoss()

        batch_size = 4
        embed_dim = 64
        num_negatives = 3

        query = torch.randn(batch_size, embed_dim, requires_grad=True)
        positive = torch.randn(batch_size, embed_dim, requires_grad=True)
        negative = torch.randn(batch_size, num_negatives, embed_dim, requires_grad=True)

        loss = loss_fn(query, positive, negative)
        loss.backward()

        # Check that gradients exist and are finite
        assert query.grad is not None and torch.isfinite(query.grad).all()
        assert positive.grad is not None and torch.isfinite(positive.grad).all()
        assert negative.grad is not None and torch.isfinite(negative.grad).all()

        # Check that gradients are non-zero
        assert query.grad.abs().sum() > 0
        assert positive.grad.abs().sum() > 0
        assert negative.grad.abs().sum() > 0


class TestQwen3ContrastiveLossProperties:
    """Test key mathematical properties."""

    def test_temperature_effect(self):
        """Test that temperature affects loss magnitude."""
        torch.manual_seed(42)

        batch_size = 8
        embed_dim = 128
        num_negatives = 5

        query = torch.randn(batch_size, embed_dim)
        positive = torch.randn(batch_size, embed_dim)
        negative = torch.randn(batch_size, num_negatives, embed_dim)

        # Use reasonable temperature values (not too extreme)
        loss_low_temp = Qwen3ContrastiveLoss(temperature=0.02)(query, positive, negative)
        loss_high_temp = Qwen3ContrastiveLoss(temperature=0.1)(query, positive, negative)

        # Both should be finite and positive
        assert torch.isfinite(loss_low_temp) and torch.isfinite(loss_high_temp)
        assert loss_low_temp > 0 and loss_high_temp > 0

        # Lower temperature typically gives higher loss values
        assert loss_low_temp > loss_high_temp

    def test_perfect_positive_gives_low_loss(self):
        """Test that identical query and positive give lower loss."""
        torch.manual_seed(42)
        loss_fn = Qwen3ContrastiveLoss()

        batch_size = 4
        embed_dim = 64
        num_negatives = 3

        query = F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1)
        negative = F.normalize(torch.randn(batch_size, num_negatives, embed_dim), p=2, dim=2)

        # Case 1: Positive is identical to query
        positive_identical = query.clone()
        loss_identical = loss_fn(query, positive_identical, negative)

        # Case 2: Positive is random
        positive_random = F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1)
        loss_random = loss_fn(query, positive_random, negative)

        # Identical positive should give lower loss
        assert loss_identical < loss_random

    def test_more_negatives_increases_loss(self):
        """Test that adding more negatives increases loss."""
        torch.manual_seed(42)
        loss_fn = Qwen3ContrastiveLoss()

        batch_size = 4
        embed_dim = 64

        query = torch.randn(batch_size, embed_dim)
        positive = torch.randn(batch_size, embed_dim)

        # Few negatives
        negative_few = torch.randn(batch_size, 2, embed_dim)
        loss_few = loss_fn(query, positive, negative_few)

        # Many negatives
        negative_many = torch.randn(batch_size, 10, embed_dim)
        loss_many = loss_fn(query, positive, negative_many)

        # More negatives should increase loss
        assert loss_many > loss_few


class TestQwen3ContrastiveLossDataloaderCompatibility:
    """Test compatibility with GredAffinityTripletsDataModule structure."""

    def test_with_triplet_batch_structure(self):
        """Test with structure matching our triplets dataloader."""
        torch.manual_seed(42)
        loss_fn = Qwen3ContrastiveLoss()

        # Simulate batch from GredAffinityTripletsDataModule
        batch_size = 4
        embed_dim = 768  # UME embedding dimension
        num_negatives = 5  # As in our dataset

        query_embeds = torch.randn(batch_size, embed_dim)
        positive_embeds = torch.randn(batch_size, embed_dim)
        negative_embeds = torch.randn(batch_size, num_negatives, embed_dim)

        loss = loss_fn(query_embeds, positive_embeds, negative_embeds)

        assert torch.isfinite(loss) and loss > 0

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        torch.manual_seed(42)
        loss_fn = Qwen3ContrastiveLoss()

        embed_dim = 128
        num_negatives = 5

        for batch_size in [1, 4, 16, 32]:
            query = torch.randn(batch_size, embed_dim)
            positive = torch.randn(batch_size, embed_dim)
            negative = torch.randn(batch_size, num_negatives, embed_dim)

            loss = loss_fn(query, positive, negative)

            assert torch.isfinite(loss) and loss > 0

    def test_with_normalized_embeddings(self):
        """Test with pre-normalized embeddings (common in embedding models)."""
        torch.manual_seed(42)
        loss_fn = Qwen3ContrastiveLoss()

        batch_size = 8
        embed_dim = 128
        num_negatives = 5

        # Pre-normalize embeddings
        query = F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1)
        positive = F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1)
        negative = F.normalize(torch.randn(batch_size, num_negatives, embed_dim), p=2, dim=2)

        loss = loss_fn(query, positive, negative)

        assert torch.isfinite(loss) and loss > 0

    def test_deterministic_with_seed(self):
        """Test that loss is deterministic with fixed seed."""
        batch_size = 4
        embed_dim = 64
        num_negatives = 3

        # First run
        torch.manual_seed(42)
        query1 = torch.randn(batch_size, embed_dim)
        positive1 = torch.randn(batch_size, embed_dim)
        negative1 = torch.randn(batch_size, num_negatives, embed_dim)
        loss1 = Qwen3ContrastiveLoss()(query1, positive1, negative1)

        # Second run with same seed
        torch.manual_seed(42)
        query2 = torch.randn(batch_size, embed_dim)
        positive2 = torch.randn(batch_size, embed_dim)
        negative2 = torch.randn(batch_size, num_negatives, embed_dim)
        loss2 = Qwen3ContrastiveLoss()(query2, positive2, negative2)

        # Should be identical
        assert torch.allclose(loss1, loss2)


@pytest.mark.parametrize("batch_size,num_negatives", [
    (2, 1),
    (4, 3),
    (8, 5),
    (16, 10),
])
def test_various_configurations(batch_size, num_negatives):
    """Parametrized test for common configurations."""
    torch.manual_seed(42)
    loss_fn = Qwen3ContrastiveLoss()

    embed_dim = 128

    query = torch.randn(batch_size, embed_dim)
    positive = torch.randn(batch_size, embed_dim)
    negative = torch.randn(batch_size, num_negatives, embed_dim)

    loss = loss_fn(query, positive, negative)

    assert torch.isfinite(loss) and loss > 0
