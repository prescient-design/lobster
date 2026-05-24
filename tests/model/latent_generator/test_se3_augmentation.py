"""Tests for SE3 augmentation utilities for protein-ligand complexes.

These tests verify that:
1. SE3 transformations preserve internal structure (distances, angles)
2. Protein and ligand receive the SAME transformation
3. Kabsch alignment can recover the original coordinates after transformation
"""

import pytest
import torch

from lobster.model.latent_generator.utils import (
    kabsch_torch_batched,
    apply_se3_augmentation_batched,
    apply_se3_augmentation_protein_ligand,
    SE3AugmentedComplex,
)


class TestSE3AugmentationProteinLigand:
    """Test SE3 augmentation for protein-ligand complexes."""

    @pytest.fixture
    def protein_coords(self) -> torch.Tensor:
        """Create mock protein coordinates [B, L, n_atoms, 3]."""
        B, L, n_atoms = 2, 50, 4  # batch, residues, backbone atoms
        # Create a simple helix-like structure
        coords = torch.zeros(B, L, n_atoms, 3)
        for b in range(B):
            for i in range(L):
                # Simple helical arrangement
                theta = i * 0.5
                for a in range(n_atoms):
                    coords[b, i, a, 0] = torch.cos(torch.tensor(theta)) * (5 + a * 0.5)
                    coords[b, i, a, 1] = torch.sin(torch.tensor(theta)) * (5 + a * 0.5)
                    coords[b, i, a, 2] = i * 1.5 + a * 0.3
        return coords

    @pytest.fixture
    def protein_mask(self) -> torch.Tensor:
        """Create protein mask [B, L]."""
        return torch.ones(2, 50, dtype=torch.bool)

    @pytest.fixture
    def ligand_coords(self) -> torch.Tensor:
        """Create mock ligand coordinates [B, N_ligand, 3]."""
        B, N = 2, 20  # batch, ligand atoms
        # Create a simple ring-like ligand
        coords = torch.zeros(B, N, 3)
        for b in range(B):
            for i in range(N):
                theta = i * 2 * 3.14159 / N
                coords[b, i, 0] = torch.cos(torch.tensor(theta)) * 3
                coords[b, i, 1] = torch.sin(torch.tensor(theta)) * 3
                coords[b, i, 2] = (i % 2) * 0.5  # alternating heights
        return coords

    @pytest.fixture
    def ligand_mask(self) -> torch.Tensor:
        """Create ligand mask [B, N_ligand]."""
        return torch.ones(2, 20, dtype=torch.bool)

    def test_protein_only_augmentation(self, protein_coords, protein_mask):
        """Test SE3 augmentation with protein only."""
        result = apply_se3_augmentation_protein_ligand(
            protein_coords=protein_coords,
            protein_mask=protein_mask,
            random_se3=True,
            translation_scale=10.0,
        )

        assert isinstance(result, SE3AugmentedComplex)
        assert result.protein_coords is not None
        assert result.protein_mask is not None
        assert result.ligand_coords is None
        assert result.ligand_mask is None
        assert result.protein_coords.shape == protein_coords.shape

    def test_ligand_only_augmentation(self, ligand_coords, ligand_mask):
        """Test SE3 augmentation with ligand only."""
        result = apply_se3_augmentation_protein_ligand(
            ligand_coords=ligand_coords,
            ligand_mask=ligand_mask,
            random_se3=True,
            translation_scale=10.0,
        )

        assert isinstance(result, SE3AugmentedComplex)
        assert result.protein_coords is None
        assert result.protein_mask is None
        assert result.ligand_coords is not None
        assert result.ligand_mask is not None
        assert result.ligand_coords.shape == ligand_coords.shape

    def test_protein_ligand_joint_augmentation(
        self, protein_coords, protein_mask, ligand_coords, ligand_mask
    ):
        """Test SE3 augmentation with both protein and ligand."""
        result = apply_se3_augmentation_protein_ligand(
            protein_coords=protein_coords,
            protein_mask=protein_mask,
            ligand_coords=ligand_coords,
            ligand_mask=ligand_mask,
            random_se3=True,
            translation_scale=10.0,
        )

        assert isinstance(result, SE3AugmentedComplex)
        assert result.protein_coords is not None
        assert result.ligand_coords is not None
        assert result.protein_coords.shape == protein_coords.shape
        assert result.ligand_coords.shape == ligand_coords.shape

    def test_se3_preserves_internal_distances(
        self, protein_coords, protein_mask, ligand_coords, ligand_mask
    ):
        """Test that SE3 transformation preserves pairwise distances within protein and ligand."""
        result = apply_se3_augmentation_protein_ligand(
            protein_coords=protein_coords,
            protein_mask=protein_mask,
            ligand_coords=ligand_coords,
            ligand_mask=ligand_mask,
            random_se3=True,
            translation_scale=10.0,
            backbone_noise=0.0,  # No noise to test pure SE3
        )

        # Flatten protein coords for distance computation
        original_protein_flat = protein_coords.reshape(protein_coords.shape[0], -1, 3)
        transformed_protein_flat = result.protein_coords.reshape(
            result.protein_coords.shape[0], -1, 3
        )

        # Compute pairwise distances for first few atoms
        n_check = min(10, original_protein_flat.shape[1])
        for b in range(protein_coords.shape[0]):
            orig_dists = torch.cdist(
                original_protein_flat[b, :n_check], original_protein_flat[b, :n_check]
            )
            trans_dists = torch.cdist(
                transformed_protein_flat[b, :n_check], transformed_protein_flat[b, :n_check]
            )
            assert torch.allclose(
                orig_dists, trans_dists, atol=1e-5
            ), "SE3 should preserve pairwise distances"

        # Same for ligand
        for b in range(ligand_coords.shape[0]):
            orig_lig_dists = torch.cdist(ligand_coords[b], ligand_coords[b])
            trans_lig_dists = torch.cdist(result.ligand_coords[b], result.ligand_coords[b])
            assert torch.allclose(
                orig_lig_dists, trans_lig_dists, atol=1e-5
            ), "SE3 should preserve ligand pairwise distances"

    def test_se3_preserves_protein_ligand_relative_distances(
        self, protein_coords, protein_mask, ligand_coords, ligand_mask
    ):
        """Test that SE3 transformation preserves distances between protein and ligand."""
        result = apply_se3_augmentation_protein_ligand(
            protein_coords=protein_coords,
            protein_mask=protein_mask,
            ligand_coords=ligand_coords,
            ligand_mask=ligand_mask,
            random_se3=True,
            translation_scale=10.0,
            backbone_noise=0.0,
        )

        # Use CA atoms for protein (index 1)
        original_ca = protein_coords[:, :, 1, :]  # [B, L, 3]
        transformed_ca = result.protein_coords[:, :, 1, :]  # [B, L, 3]

        for b in range(protein_coords.shape[0]):
            # Compute protein-ligand distances
            orig_pl_dists = torch.cdist(original_ca[b], ligand_coords[b])
            trans_pl_dists = torch.cdist(transformed_ca[b], result.ligand_coords[b])
            assert torch.allclose(
                orig_pl_dists, trans_pl_dists, atol=1e-4
            ), "SE3 should preserve protein-ligand distances"

    def test_kabsch_alignment_recovers_original(
        self, protein_coords, protein_mask, ligand_coords, ligand_mask
    ):
        """Test that Kabsch alignment can recover original structure from transformed."""
        result = apply_se3_augmentation_protein_ligand(
            protein_coords=protein_coords,
            protein_mask=protein_mask,
            ligand_coords=ligand_coords,
            ligand_mask=ligand_mask,
            random_se3=True,
            translation_scale=10.0,
            backbone_noise=0.0,
        )

        # Flatten for Kabsch alignment
        original_flat = protein_coords.reshape(protein_coords.shape[0], -1, 3)
        transformed_flat = result.protein_coords.reshape(result.protein_coords.shape[0], -1, 3)
        mask_flat = protein_mask.unsqueeze(-1).expand(-1, -1, 4).reshape(
            protein_mask.shape[0], -1
        )

        # Align transformed back to original using Kabsch
        aligned = kabsch_torch_batched(transformed_flat, original_flat, mask_flat)

        # Check RMSD is near zero
        diff = (aligned - original_flat) * mask_flat.unsqueeze(-1)
        rmsd = torch.sqrt((diff**2).sum(dim=(1, 2)) / mask_flat.sum(dim=1))
        assert (rmsd < 1e-4).all(), f"Kabsch alignment should recover original, got RMSD: {rmsd}"

    def test_rotation_only(self, protein_coords, protein_mask):
        """Test rotation-only transformation."""
        result = apply_se3_augmentation_protein_ligand(
            protein_coords=protein_coords,
            protein_mask=protein_mask,
            random_se3=True,
            only_rot=True,
            translation_scale=100.0,  # Should be ignored
            backbone_noise=0.0,
        )

        # Both should be centered at origin after rotation-only
        original_center = protein_coords.reshape(protein_coords.shape[0], -1, 3).mean(dim=1)
        transformed_center = result.protein_coords.reshape(
            result.protein_coords.shape[0], -1, 3
        ).mean(dim=1)

        # Centers should both be near zero (centered first, then rotated)
        # Note: The implementation centers, rotates, then adds translation (0 in this case)
        # So both should have mean near zero
        assert torch.allclose(
            transformed_center, torch.zeros_like(transformed_center), atol=1e-4
        ), "Rotation-only should keep center near origin"

    def test_translation_only(self, protein_coords, protein_mask):
        """Test translation-only transformation."""
        torch.manual_seed(42)  # For reproducibility
        result = apply_se3_augmentation_protein_ligand(
            protein_coords=protein_coords,
            protein_mask=protein_mask,
            random_se3=True,
            only_trans=True,
            translation_scale=10.0,
            backbone_noise=0.0,
        )

        # Internal distances should be preserved
        original_flat = protein_coords.reshape(protein_coords.shape[0], -1, 3)
        transformed_flat = result.protein_coords.reshape(result.protein_coords.shape[0], -1, 3)

        for b in range(protein_coords.shape[0]):
            orig_dists = torch.cdist(original_flat[b, :10], original_flat[b, :10])
            trans_dists = torch.cdist(transformed_flat[b, :10], transformed_flat[b, :10])
            assert torch.allclose(
                orig_dists, trans_dists, atol=1e-5
            ), "Translation should preserve distances"

    def test_no_se3(self, protein_coords, protein_mask):
        """Test with SE3 disabled."""
        result = apply_se3_augmentation_protein_ligand(
            protein_coords=protein_coords,
            protein_mask=protein_mask,
            random_se3=False,
        )

        assert torch.allclose(
            result.protein_coords, protein_coords, atol=1e-6
        ), "No SE3 should return identical coordinates"

    def test_backbone_noise(self, protein_coords, protein_mask):
        """Test backbone noise addition."""
        torch.manual_seed(42)
        result = apply_se3_augmentation_protein_ligand(
            protein_coords=protein_coords,
            protein_mask=protein_mask,
            random_se3=False,  # Disable SE3 to isolate noise effect
            backbone_noise=0.1,
        )

        # Should be different due to noise
        assert not torch.allclose(
            result.protein_coords, protein_coords, atol=1e-6
        ), "Backbone noise should modify coordinates"

        # But difference should be bounded by noise scale
        diff = (result.protein_coords - protein_coords).abs()
        # 3-sigma bound for Gaussian noise
        assert (diff < 0.5).all(), "Noise should be bounded"

    def test_no_inputs_raises_error(self):
        """Test that providing no inputs raises an error."""
        with pytest.raises(ValueError, match="At least one"):
            apply_se3_augmentation_protein_ligand()

    def test_partial_mask(self, protein_coords, ligand_coords):
        """Test with partial masks (some positions masked out)."""
        protein_mask = torch.ones(2, 50, dtype=torch.bool)
        protein_mask[0, 25:] = False  # Mask out second half for first batch
        protein_mask[1, :10] = False  # Mask out first 10 for second batch

        ligand_mask = torch.ones(2, 20, dtype=torch.bool)
        ligand_mask[0, 10:] = False  # Mask out second half

        result = apply_se3_augmentation_protein_ligand(
            protein_coords=protein_coords,
            protein_mask=protein_mask,
            ligand_coords=ligand_coords,
            ligand_mask=ligand_mask,
            random_se3=True,
            translation_scale=10.0,
        )

        assert result.protein_coords.shape == protein_coords.shape
        assert result.ligand_coords.shape == ligand_coords.shape


class TestSE3AugmentationBatched:
    """Test the simplified batched SE3 augmentation interface."""

    @pytest.fixture
    def flat_coords(self) -> torch.Tensor:
        """Create flat coordinates [B, N, 3]."""
        B, N = 3, 100
        coords = torch.randn(B, N, 3) * 10
        return coords

    @pytest.fixture
    def structured_coords(self) -> torch.Tensor:
        """Create structured coordinates [B, L, n_atoms, 3]."""
        B, L, n_atoms = 3, 50, 4
        coords = torch.randn(B, L, n_atoms, 3) * 10
        return coords

    def test_flat_coords(self, flat_coords):
        """Test with flat coordinate input."""
        mask = torch.ones(flat_coords.shape[0], flat_coords.shape[1], dtype=torch.bool)
        result = apply_se3_augmentation_batched(
            coords=flat_coords,
            mask=mask,
            random_se3=True,
            translation_scale=5.0,
        )

        assert result.shape == flat_coords.shape

        # Verify distances preserved
        for b in range(flat_coords.shape[0]):
            orig_dists = torch.cdist(flat_coords[b, :10], flat_coords[b, :10])
            trans_dists = torch.cdist(result[b, :10], result[b, :10])
            assert torch.allclose(orig_dists, trans_dists, atol=1e-5)

    def test_structured_coords(self, structured_coords):
        """Test with structured coordinate input [B, L, n_atoms, 3]."""
        mask = torch.ones(
            structured_coords.shape[0], structured_coords.shape[1], dtype=torch.bool
        )
        result = apply_se3_augmentation_batched(
            coords=structured_coords,
            mask=mask,
            random_se3=True,
            translation_scale=5.0,
        )

        assert result.shape == structured_coords.shape

    def test_kabsch_alignment_flat(self, flat_coords):
        """Test Kabsch alignment on flat augmented coordinates."""
        mask = torch.ones(flat_coords.shape[0], flat_coords.shape[1], dtype=torch.bool)

        transformed = apply_se3_augmentation_batched(
            coords=flat_coords,
            mask=mask,
            random_se3=True,
            translation_scale=20.0,
            backbone_noise=0.0,
        )

        # Align back using Kabsch
        aligned = kabsch_torch_batched(transformed, flat_coords, mask)

        # Check RMSD
        diff = (aligned - flat_coords) * mask.unsqueeze(-1)
        rmsd = torch.sqrt((diff**2).sum(dim=(1, 2)) / mask.sum(dim=1))
        assert (rmsd < 1e-4).all(), f"Kabsch should recover original, got RMSD: {rmsd}"


class TestSE3EquivarianceProperties:
    """Test SE3 equivariance properties of the augmentation."""

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with fixed seed."""
        coords = torch.randn(2, 50, 4, 3) * 10
        mask = torch.ones(2, 50, dtype=torch.bool)

        torch.manual_seed(12345)
        result1 = apply_se3_augmentation_protein_ligand(
            protein_coords=coords.clone(),
            protein_mask=mask,
            random_se3=True,
        )

        torch.manual_seed(12345)
        result2 = apply_se3_augmentation_protein_ligand(
            protein_coords=coords.clone(),
            protein_mask=mask,
            random_se3=True,
        )

        assert torch.allclose(
            result1.protein_coords, result2.protein_coords, atol=1e-6
        ), "Same seed should give same result"

    def test_different_seeds_give_different_results(self):
        """Test that different seeds give different transformations."""
        coords = torch.randn(2, 50, 4, 3) * 10
        mask = torch.ones(2, 50, dtype=torch.bool)

        torch.manual_seed(111)
        result1 = apply_se3_augmentation_protein_ligand(
            protein_coords=coords.clone(),
            protein_mask=mask,
            random_se3=True,
        )

        torch.manual_seed(222)
        result2 = apply_se3_augmentation_protein_ligand(
            protein_coords=coords.clone(),
            protein_mask=mask,
            random_se3=True,
        )

        assert not torch.allclose(
            result1.protein_coords, result2.protein_coords, atol=1e-3
        ), "Different seeds should give different results"

    def test_composition_of_se3_is_se3(self):
        """Test that applying SE3 twice still preserves distances (composition of SE3 is SE3)."""
        coords = torch.randn(2, 30, 3).double() * 10  # Use double for better precision
        mask = torch.ones(2, 30, dtype=torch.bool)

        # Apply twice
        result1 = apply_se3_augmentation_batched(
            coords=coords.clone(),
            mask=mask,
            random_se3=True,
            translation_scale=5.0,
            backbone_noise=0.0,
        )

        result2 = apply_se3_augmentation_batched(
            coords=result1.clone(),
            mask=mask,
            random_se3=True,
            translation_scale=5.0,
            backbone_noise=0.0,
        )

        # Distances should still match original (use relative tolerance for numerical stability)
        for b in range(coords.shape[0]):
            orig_dists = torch.cdist(coords[b], coords[b])
            final_dists = torch.cdist(result2[b], result2[b])
            assert torch.allclose(
                orig_dists, final_dists, atol=1e-4, rtol=1e-4
            ), "Composition of SE3 should preserve distances"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

