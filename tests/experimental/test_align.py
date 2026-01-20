"""Tests for spatial alignment functions."""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if torch not installed
torch = pytest.importorskip("torch")

import anndata as ad

import squidpy as sq


def _create_test_adata(n_cells: int = 500, seed: int = 42) -> ad.AnnData:
    """Create a simple test AnnData with spatial coordinates."""
    rng = np.random.default_rng(seed)

    # Create circular pattern
    theta = rng.uniform(0, 2 * np.pi, n_cells)
    r = rng.uniform(0, 100, n_cells)
    x = r * np.cos(theta) + 500
    y = r * np.sin(theta) + 500

    adata = ad.AnnData(X=rng.random((n_cells, 10)), obsm={"spatial": np.column_stack([x, y])})
    return adata


def _create_rotated_adata(adata: ad.AnnData, rotation_deg: float = 30, translation: tuple = (20, 20)) -> ad.AnnData:
    """Create a rotated and translated copy of AnnData."""
    theta = np.radians(rotation_deg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    coords = adata.obsm["spatial"]
    center = coords.mean(axis=0)
    rotated = (rotation_matrix @ (coords - center).T).T + center + np.array(translation)

    adata_rotated = adata.copy()
    adata_rotated.obsm["spatial"] = rotated
    return adata_rotated


class TestAlignSpatial:
    """Tests for align_spatial function."""

    def test_align_spatial_basic(self):
        """Test basic alignment runs without error."""
        adata_source = _create_test_adata(seed=42)
        adata_target = _create_test_adata(seed=43)

        # Should run without error
        sq.experimental.align_spatial(
            adata_source,
            adata_target,
            niter=100,  # Small for fast test
            diffeo_start=50,
            verbose=False,
        )

        # Check outputs exist
        assert "spatial_aligned" in adata_source.obsm
        assert "spatial_alignment" in adata_source.uns
        assert adata_source.obsm["spatial_aligned"].shape == adata_source.obsm["spatial"].shape

    def test_align_spatial_affine_only(self):
        """Test affine-only alignment."""
        adata_source = _create_test_adata(seed=42)
        adata_target = _create_test_adata(seed=43)

        sq.experimental.align_spatial(
            adata_source,
            adata_target,
            method="affine",
            niter=100,
            verbose=False,
        )

        assert "spatial_aligned" in adata_source.obsm

    def test_align_spatial_with_rotation(self):
        """Test alignment with initial rotation."""
        adata_source = _create_test_adata(seed=42)
        adata_target = _create_rotated_adata(adata_source, rotation_deg=30)

        sq.experimental.align_spatial(
            adata_source,
            adata_target,
            initial_rotation_deg=30,
            niter=100,
            verbose=False,
        )

        assert "spatial_aligned" in adata_source.obsm

    def test_align_spatial_copy(self):
        """Test copy=True returns new AnnData."""
        adata_source = _create_test_adata(seed=42)
        adata_target = _create_test_adata(seed=43)

        result = sq.experimental.align_spatial(
            adata_source,
            adata_target,
            niter=100,
            copy=True,
            verbose=False,
        )

        assert result is not adata_source
        assert "spatial_aligned" in result.obsm
        assert "spatial_aligned" not in adata_source.obsm

    def test_align_spatial_custom_key(self):
        """Test custom key_added parameter."""
        adata_source = _create_test_adata(seed=42)
        adata_target = _create_test_adata(seed=43)

        sq.experimental.align_spatial(
            adata_source,
            adata_target,
            niter=100,
            key_added="my_aligned_coords",
            verbose=False,
        )

        assert "my_aligned_coords" in adata_source.obsm

    def test_align_spatial_missing_spatial_key(self):
        """Test error when spatial key is missing."""
        adata_source = _create_test_adata(seed=42)
        adata_target = _create_test_adata(seed=43)

        with pytest.raises(KeyError, match="not found in adata_source.obsm"):
            sq.experimental.align_spatial(
                adata_source,
                adata_target,
                spatial_key="nonexistent",
                niter=100,
            )

    def test_align_spatial_transform_stored(self):
        """Test that transformation is properly stored."""
        adata_source = _create_test_adata(seed=42)
        adata_target = _create_test_adata(seed=43)

        sq.experimental.align_spatial(
            adata_source,
            adata_target,
            niter=100,
            verbose=False,
        )

        transform = adata_source.uns["spatial_alignment"]
        assert "A" in transform
        assert "v" in transform
        assert "xv" in transform
        assert "loss_history" in transform
        assert transform["A"].shape == (3, 3)

    def test_align_spatial_with_landmarks(self):
        """Test alignment with landmark points."""
        adata_source = _create_test_adata(seed=42)
        adata_target = _create_test_adata(seed=43)

        # Create some landmark points
        landmark_source = np.array([[450, 450], [550, 450], [500, 550]])
        landmark_target = np.array([[460, 460], [560, 460], [510, 560]])

        sq.experimental.align_spatial(
            adata_source,
            adata_target,
            landmark_points_source=landmark_source,
            landmark_points_target=landmark_target,
            niter=100,
            verbose=False,
        )

        assert "spatial_aligned" in adata_source.obsm


class TestRasterizeCoordinates:
    """Tests for rasterize_coordinates function."""

    def test_rasterize_basic(self):
        """Test basic rasterization."""
        coords = np.random.rand(100, 2) * 1000

        X, Y, image = sq.experimental.rasterize_coordinates(coords, resolution=30.0)

        assert X.ndim == 1
        assert Y.ndim == 1
        assert image.ndim == 3
        assert image.shape[0] == 1  # Single channel

    def test_rasterize_resolution(self):
        """Test that resolution affects image size."""
        coords = np.random.rand(100, 2) * 1000

        _, _, image_low = sq.experimental.rasterize_coordinates(coords, resolution=100.0)
        _, _, image_high = sq.experimental.rasterize_coordinates(coords, resolution=20.0)

        # Higher resolution should produce larger image
        assert image_high.shape[1] > image_low.shape[1]
        assert image_high.shape[2] > image_low.shape[2]


class TestApplyTransform:
    """Tests for apply_transform function."""

    def test_apply_transform_basic(self):
        """Test applying saved transform to new coordinates."""
        adata_source = _create_test_adata(seed=42)
        adata_target = _create_test_adata(seed=43)

        sq.experimental.align_spatial(adata_source, adata_target, niter=100, verbose=False)

        # Apply transform to new coordinates
        new_coords = np.random.rand(50, 2) * 100 + 450  # Within data range
        transformed = sq.experimental.apply_transform(
            new_coords, adata_source.uns["spatial_alignment"], direction="source_to_target"
        )

        assert transformed.shape == new_coords.shape

    def test_apply_transform_directions(self):
        """Test both transform directions."""
        adata_source = _create_test_adata(seed=42)
        adata_target = _create_test_adata(seed=43)

        sq.experimental.align_spatial(adata_source, adata_target, niter=100, verbose=False)

        coords = np.array([[500.0, 500.0], [520.0, 520.0]])

        # Transform in both directions
        forward = sq.experimental.apply_transform(
            coords, adata_source.uns["spatial_alignment"], direction="source_to_target"
        )
        backward = sq.experimental.apply_transform(
            coords, adata_source.uns["spatial_alignment"], direction="target_to_source"
        )

        # Results should be different
        assert not np.allclose(forward, backward)
