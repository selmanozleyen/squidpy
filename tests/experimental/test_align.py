"""Tests for spatial alignment functions.

These tests are inspired by STalign (https://github.com/JEFworks-Lab/STalign)
examples and cover various alignment scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if torch not installed
torch = pytest.importorskip("torch")

import anndata as ad

import squidpy as sq


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


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


def _create_brain_like_coordinates(n_cells: int = 1000, seed: int = 42) -> np.ndarray:
    """Create brain-like coronal section coordinates (elliptical shape)."""
    rng = np.random.default_rng(seed)

    # Elliptical brain section shape
    theta = rng.uniform(0, 2 * np.pi, n_cells)
    r = rng.uniform(0.3, 1.0, n_cells)

    # Elongated ellipse
    x = r * np.cos(theta) * 300 + 500
    y = r * np.sin(theta) * 200 + 400

    return np.column_stack([x, y])


def _create_test_image(shape: tuple = (100, 100), pattern: str = "circle", seed: int = 42) -> np.ndarray:
    """Create synthetic test image with various patterns."""
    rng = np.random.default_rng(seed)
    h, w = shape

    if pattern == "circle":
        y, x = np.ogrid[:h, :w]
        center = (h // 2, w // 2)
        r = min(h, w) // 3
        mask = ((x - center[1]) ** 2 + (y - center[0]) ** 2) < r**2
        img = mask.astype(float) * 0.8 + 0.1
    elif pattern == "ellipse":
        y, x = np.ogrid[:h, :w]
        center = (h // 2, w // 2)
        rx, ry = w // 3, h // 4
        mask = ((x - center[1]) / rx) ** 2 + ((y - center[0]) / ry) ** 2 < 1
        img = mask.astype(float) * 0.8 + 0.1
    elif pattern == "noise":
        img = rng.random((h, w))
    else:
        img = np.ones((h, w)) * 0.5

    # Add some noise
    img += rng.normal(0, 0.05, (h, w))
    return np.clip(img, 0, 1)


# =============================================================================
# Tests for align_spatial (coordinate alignment)
# =============================================================================


class TestAlignSpatial:
    """Tests for align_spatial function."""

    def test_align_spatial_basic(self):
        """Test basic alignment runs without error."""
        adata_source = _create_test_adata(seed=42)
        adata_target = _create_test_adata(seed=43)

        sq.experimental.align_spatial(
            adata_source,
            adata_target,
            niter=100,
            diffeo_start=50,
            verbose=False,
        )

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
        assert adata_source.uns["spatial_alignment"]["method"] == "affine"

    def test_align_spatial_with_rotation(self):
        """Test alignment with initial rotation (STalign-like scenario)."""
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
        """Test alignment with landmark points (STalign-like scenario)."""
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


# =============================================================================
# Tests inspired by STalign examples
# =============================================================================


class TestSTalignInspiredScenarios:
    """Tests inspired by STalign tutorial scenarios."""

    def test_brain_section_alignment(self):
        """
        Test aligning two brain-like coronal sections.
        Inspired by: STalign tutorial - aligning MERFISH brain sections.
        """
        # Create two brain-like coordinate patterns
        coords_source = _create_brain_like_coordinates(n_cells=500, seed=42)
        coords_target = _create_brain_like_coordinates(n_cells=500, seed=43)

        adata_source = ad.AnnData(
            X=np.random.rand(500, 10),
            obsm={"spatial": coords_source},
        )
        adata_target = ad.AnnData(
            X=np.random.rand(500, 10),
            obsm={"spatial": coords_target},
        )

        sq.experimental.align_spatial(
            adata_source,
            adata_target,
            niter=100,
            diffeo_start=50,
            verbose=False,
        )

        assert "spatial_aligned" in adata_source.obsm
        aligned = adata_source.obsm["spatial_aligned"]
        assert not np.allclose(aligned, coords_source)  # Coordinates changed

    def test_partial_overlap_alignment(self):
        """
        Test aligning datasets with partial overlap.
        Inspired by: STalign tutorial - aligning partially-matched tissue sections.
        """
        rng = np.random.default_rng(42)

        # Source: full tissue section
        x_source = rng.uniform(0, 1000, 500)
        y_source = rng.uniform(0, 1000, 500)

        # Target: partial section (shifted)
        x_target = rng.uniform(200, 1200, 400)  # Shifted right
        y_target = rng.uniform(100, 900, 400)

        adata_source = ad.AnnData(
            X=np.random.rand(500, 10),
            obsm={"spatial": np.column_stack([x_source, y_source])},
        )
        adata_target = ad.AnnData(
            X=np.random.rand(400, 10),
            obsm={"spatial": np.column_stack([x_target, y_target])},
        )

        sq.experimental.align_spatial(
            adata_source,
            adata_target,
            niter=100,
            verbose=False,
        )

        assert "spatial_aligned" in adata_source.obsm

    def test_different_cell_densities(self):
        """
        Test aligning datasets with different cell densities.
        Inspired by: Cross-technology alignment scenarios.
        """
        rng = np.random.default_rng(42)

        # Source: dense dataset
        x_source = rng.uniform(0, 500, 1000)
        y_source = rng.uniform(0, 500, 1000)

        # Target: sparse dataset
        x_target = rng.uniform(0, 500, 200)
        y_target = rng.uniform(0, 500, 200)

        adata_source = ad.AnnData(
            X=np.random.rand(1000, 10),
            obsm={"spatial": np.column_stack([x_source, y_source])},
        )
        adata_target = ad.AnnData(
            X=np.random.rand(200, 10),
            obsm={"spatial": np.column_stack([x_target, y_target])},
        )

        sq.experimental.align_spatial(
            adata_source,
            adata_target,
            niter=100,
            resolution=50.0,  # Larger resolution for sparse data
            verbose=False,
        )

        assert "spatial_aligned" in adata_source.obsm

    def test_rotated_section_alignment(self):
        """
        Test aligning sections where one is rotated.
        Common in serial section alignment.
        """
        adata_source = _create_test_adata(n_cells=500, seed=42)

        # Create rotated target (30 degrees - smaller rotation for faster convergence)
        adata_target = _create_rotated_adata(adata_source, rotation_deg=30, translation=(20, 20))

        # Align with initial rotation hint
        sq.experimental.align_spatial(
            adata_source,
            adata_target,
            initial_rotation_deg=30,
            niter=200,
            diffeo_start=50,
            verbose=False,
        )

        aligned = adata_source.obsm["spatial_aligned"]

        # Simply check that alignment completes and produces output
        assert "spatial_aligned" in adata_source.obsm
        assert aligned.shape == adata_source.obsm["spatial"].shape
        # Coordinates should have changed
        assert not np.allclose(aligned, adata_source.obsm["spatial"])


# =============================================================================
# Tests for rasterize_coordinates
# =============================================================================


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

    def test_rasterize_blur(self):
        """Test blur parameter affects smoothness."""
        coords = np.random.rand(50, 2) * 100

        _, _, img_sharp = sq.experimental.rasterize_coordinates(coords, resolution=5.0, blur=0.5)
        _, _, img_smooth = sq.experimental.rasterize_coordinates(coords, resolution=5.0, blur=3.0)

        # Smoother image should have less variance
        assert img_smooth.var() < img_sharp.var()


# =============================================================================
# Tests for apply_transform
# =============================================================================


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

    def test_apply_transform_consistency(self):
        """Test that applying transform matches alignment result."""
        adata_source = _create_test_adata(n_cells=200, seed=42)
        adata_target = _create_test_adata(seed=43)

        sq.experimental.align_spatial(adata_source, adata_target, niter=100, verbose=False)

        # Re-apply transform to original coordinates
        reapplied = sq.experimental.apply_transform(
            adata_source.obsm["spatial"],
            adata_source.uns["spatial_alignment"],
            direction="source_to_target",
        )

        # Should match the stored aligned coordinates
        np.testing.assert_array_almost_equal(reapplied, adata_source.obsm["spatial_aligned"], decimal=5)


# =============================================================================
# Tests for align_images
# =============================================================================


class TestAlignImages:
    """Tests for image alignment functions."""

    def test_align_images_basic(self):
        """Test basic image alignment."""
        source_img = _create_test_image(shape=(50, 50), pattern="circle", seed=42)
        target_img = _create_test_image(shape=(50, 50), pattern="circle", seed=43)

        transform = sq.experimental.align_images(
            source_img,
            target_img,
            niter=50,
            verbose=False,
        )

        assert "A" in transform
        assert "v" in transform
        assert "xv" in transform
        assert transform["A"].shape == (3, 3)

    def test_align_images_affine(self):
        """Test affine-only image alignment."""
        source_img = _create_test_image(shape=(50, 50), pattern="ellipse", seed=42)
        target_img = _create_test_image(shape=(50, 50), pattern="ellipse", seed=43)

        transform = sq.experimental.align_images(
            source_img,
            target_img,
            method="affine",
            niter=50,
            verbose=False,
        )

        assert transform["method"] == "affine"

    def test_align_images_different_shapes(self):
        """Test aligning images with different shapes."""
        source_img = _create_test_image(shape=(40, 60), pattern="circle", seed=42)
        target_img = _create_test_image(shape=(50, 50), pattern="circle", seed=43)

        transform = sq.experimental.align_images(
            source_img,
            target_img,
            niter=50,
            verbose=False,
        )

        assert "A" in transform

    def test_align_images_rgb(self):
        """Test aligning RGB images."""
        rng = np.random.default_rng(42)
        source_img = rng.random((50, 50, 3))
        target_img = rng.random((50, 50, 3))

        transform = sq.experimental.align_images(
            source_img,
            target_img,
            niter=50,
            verbose=False,
        )

        assert "A" in transform


# =============================================================================
# Tests for transform_image
# =============================================================================


class TestTransformImage:
    """Tests for image transformation function."""

    def test_transform_image_basic(self):
        """Test basic image transformation."""
        source_img = _create_test_image(shape=(50, 50), pattern="circle", seed=42)
        target_img = _create_test_image(shape=(50, 50), pattern="circle", seed=43)

        transform = sq.experimental.align_images(
            source_img,
            target_img,
            niter=50,
            verbose=False,
        )

        transformed = sq.experimental.transform_image(source_img, transform)

        assert transformed.shape == source_img.shape

    def test_transform_image_preserves_format(self):
        """Test that image format is preserved."""
        rng = np.random.default_rng(42)

        # Test 2D image
        img_2d = rng.random((50, 50))
        transform = sq.experimental.align_images(img_2d, img_2d, niter=20, verbose=False)
        result_2d = sq.experimental.transform_image(img_2d, transform)
        assert result_2d.ndim == 2

        # Test HWC image
        img_hwc = rng.random((50, 50, 3))
        transform = sq.experimental.align_images(img_hwc, img_hwc, niter=20, verbose=False)
        result_hwc = sq.experimental.transform_image(img_hwc, transform)
        assert result_hwc.shape == img_hwc.shape

    def test_transform_image_custom_output_shape(self):
        """Test transformation with custom output shape."""
        source_img = _create_test_image(shape=(50, 50), pattern="circle", seed=42)
        target_img = _create_test_image(shape=(50, 50), pattern="circle", seed=43)

        transform = sq.experimental.align_images(
            source_img,
            target_img,
            niter=50,
            verbose=False,
        )

        transformed = sq.experimental.transform_image(source_img, transform, output_shape=(60, 60))

        assert transformed.shape == (60, 60)


# =============================================================================
# Tests for align_to_image (coordinate-to-image alignment)
# =============================================================================


class TestAlignToImage:
    """Tests for align_to_image function (STalign-like cell-to-histology alignment)."""

    def test_align_to_image_basic(self):
        """Test basic coordinate-to-image alignment."""
        adata_source = _create_test_adata(n_cells=300, seed=42)
        target_img = _create_test_image(shape=(50, 50), pattern="circle", seed=43)

        sq.experimental.align_to_image(
            adata_source,
            target_img,
            niter=50,
            resolution=20.0,
            verbose=False,
        )

        assert "spatial_aligned" in adata_source.obsm
        assert "spatial_alignment" in adata_source.uns
        assert "target_image_shape" in adata_source.uns["spatial_alignment"]

    def test_align_to_image_rgb(self):
        """Test alignment to RGB histology-like image."""
        adata_source = _create_test_adata(n_cells=300, seed=42)

        # Create RGB target image
        rng = np.random.default_rng(43)
        target_img = rng.random((50, 50, 3))

        sq.experimental.align_to_image(
            adata_source,
            target_img,
            niter=50,
            resolution=20.0,
            verbose=False,
        )

        assert "spatial_aligned" in adata_source.obsm

    def test_align_to_image_with_rotation(self):
        """Test coordinate-to-image alignment with initial rotation."""
        adata_source = _create_test_adata(n_cells=300, seed=42)
        target_img = _create_test_image(shape=(50, 50), pattern="ellipse", seed=43)

        sq.experimental.align_to_image(
            adata_source,
            target_img,
            initial_rotation_deg=30,
            niter=50,
            resolution=20.0,
            verbose=False,
        )

        assert "spatial_aligned" in adata_source.obsm

    def test_align_to_image_affine_only(self):
        """Test affine-only coordinate-to-image alignment."""
        adata_source = _create_test_adata(n_cells=300, seed=42)
        target_img = _create_test_image(shape=(50, 50), pattern="circle", seed=43)

        sq.experimental.align_to_image(
            adata_source,
            target_img,
            method="affine",
            niter=50,
            resolution=20.0,
            verbose=False,
        )

        assert adata_source.uns["spatial_alignment"]["method"] == "affine"

    def test_align_to_image_copy(self):
        """Test copy mode for align_to_image."""
        adata_source = _create_test_adata(n_cells=300, seed=42)
        target_img = _create_test_image(shape=(50, 50), pattern="circle", seed=43)

        result = sq.experimental.align_to_image(
            adata_source,
            target_img,
            niter=50,
            resolution=20.0,
            copy=True,
            verbose=False,
        )

        assert result is not adata_source
        assert "spatial_aligned" in result.obsm
        assert "spatial_aligned" not in adata_source.obsm

    def test_align_to_image_with_landmarks(self):
        """Test coordinate-to-image alignment with landmark points."""
        adata_source = _create_test_adata(n_cells=300, seed=42)
        target_img = _create_test_image(shape=(50, 50), pattern="circle", seed=43)

        # Create landmark points
        landmark_source = np.array([[450, 450], [550, 450], [500, 550]])
        landmark_target = np.array([[20, 20], [30, 20], [25, 30]])

        sq.experimental.align_to_image(
            adata_source,
            target_img,
            landmark_points_source=landmark_source,
            landmark_points_target=landmark_target,
            niter=50,
            resolution=20.0,
            verbose=False,
        )

        assert "spatial_aligned" in adata_source.obsm


class TestSpatialDataImageSupport:
    """Tests for spatialdata image integration."""

    def test_extract_image_formats(self):
        """Test that different image formats are handled correctly."""
        # Test 2D grayscale
        img_2d = np.random.rand(50, 50)
        result = sq.experimental.align_images(img_2d, img_2d, niter=20, verbose=False)
        assert "A" in result

        # Test HWC format
        img_hwc = np.random.rand(50, 50, 3)
        result = sq.experimental.align_images(img_hwc, img_hwc, niter=20, verbose=False)
        assert "A" in result

        # Test CHW format
        img_chw = np.random.rand(3, 50, 50)
        result = sq.experimental.align_images(img_chw, img_chw, niter=20, verbose=False)
        assert "A" in result

    def test_align_images_with_extent(self):
        """Test image alignment with physical coordinate extents."""
        source_img = _create_test_image(shape=(50, 50), pattern="circle", seed=42)
        target_img = _create_test_image(shape=(50, 50), pattern="circle", seed=43)

        # Define physical extents
        source_extent = (0, 1000, 0, 1000)
        target_extent = (100, 1100, 100, 1100)

        transform = sq.experimental.align_images(
            source_img,
            target_img,
            source_extent=source_extent,
            target_extent=target_extent,
            niter=50,
            verbose=False,
        )

        assert transform["source_extent"] == source_extent
        assert transform["target_extent"] == target_extent


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow_coordinates(self):
        """Test full workflow: align -> transform new points."""
        # Create source and target
        adata_source = _create_test_adata(n_cells=300, seed=42)
        adata_target = _create_rotated_adata(adata_source, rotation_deg=20, translation=(30, 30))

        # Align
        sq.experimental.align_spatial(
            adata_source,
            adata_target,
            initial_rotation_deg=20,
            niter=100,
            verbose=False,
        )

        # Generate new points and transform them
        new_points = np.random.rand(50, 2) * 100 + 450
        transformed = sq.experimental.apply_transform(
            new_points,
            adata_source.uns["spatial_alignment"],
            direction="source_to_target",
        )

        assert transformed.shape == new_points.shape
        assert not np.allclose(transformed, new_points)

    def test_full_workflow_images(self):
        """Test full workflow: align images -> transform image."""
        source_img = _create_test_image(shape=(50, 50), pattern="circle", seed=42)
        target_img = _create_test_image(shape=(50, 50), pattern="circle", seed=43)

        # Align
        transform = sq.experimental.align_images(
            source_img,
            target_img,
            niter=50,
            verbose=False,
        )

        # Transform
        transformed = sq.experimental.transform_image(source_img, transform)

        assert transformed.shape == source_img.shape
        # Check that loss history was recorded
        assert len(transform["loss_history"]) == 50
        assert "A" in transform

    def test_full_workflow_coordinates_to_image(self):
        """Test full workflow: align coordinates to image -> transform new points."""
        adata_source = _create_test_adata(n_cells=300, seed=42)
        target_img = _create_test_image(shape=(50, 50), pattern="ellipse", seed=43)

        # Align coordinates to image
        sq.experimental.align_to_image(
            adata_source,
            target_img,
            niter=50,
            resolution=20.0,
            verbose=False,
        )

        # Apply transform to new coordinates
        new_coords = np.random.rand(20, 2) * 100 + 450
        transformed = sq.experimental.apply_transform(
            new_coords,
            adata_source.uns["spatial_alignment"],
            direction="source_to_target",
        )

        assert transformed.shape == new_coords.shape

    def test_stalign_merfish_visium_scenario(self):
        """
        Test scenario inspired by STalign MERFISH-Visium alignment tutorial.

        This simulates aligning single-cell MERFISH data to a Visium
        histology image.
        """
        # Create MERFISH-like dense single-cell data
        rng = np.random.default_rng(42)
        n_cells = 500
        x_merfish = rng.uniform(0, 1000, n_cells)
        y_merfish = rng.uniform(0, 1000, n_cells)

        adata_merfish = ad.AnnData(
            X=np.random.rand(n_cells, 100),
            obsm={"spatial": np.column_stack([x_merfish, y_merfish])},
        )

        # Create Visium-like histology image
        histology_img = _create_test_image(shape=(100, 100), pattern="ellipse", seed=43)

        # Align MERFISH to histology
        sq.experimental.align_to_image(
            adata_merfish,
            histology_img,
            resolution=30.0,
            niter=50,
            verbose=False,
        )

        assert "spatial_aligned" in adata_merfish.obsm

        # Verify we can apply transform to new cells
        new_cells = np.random.rand(10, 2) * 1000
        transformed = sq.experimental.apply_transform(
            new_cells,
            adata_merfish.uns["spatial_alignment"],
            direction="source_to_target",
        )
        assert transformed.shape == new_cells.shape
