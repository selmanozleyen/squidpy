#!/usr/bin/env python
"""Quick script to visualize the affine alignment result without re-running LDDMM."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import spatialdata as sd

# The affine matrix from your previous run
A = np.array([
    [1.05124098e+00, -4.30475697e-02, 7.15302695e+00],
    [8.50713671e-03, 9.13436548e-01, -4.43275975e+01],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

print("Loading SpatialData...")
sdata = sd.read_zarr("/Users/selman/projects/squidpy/data/s1-left-stack-filtered.zarr")

source_key = "s1-ss6-left-hne"
target_key = "s1-ss8-left-hne"
scale = "scale3"  # The coarsest scale used in alignment

print(f"Loading images at {scale}...")

# Get images
source_node = sdata.images[source_key][scale].image
target_node = sdata.images[target_key][scale].image

source_img = np.asarray(source_node.values)
target_img = np.asarray(target_node.values)

print(f"Source shape: {source_img.shape}")
print(f"Target shape: {target_img.shape}")

# Normalize to [0, 1]
source_img = (source_img - source_img.min()) / (source_img.max() - source_img.min() + 1e-8)
target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)

# Apply affine transform using torch grid_sample
_, h_I, w_I = source_img.shape
_, h_J, w_J = target_img.shape

print(f"Applying affine transform...")

# Convert to torch tensors
source_tensor = torch.tensor(source_img, dtype=torch.float32).unsqueeze(0)  # (1, C, H, W)

# Create grid for target image coordinates
# grid_sample expects grid in [-1, 1] range
y_coords = torch.linspace(-1, 1, h_J)
x_coords = torch.linspace(-1, 1, w_J)
grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

# Stack to (H, W, 2) with (x, y) order for grid_sample
grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

# Convert grid to pixel coordinates, apply inverse affine, convert back
# Grid is in [-1, 1], need to convert to pixel coords
grid_pixels_x = (grid[..., 0] + 1) / 2 * (w_J - 1)
grid_pixels_y = (grid[..., 1] + 1) / 2 * (h_J - 1)

# Stack as homogeneous coordinates (row, col, 1)
ones = torch.ones_like(grid_pixels_x)
grid_homogeneous = torch.stack([grid_pixels_y, grid_pixels_x, ones], dim=-1)  # (H, W, 3)

# Apply INVERSE of affine to get source coordinates
# A maps source -> target, so A_inv maps target -> source
A_inv = np.linalg.inv(A)
A_inv_tensor = torch.tensor(A_inv, dtype=torch.float32)

# Reshape for matrix multiplication
grid_flat = grid_homogeneous.reshape(-1, 3)  # (H*W, 3)
source_coords = (A_inv_tensor @ grid_flat.T).T  # (H*W, 3)
source_coords = source_coords.reshape(h_J, w_J, 3)

# Convert back to normalized [-1, 1] for grid_sample
source_y_norm = source_coords[..., 0] / (h_I - 1) * 2 - 1
source_x_norm = source_coords[..., 1] / (w_I - 1) * 2 - 1

# grid_sample expects (x, y) order
sample_grid = torch.stack([source_x_norm, source_y_norm], dim=-1).unsqueeze(0)  # (1, H, W, 2)

# Apply transform
transformed = torch.nn.functional.grid_sample(
    source_tensor,
    sample_grid,
    mode='bilinear',
    padding_mode='zeros',
    align_corners=True
)
transformed = transformed.squeeze(0).numpy()  # (C, H, W)

print(f"Transformed shape: {transformed.shape}")

# Convert to HWC for visualization
source_hwc = np.moveaxis(source_img, 0, -1)
target_hwc = np.moveaxis(target_img, 0, -1)
transformed_hwc = np.moveaxis(transformed, 0, -1)

# Clip to valid range
transformed_hwc = np.clip(transformed_hwc, 0, 1)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Individual images
axes[0, 0].imshow(source_hwc)
axes[0, 0].set_title(f"Source: {source_key}\n{source_hwc.shape[:2]}")
axes[0, 0].axis("off")

axes[0, 1].imshow(target_hwc)
axes[0, 1].set_title(f"Target: {target_key}\n{target_hwc.shape[:2]}")
axes[0, 1].axis("off")

axes[0, 2].imshow(transformed_hwc)
axes[0, 2].set_title(f"Source (transformed)\n{transformed_hwc.shape[:2]}")
axes[0, 2].axis("off")

# Row 2: Overlays and difference
# Overlay: blend transformed source with target
overlay = 0.5 * transformed_hwc + 0.5 * target_hwc
axes[1, 0].imshow(overlay)
axes[1, 0].set_title("Overlay (50/50 blend)")
axes[1, 0].axis("off")

# Checkerboard comparison
checker_size = 100
checker = np.zeros((h_J, w_J), dtype=bool)
for i in range(0, h_J, checker_size):
    for j in range(0, w_J, checker_size):
        if ((i // checker_size) + (j // checker_size)) % 2 == 0:
            checker[i:i+checker_size, j:j+checker_size] = True

checkerboard = np.where(checker[..., None], transformed_hwc, target_hwc)
axes[1, 1].imshow(checkerboard)
axes[1, 1].set_title("Checkerboard comparison")
axes[1, 1].axis("off")

# Difference image (absolute)
diff = np.abs(transformed_hwc - target_hwc)
axes[1, 2].imshow(diff)
axes[1, 2].set_title(f"Absolute difference\nMean: {diff.mean():.4f}")
axes[1, 2].axis("off")

plt.suptitle(f"Affine Alignment Result\n{source_key} → {target_key} (scale: {scale})", fontsize=14)
plt.tight_layout()

output_path = "affine_alignment_result.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved visualization to: {output_path}")

plt.show()

# Also print the affine matrix interpretation
print("\n" + "="*60)
print("Affine Matrix Interpretation:")
print("="*60)
print(f"A =\n{A}")
print(f"\nScale X: {A[0,0]:.4f} ({(A[0,0]-1)*100:+.1f}%)")
print(f"Scale Y: {A[1,1]:.4f} ({(A[1,1]-1)*100:+.1f}%)")
print(f"Rotation (approx): {np.degrees(np.arctan2(A[1,0], A[0,0])):.2f} degrees")
print(f"Translation X: {A[0,2]:.2f} pixels")
print(f"Translation Y: {A[1,2]:.2f} pixels")
