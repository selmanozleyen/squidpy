#!/usr/bin/env python
"""Test script to demonstrate sq.experimental.tl.align with SpatialData.

This script:
1. Loads a SpatialData object
2. Runs alignment between two images
3. Visualizes and saves results as proof
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Import after checking torch availability
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    import spatialdata as sd
    import squidpy as sq

    # Path to your data
    data_path = Path("/Users/selman/projects/squidpy/data/s1-left-stack-filtered.zarr")

    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        print("Please update the data_path variable to point to your SpatialData.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Loading SpatialData...")
    print(f"{'='*60}")
    sdata = sd.read_zarr(data_path)

    # Show available elements
    print(f"\nSpatialData contents:")
    print(f"  Images: {list(sdata.images.keys())}")
    print(f"  Points: {list(sdata.points.keys()) if sdata.points else 'None'}")
    print(f"  Shapes: {list(sdata.shapes.keys()) if sdata.shapes else 'None'}")
    print(f"  Labels: {list(sdata.labels.keys()) if sdata.labels else 'None'}")

    # Check available scales for each image
    print(f"\nImage scales:")
    for img_key in sdata.images.keys():
        img_node = sdata.images[img_key]
        if hasattr(img_node, "keys"):
            scales = list(img_node.keys())
            print(f"  {img_key}: {scales}")
        else:
            print(f"  {img_key}: single-scale")

    # Select source and target images
    image_keys = list(sdata.images.keys())
    if len(image_keys) < 2:
        print(f"\nERROR: Need at least 2 images for alignment. Found: {image_keys}")
        sys.exit(1)

    # Use first two images (you can modify these)
    source_key = image_keys[0]
    target_key = image_keys[1]

    print(f"\n{'='*60}")
    print(f"Aligning: '{source_key}' -> '{target_key}'")
    print(f"{'='*60}")

    # Run alignment using the new API
    transform = sq.experimental.tl.align(
        sdata,
        source_image_key=source_key,
        target_image_key=target_key,
        scale="auto",  # Uses coarsest scale (smallest image) for efficiency
        method="affine",  # Use affine for faster test; change to "lddmm" for full registration
        niter=500,  # Reduced iterations for quick test
        device="cpu",
        verbose=True,
    )

    print(f"\n{'='*60}")
    print("Alignment Results")
    print(f"{'='*60}")
    print(f"Scale used: {transform['scale_used']}")
    print(f"Method: {transform['method']}")
    print(f"Final loss: {transform['loss_history'][-1][0]:.4f}" if transform['loss_history'] else "N/A")
    print(f"\nAffine matrix A:")
    print(transform['A'])

    # Visualize results
    print(f"\n{'='*60}")
    print("Creating visualization...")
    print(f"{'='*60}")

    # Load images for visualization (use same scale as alignment)
    from squidpy.experimental.tl._align import _get_element_data_for_align, _ensure_3_channels, _normalize_image_range

    source_img, _ = _get_element_data_for_align(sdata, source_key, transform['scale_used'])
    target_img, _ = _get_element_data_for_align(sdata, target_key, transform['scale_used'])

    source_img = _normalize_image_range(_ensure_3_channels(source_img))
    target_img = _normalize_image_range(_ensure_3_channels(target_img))

    # Apply transform to source image
    from squidpy.experimental._lddmm import transform_image_source_to_target

    A = torch.tensor(transform["A"])
    v = torch.tensor(transform["v"])
    xv = [torch.tensor(x) for x in transform["xv"]]

    _, h_I, w_I = source_img.shape
    _, h_J, w_J = target_img.shape

    xI = [np.arange(h_I, dtype=np.float64), np.arange(w_I, dtype=np.float64)]
    XJ = [np.arange(h_J, dtype=np.float64), np.arange(w_J, dtype=np.float64)]

    source_transformed = transform_image_source_to_target(xv, v, A, xI, source_img, XJ)
    source_transformed = source_transformed.cpu().numpy()

    # Convert to HWC format for visualization
    source_hwc = np.moveaxis(source_img, 0, -1)
    target_hwc = np.moveaxis(target_img, 0, -1)
    transformed_hwc = np.moveaxis(source_transformed, 0, -1)

    # For overlay comparison, we need same-sized images
    # Pad or crop to make them comparable - use target size as reference
    def pad_or_crop_to_target(img, target_h, target_w):
        """Pad or crop image to target size."""
        h, w = img.shape[:2]
        result = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)

        # Calculate crop/pad regions
        src_h_start = max(0, (h - target_h) // 2)
        src_w_start = max(0, (w - target_w) // 2)
        dst_h_start = max(0, (target_h - h) // 2)
        dst_w_start = max(0, (target_w - w) // 2)

        copy_h = min(h, target_h)
        copy_w = min(w, target_w)

        result[dst_h_start:dst_h_start+copy_h, dst_w_start:dst_w_start+copy_w] = \
            img[src_h_start:src_h_start+copy_h, src_w_start:src_w_start+copy_w]

        return result

    # Pad source to match target for overlay comparison
    source_padded = pad_or_crop_to_target(source_hwc, h_J, w_J)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Before alignment
    ax = axes[0, 0]
    ax.imshow(source_hwc)
    ax.set_title(f"Source: {source_key}\n{source_hwc.shape[:2]}")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(target_hwc)
    ax.set_title(f"Target: {target_key}\n{target_hwc.shape[:2]}")
    ax.axis("off")

    ax = axes[0, 2]
    # Overlay before alignment (blend) - use padded source
    overlay_before = 0.5 * source_padded + 0.5 * target_hwc
    ax.imshow(overlay_before)
    ax.set_title("Overlay BEFORE alignment\n(source padded to target size)")
    ax.axis("off")

    # Row 2: After alignment
    ax = axes[1, 0]
    ax.imshow(transformed_hwc)
    ax.set_title(f"Source (transformed)\n{transformed_hwc.shape[:2]}")
    ax.axis("off")

    ax = axes[1, 1]
    ax.imshow(target_hwc)
    ax.set_title(f"Target: {target_key}\n{target_hwc.shape[:2]}")
    ax.axis("off")

    ax = axes[1, 2]
    # Overlay after alignment - transformed is already in target space
    overlay_after = 0.5 * transformed_hwc + 0.5 * target_hwc
    ax.imshow(overlay_after)
    ax.set_title("Overlay AFTER alignment")
    ax.axis("off")

    plt.suptitle(f"Alignment: {source_key} -> {target_key}\nScale: {transform['scale_used']}, Method: {transform['method']}", fontsize=14)
    plt.tight_layout()

    # Save figure
    output_path = Path("alignment_result.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path.absolute()}")

    # Also show if in interactive mode
    plt.show()

    # Plot loss history
    if transform['loss_history']:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        losses = [l[0] for l in transform['loss_history']]
        ax2.plot(losses)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Loss")
        ax2.set_title("Optimization Loss History")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        loss_path = Path("alignment_loss.png")
        plt.savefig(loss_path, dpi=150, bbox_inches="tight")
        print(f"Loss plot saved to: {loss_path.absolute()}")
        plt.show()

    print(f"\n{'='*60}")
    print("DONE! Check the saved images for proof of alignment.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
