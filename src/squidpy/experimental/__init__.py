"""Experimental module for Squidpy.

This module contains experimental features that are still under development.
These features may change or be removed in future releases.
"""

from __future__ import annotations

from squidpy.experimental._align import (
    align_images,
    align_spatial,
    align_to_image,
    apply_transform,
    rasterize_coordinates,
    transform_image,
)

from . import im, pl, tl

__all__ = [
    "im",
    "pl",
    "tl",
    # Specific alignment functions (legacy)
    "align_spatial",
    "align_to_image",
    "align_images",
    # Utilities
    "rasterize_coordinates",
    "apply_transform",
    "transform_image",
]
