"""LDDMM (Large Deformation Diffeomorphic Metric Mapping) module.

This module provides functions for diffeomorphic registration of spatial
transcriptomics data, ported from STalign (https://github.com/JEFworks-Lab/STalign).
"""

from __future__ import annotations

from squidpy.experimental._lddmm._core import LDDMM
from squidpy.experimental._lddmm._rasterize import normalize, rasterize
from squidpy.experimental._lddmm._transforms import (
    L_T_from_points,
    build_transform,
    extent_from_x,
    transform_image_source_to_target,
    transform_points_source_to_target,
    transform_points_target_to_source,
)

__all__ = [
    "normalize",
    "rasterize",
    "LDDMM",
    "extent_from_x",
    "L_T_from_points",
    "build_transform",
    "transform_points_source_to_target",
    "transform_points_target_to_source",
    "transform_image_source_to_target",
]
