"""Public ``*Params`` types for :mod:`squidpy.experimental`, and their defaults.

Every tunable-parameter mapping the experimental API accepts lives here: eight
:class:`~typing.TypedDict` declarations, each paired with the ``_*_DEFAULTS``
mapping that fills its absent keys. They are declarations only -- the functions
that consume them, and the validators that range-check them, stay next to the
code they belong to.

All are ``total=False``: callers pass a partial mapping, get static key and value
checking at the call site, and :func:`squidpy.experimental.utils._params.resolve_params`
merges it over the defaults and validates the result at runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from squidpy._utils import RNGLike, SeedLike

#: Default Ruderman Lab-L cutoff for the Reinhard luminosity mask, mirroring
#: :data:`squidpy.experimental.im._stain._constants.DEFAULT_LUMINOSITY_THRESHOLD`.
#: Held separately because every `experimental` implementation module imports this
#: one, so it must import nothing from them or the import graph closes on itself.
#: A test asserts the two stay equal.
DEFAULT_LUMINOSITY_THRESHOLD: float = 0.8

__all__ = [
    "BackgroundDetectionParams",
    "FelzenszwalbParams",
    "WekaParams",
    "ReinhardParams",
    "MacenkoParams",
    "VahadaneParams",
    "TilingQCParams",
    "StitchParams",
]


class BackgroundDetectionParams(TypedDict, total=False):
    """Which corners are background, and how large the corner boxes should be.

    If no corners are flagged ``True``, orientation falls back to bright
    background -- see ``any_corner``.
    """

    ymin_xmin_is_bg: bool
    """Whether the ``(ymin, xmin)`` corner is background."""

    ymax_xmin_is_bg: bool
    """Whether the ``(ymax, xmin)`` corner is background."""

    ymin_xmax_is_bg: bool
    """Whether the ``(ymin, xmax)`` corner is background."""

    ymax_xmax_is_bg: bool
    """Whether the ``(ymax, xmax)`` corner is background."""

    corner_size_pct: float
    """Corner box size as a fraction of height/width."""


#: Annotated with the TypedDicts so the type checker verifies every default.
_BACKGROUND_DEFAULTS: BackgroundDetectionParams = {
    "ymin_xmin_is_bg": True,
    "ymax_xmin_is_bg": True,
    "ymin_xmax_is_bg": True,
    "ymax_xmax_is_bg": True,
    "corner_size_pct": 0.01,
}


class FelzenszwalbParams(TypedDict, total=False):
    """Size-aware superpixel defaults for felzenszwalb segmentation.
    """

    grid_rows: int
    """Target superpixel grid rows."""

    grid_cols: int
    """Target superpixel grid columns."""

    sigma_frac: float
    """Blur = this * short side, clipped to ``[1, 5]`` px."""

    scale_coef: float
    """``scale`` = coef * target_area."""

    min_size_coef: float
    """``min_size`` = coef * target_area."""


_FELZENSZWALB_DEFAULTS: FelzenszwalbParams = {
    "grid_rows": 100,
    "grid_cols": 100,
    "sigma_frac": 0.008,
    "scale_coef": 0.25,
    "min_size_coef": 0.20,
}


class WekaParams(TypedDict, total=False):
    """Parameters for WEKA-like trainable segmentation.
    """

    sigma_min: float
    """Smallest scale in the multiscale feature bank."""

    sigma_max: float
    """Largest scale in the multiscale feature bank."""

    edges: bool
    """Include edge features."""

    pseudo_tissue_percentile: float
    """Percentile of distance-from-bg to label as tissue."""

    pseudo_min_pixels: int
    """Minimum number of tissue pixels to seed."""

    rf_estimators: int
    """Number of trees in the random forest."""

    rf_max_depth: int | None
    """Maximum tree depth; ``None`` for unlimited."""

    rf_max_samples: float
    """Fraction of samples drawn to train each tree."""

    rng: SeedLike | RNGLike | None
    """Source of randomness; ``None`` draws from OS entropy."""

    refine_with_classifier: bool
    """Run the second-stage background refinement."""

    refine_n_samples_per_class: int
    """Training samples drawn per class in the refinement step."""

    refine_bg_prob_threshold: float
    """Only drop pixels whose background probability exceeds this."""

    border_margin_px: int | Sequence[int]
    """Border ignored when seeding and predicting."""


_WEKA_DEFAULTS: WekaParams = {
    "sigma_min": 1.0,
    "sigma_max": 16.0,
    "edges": True,
    "pseudo_tissue_percentile": 90.0,
    "pseudo_min_pixels": 50,
    "rf_estimators": 100,
    "rf_max_depth": 10,
    "rf_max_samples": 0.05,
    "rng": None,
    "refine_with_classifier": True,
    "refine_n_samples_per_class": 50_000,
    "refine_bg_prob_threshold": 0.6,
    "border_margin_px": 0,
}


class ReinhardParams(TypedDict, total=False):
    """Tuning knobs for Reinhard stain normalization.

    Pass a mapping of these keys as ``method_params``; every key is optional
    and falls back to ``_REINHARD_DEFAULTS``. Values are coerced and
    range-checked by ``validate_reinhard_params`` when resolved.
    """

    luminosity_threshold: float
    """Normalised Ruderman Lab-L cutoff in ``(0, 1]``; pixels brighter than this are excluded from the fit."""

    mask_background: bool
    """If ``True``, fit channel statistics over tissue pixels only; if ``False``, use every pixel (vanilla Reinhard)."""


#: Annotated with the TypedDict so the type checker verifies every default.
_REINHARD_DEFAULTS: ReinhardParams = {
    "luminosity_threshold": DEFAULT_LUMINOSITY_THRESHOLD,
    "mask_background": True,
}


class MacenkoParams(TypedDict, total=False):
    """Tuning knobs for Macenko stain-matrix fitting.
    """

    alpha: float
    """Angular percentile (deg) for the two stain directions; the extremes are taken at ``alpha`` / ``100 - alpha``."""

    beta: float
    """Mean-absorbance cutoff selecting tissue pixels (optical-density space)."""


#: Annotated with the TypedDicts so the type checker verifies every default.
_MACENKO_DEFAULTS: MacenkoParams = {"alpha": 1.0, "beta": 0.15}


class VahadaneParams(TypedDict, total=False):
    """Tuning knobs for Vahadane (sparse-NMF) stain-matrix fitting.
    """

    beta: float
    """Mean-absorbance cutoff selecting tissue pixels (optical-density space)."""

    lambda1: float
    """L1 sparsity regularisation on the concentration factor of the NMF."""

    n_iter: int
    """Maximum NMF iterations."""

    rng: SeedLike | RNGLike | None
    """Source of randomness for NMF initialisation tie-breaking; ``None`` draws from OS entropy."""


_VAHADANE_DEFAULTS: VahadaneParams = {"beta": 0.15, "lambda1": 0.1, "n_iter": 200, "rng": None}


class TilingQCParams(TypedDict, total=False):
    """Advanced tuning knobs for :func:`~squidpy.experimental.tl.calculate_tiling_qc`.

    Pass a mapping of these keys as ``tiling_qc_params``; every key is optional
    and falls back to ``_QC_DEFAULTS``. Values are coerced and range-checked by
    ``validate_qc_params`` when resolved.
    """

    distance_tol: float
    """Maximum perpendicular distance (pixels) from the fitted line for a contour point to count as straight."""

    min_area: int
    """Cells smaller than this (pixels at analysis resolution) are skipped (NaN scores)."""

    max_contour_points: int
    """Cap on contour resolution; longer contours are arc-length-resampled before the O(n^2) collinearity scan."""


#: Annotated with the TypedDict so the type checker verifies every default.
_QC_DEFAULTS: TilingQCParams = {
    "distance_tol": 0.75,
    "min_area": 20,
    "max_contour_points": 500,
}


class StitchParams(TypedDict, total=False):
    """Advanced tuning knobs for :func:`~squidpy.experimental.tl.assign_stitch_groups`.

    Defaults work for typical 2D segmentation tiles produced by cellpose-like
    pipelines. Pass a mapping of these keys as ``stitch_params``; every key is
    optional and falls back to ``_STITCH_DEFAULTS``. Values are coerced and
    range-checked by ``validate_stitch_params`` when resolved. These are
    advanced knobs -- the defaults rarely need changing.
    """

    distance_tol: float
    """Sub-pixel tolerance for "lies on a bbox edge"."""

    min_edge_length: float
    """Absolute floor on cut-edge length (pixels)."""

    min_edge_length_ratio: float
    """Minimum cut-edge length relative to the cell's equivalent diameter."""

    min_edge_coverage: float
    """Minimum fraction of parallel-axis positions covered by near-edge contour points."""

    candidate_min_iou: float
    """Loose 1-D IoU floor at candidate enumeration."""

    close_radius: int
    """Morphological closing disk radius for the union mask. Also the length scale for
    ``gap_proximity`` (normalised by ``2 * close_radius``)."""


#: Annotated with the TypedDict so the type checker verifies every default.
_STITCH_DEFAULTS: StitchParams = {
    "distance_tol": 0.75,
    "min_edge_length": 5.0,
    "min_edge_length_ratio": 0.4,
    "min_edge_coverage": 0.5,
    "candidate_min_iou": 0.2,
    "close_radius": 3,
}
