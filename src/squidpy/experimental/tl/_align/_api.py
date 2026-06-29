"""Public alignment functions built on the :mod:`squidpy.experimental.methods` core.

These are thin orchestrators: build (or accept) an :class:`~squidpy.experimental.methods.Aligner`,
resolve inputs to in-memory arrays, run the aligner, write the result back. All
container I/O and write-back live in :mod:`._io`; the aligners themselves never
see a container.

The surface mirrors :func:`squidpy.gr.spatial_neighbors_knn` and friends: one thin
function per method (:func:`align_stalign`, :func:`align_landmarks_similarity`,
:func:`align_landmarks_affine`) plus a ``*_from_aligner`` escape hatch
(:func:`align_from_aligner`, :func:`align_landmarks_from_aligner`) that runs an
explicit aligner instance -- the bridge to custom aligners (see :doc:`/extensibility`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

from squidpy._validators import assert_one_of
from squidpy.experimental.methods import (
    AffineAligner,
    Aligner,
    SimilarityAligner,
    StalignAligner,
    StalignConfig,
)
from squidpy.experimental.tl._align._io import (
    get_coords,
    resolve_obs_pair,
    writeback_affine_sdata,
    writeback_obs,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from squidpy.experimental.methods import AffineFitResult, AlignResult

OUTPUT_MODES = ("object", "copy", "inplace")
ON_VALUES = ("obs", "image")

__all__ = [
    "align_stalign",
    "align_from_aligner",
    "align_landmarks_similarity",
    "align_landmarks_affine",
    "align_landmarks_from_aligner",
]


# ---------------------------------------------------------------------------
# Sample alignment (point clouds): align_samples family
# ---------------------------------------------------------------------------


def align_stalign(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None = None,
    *,
    on: Literal["obs", "image"] = "obs",
    ref_key: str | None = None,
    query_key: str | None = None,
    spatial_key: str = "spatial",
    output_mode: Literal["object", "copy", "inplace"] = "object",
    key_added: str | None = None,
    landmarks_source: npt.ArrayLike | None = None,
    landmarks_target: npt.ArrayLike | None = None,
    dx: float = 30.0,
    blur: float | Sequence[float] = (2.0, 1.0, 0.5),
    raster_expand: float = 1.1,
    a: float = 500.0,
    p: float = 2.0,
    expand: float = 2.0,
    nt: int = 3,
    niter: int = 5000,
    diffeo_start: int = 0,
    epL: float = 2e-8,
    epT: float = 2e-1,
    epV: float = 2e3,
    sigmaM: float = 1.0,
    sigmaB: float = 2.0,
    sigmaA: float = 5.0,
    sigmaR: float = 5e5,
    sigmaP: float = 2e1,
) -> AlignResult | AnnData | SpatialData | None:
    """Align a query sample onto a reference sample with STalign (JAX LDDMM).

    Thin wrapper over :class:`~squidpy.experimental.methods.StalignAligner`; the
    solver knobs below map one-to-one onto :class:`~squidpy.experimental.methods.StalignConfig`.

    Parameters
    ----------
    data_ref, data_query
        Both :class:`~anndata.AnnData`, or both :class:`~spatialdata.SpatialData`,
        or ``data_ref`` a SpatialData with ``data_query=None`` to align two of its
        own tables (selected by ``ref_key`` / ``query_key``).
    on
        ``"obs"`` aligns the ``obsm`` point clouds. ``"image"`` is reserved and
        currently raises :class:`NotImplementedError`.
    ref_key, query_key
        Table keys, required (and only valid) for SpatialData inputs.
    spatial_key
        ``obsm`` key holding the ``(x, y)`` coordinates. Defaults to ``"spatial"``.
    output_mode
        - ``"object"`` (default) -- return the fitted :class:`~squidpy.experimental.tl.AlignResult`; nothing is written.
        - ``"inplace"`` -- write the aligned coordinates into the query and return ``None``.
        - ``"copy"`` -- write into a copy of the query and return the copy.
    key_added
        Destination ``obsm`` key for the aligned coordinates. If ``None`` it
        defaults to ``f"aligned_{spatial_key}"``; if that key already exists and
        ``key_added`` was not given explicitly, a :class:`ValueError` is raised
        (pass ``key_added`` to overwrite intentionally).
    landmarks_source, landmarks_target
        Optional corresponding ``(x, y)`` landmark arrays used to initialise the
        affine. Must be provided together.
    dx, blur, raster_expand
        Rasterization of the point clouds into density images: grid spacing,
        Gaussian blur scale(s), and field-of-view padding factor.
    a, p, expand, nt, niter, diffeo_start
        LDDMM controls: kernel width ``a``, regularisation power ``p``,
        velocity-grid padding ``expand``, number of integration time steps
        ``nt``, iterations ``niter``, and the iteration at which the
        diffeomorphic (non-affine) part starts updating ``diffeo_start``.
    epL, epT, epV
        Gradient-descent step sizes for the linear part, translation, and
        velocity field.
    sigmaM, sigmaB, sigmaA, sigmaR, sigmaP
        Noise scales for the matching, background, artifact, regularisation, and
        landmark-point terms of the objective.

    See Also
    --------
    align_from_aligner : Use a pre-built :class:`~squidpy.experimental.methods.StalignAligner` (or a custom aligner).
    """
    config = StalignConfig(
        landmarks_source=landmarks_source,
        landmarks_target=landmarks_target,
        dx=dx,
        blur=blur,
        raster_expand=raster_expand,
        a=a,
        p=p,
        expand=expand,
        nt=nt,
        niter=niter,
        diffeo_start=diffeo_start,
        epL=epL,
        epT=epT,
        epV=epV,
        sigmaM=sigmaM,
        sigmaB=sigmaB,
        sigmaA=sigmaA,
        sigmaR=sigmaR,
        sigmaP=sigmaP,
    )
    return _run_align_samples(
        StalignAligner(config),
        data_ref,
        data_query,
        on=on,
        ref_key=ref_key,
        query_key=query_key,
        spatial_key=spatial_key,
        output_mode=output_mode,
        key_added=key_added,
    )


def align_from_aligner(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None,
    aligner: Aligner,
    *,
    on: Literal["obs", "image"] = "obs",
    ref_key: str | None = None,
    query_key: str | None = None,
    spatial_key: str = "spatial",
    output_mode: Literal["object", "copy", "inplace"] = "object",
    key_added: str | None = None,
) -> AlignResult | AnnData | SpatialData | None:
    """Align a query sample onto a reference sample using an explicit aligner instance.

    The escape hatch for sample alignment: pass any
    :class:`~squidpy.experimental.methods.Aligner` (a built-in
    :class:`~squidpy.experimental.methods.StalignAligner`, or your own subclass --
    see :doc:`/extensibility`). The aligner's
    :meth:`~squidpy.experimental.methods.Aligner.align` receives the resolved
    ``(x, y)`` point clouds; this function handles all container I/O and write-back.

    Parameters
    ----------
    data_ref, data_query, on, ref_key, query_key, spatial_key, output_mode, key_added
        See :func:`align_stalign`.
    aligner
        The :class:`~squidpy.experimental.methods.Aligner` to run.

    See Also
    --------
    align_stalign : Build and run a :class:`~squidpy.experimental.methods.StalignAligner` in one call.
    """
    return _run_align_samples(
        aligner,
        data_ref,
        data_query,
        on=on,
        ref_key=ref_key,
        query_key=query_key,
        spatial_key=spatial_key,
        output_mode=output_mode,
        key_added=key_added,
    )


def _run_align_samples(
    aligner: Aligner,
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None,
    *,
    on: str,
    ref_key: str | None,
    query_key: str | None,
    spatial_key: str,
    output_mode: str,
    key_added: str | None,
) -> AlignResult | AnnData | SpatialData | None:
    """Shared core: resolve the obs point clouds, run the aligner, write back."""
    assert_one_of(output_mode, OUTPUT_MODES, name="output_mode")
    assert_one_of(on, ON_VALUES, name="on")
    if on == "image":
        raise NotImplementedError("`align(on='image')` is not implemented yet; use `on='obs'`.")

    ref_adata, query_adata, container, element_key = resolve_obs_pair(data_ref, data_query, ref_key, query_key)
    ref_xy = get_coords(ref_adata, spatial_key)
    query_xy = get_coords(query_adata, spatial_key)

    result = aligner.align(ref_xy, query_xy)

    return writeback_obs(
        result,
        output_mode=output_mode,
        query_adata=query_adata,
        container=container,
        element_key=element_key,
        spatial_key=spatial_key,
        key_added=key_added,
    )


# ---------------------------------------------------------------------------
# Landmark alignment (closed-form): align_landmarks family
# ---------------------------------------------------------------------------


def align_landmarks_similarity(
    ref: np.ndarray | Sequence[tuple[float, float]],
    query: np.ndarray | Sequence[tuple[float, float]],
    *,
    data: AnnData | SpatialData | None = None,
    cs_name_ref: str | None = None,
    cs_name_query: str | None = None,
    spatial_key: str = "spatial",
    output_mode: Literal["object", "copy", "inplace"] = "object",
    key_added: str | None = None,
) -> AlignResult | AnnData | SpatialData | None:
    """Align by a closed-form 4-DOF similarity fit on pre-paired landmarks.

    Rotation + uniform scale + translation. Thin wrapper over
    :class:`~squidpy.experimental.methods.SimilarityAligner`.

    Parameters
    ----------
    ref, query, data, cs_name_ref, cs_name_query, spatial_key, output_mode, key_added
        See :func:`align_landmarks_affine`.

    See Also
    --------
    align_landmarks_affine : 6-DOF affine variant.
    align_landmarks_from_aligner : Use a pre-built (or custom) landmark aligner.
    """
    return _run_align_landmarks(
        SimilarityAligner(source_cs=cs_name_query, target_cs=cs_name_ref),
        ref,
        query,
        data=data,
        cs_name_ref=cs_name_ref,
        cs_name_query=cs_name_query,
        spatial_key=spatial_key,
        output_mode=output_mode,
        key_added=key_added,
    )


def align_landmarks_affine(
    ref: np.ndarray | Sequence[tuple[float, float]],
    query: np.ndarray | Sequence[tuple[float, float]],
    *,
    data: AnnData | SpatialData | None = None,
    cs_name_ref: str | None = None,
    cs_name_query: str | None = None,
    spatial_key: str = "spatial",
    output_mode: Literal["object", "copy", "inplace"] = "object",
    key_added: str | None = None,
) -> AlignResult | AnnData | SpatialData | None:
    """Align by a closed-form 6-DOF affine fit on pre-paired landmarks.

    Rotation + non-uniform scale + shear + translation. Thin wrapper over
    :class:`~squidpy.experimental.methods.AffineAligner`.

    Parameters
    ----------
    ref, query
        Equal-length ``(N, 2)`` ``(x, y)`` landmark arrays (``N >= 3``), paired by
        row order. No automatic correspondence matching is performed.
    data
        Target to write the alignment into. Required for ``output_mode`` other
        than ``"object"``.
    cs_name_ref, cs_name_query
        Coordinate-system names. For a SpatialData ``data`` the fitted affine is
        registered on every element in ``cs_name_query``, mapping into ``cs_name_ref``.
    spatial_key
        ``obsm`` key when ``data`` is an :class:`~anndata.AnnData`.
    output_mode
        See :func:`align_stalign`. ``"object"`` (default) returns the fitted
        :class:`~squidpy.experimental.tl.AlignResult`.
    key_added
        Destination ``obsm`` key when ``data`` is an AnnData (see :func:`align_stalign`).

    See Also
    --------
    align_landmarks_similarity : 4-DOF similarity variant.
    align_landmarks_from_aligner : Use a pre-built (or custom) landmark aligner.
    """
    return _run_align_landmarks(
        AffineAligner(source_cs=cs_name_query, target_cs=cs_name_ref),
        ref,
        query,
        data=data,
        cs_name_ref=cs_name_ref,
        cs_name_query=cs_name_query,
        spatial_key=spatial_key,
        output_mode=output_mode,
        key_added=key_added,
    )


def align_landmarks_from_aligner(
    ref: np.ndarray | Sequence[tuple[float, float]],
    query: np.ndarray | Sequence[tuple[float, float]],
    aligner: Aligner,
    *,
    data: AnnData | SpatialData | None = None,
    cs_name_ref: str | None = None,
    cs_name_query: str | None = None,
    spatial_key: str = "spatial",
    output_mode: Literal["object", "copy", "inplace"] = "object",
    key_added: str | None = None,
) -> AlignResult | AnnData | SpatialData | None:
    """Align by landmarks using an explicit aligner instance.

    The escape hatch for landmark alignment. The aligner must return an
    :class:`~squidpy.experimental.methods.AffineFitResult` (the SpatialData
    write-back registers its affine on a coordinate system).

    Parameters
    ----------
    ref, query, data, cs_name_ref, cs_name_query, spatial_key, output_mode, key_added
        See :func:`align_landmarks_affine`.
    aligner
        The :class:`~squidpy.experimental.methods.Aligner` to run.
    """
    return _run_align_landmarks(
        aligner,
        ref,
        query,
        data=data,
        cs_name_ref=cs_name_ref,
        cs_name_query=cs_name_query,
        spatial_key=spatial_key,
        output_mode=output_mode,
        key_added=key_added,
    )


def _run_align_landmarks(
    aligner: Aligner,
    ref: np.ndarray | Sequence[tuple[float, float]],
    query: np.ndarray | Sequence[tuple[float, float]],
    *,
    data: AnnData | SpatialData | None,
    cs_name_ref: str | None,
    cs_name_query: str | None,
    spatial_key: str,
    output_mode: str,
    key_added: str | None,
) -> AlignResult | AnnData | SpatialData | None:
    """Shared core: run the landmark aligner, then write the affine back per ``output_mode``."""
    assert_one_of(output_mode, OUTPUT_MODES, name="output_mode")

    result = aligner.align(ref, query)

    if output_mode == "object":
        return result
    if data is None:
        raise ValueError("`data` is required when `output_mode` is 'copy' or 'inplace'.")

    if isinstance(data, SpatialData):
        return writeback_affine_sdata(
            cast("AffineFitResult", result),
            data,
            output_mode=output_mode,
            moving_cs=cs_name_query,
            target_cs=cs_name_ref,
        )
    if isinstance(data, AnnData):
        return writeback_obs(
            result,
            output_mode=output_mode,
            query_adata=data,
            container=None,
            element_key=None,
            spatial_key=spatial_key,
            key_added=key_added,
        )
    raise TypeError(f"`data` must be AnnData or SpatialData, got {type(data).__name__}.")
