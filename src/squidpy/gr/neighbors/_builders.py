"""Strategy-pattern graph builders for spatial neighbor computation."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from itertools import chain

import numpy as np
from fast_array_utils import stats as fau_stats
from numba import njit, prange
from scipy.sparse import (
    SparseEfficiencyWarning,
    csr_array,
    csr_matrix,
    isspmatrix_csr,
    spmatrix,
)
from scipy.spatial import Delaunay as _Delaunay
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors

from squidpy._constants._constants import Transform
from squidpy._utils import NDArrayA

__all__ = [
    "GraphBuilder",
    "KNNBuilder",
    "DelaunayBuilder",
    "RadiusBuilder",
    "GridBuilder",
]


@njit
def _csr_bilateral_diag_scale_helper(
    mat: csr_array | csr_matrix,
    degrees: NDArrayA,
) -> NDArrayA:
    """D_i * data_k * D_j for each non-zero k at position (i, j) in CSR order."""
    res = np.empty_like(mat.data, dtype=np.float32)
    for i in prange(len(mat.indptr) - 1):
        ixs = mat.indices[mat.indptr[i] : mat.indptr[i + 1]]
        res[mat.indptr[i] : mat.indptr[i + 1]] = degrees[i] * degrees[ixs] * mat.data[mat.indptr[i] : mat.indptr[i + 1]]
    return res


def _symmetric_normalize_csr(adj: spmatrix) -> csr_matrix:
    """Return D^{-1/2} * A * D^{-1/2} where D = diag(degrees(A))."""
    degrees = np.squeeze(np.array(np.sqrt(1.0 / fau_stats.sum(adj, axis=0))))
    if adj.shape[0] != len(degrees):
        raise ValueError("len(degrees) must equal number of rows of adj")
    res_data = _csr_bilateral_diag_scale_helper(adj, degrees)
    return csr_matrix((res_data, adj.indices, adj.indptr), shape=adj.shape)


def _transform_a_spectral(a: spmatrix) -> spmatrix:
    if not isspmatrix_csr(a):
        a = a.tocsr()
    if not a.nnz:
        return a
    return _symmetric_normalize_csr(a)


def _transform_a_cosine(a: spmatrix) -> spmatrix:
    return cosine_similarity(a, dense_output=False)


class GraphBuilder(ABC):
    """Base class for spatial graph construction strategies.

    Subclasses implement :meth:`_build_graph` to produce raw adjacency and
    distance matrices.  Post-processing (percentile filtering, radius-tuple
    pruning, transforms) is handled by :meth:`build`.

    Parameters
    ----------
    transform
        Adjacency matrix transform to apply after graph construction.
        Use ``Transform.SPECTRAL`` or ``Transform.COSINE``, or leave as
        ``Transform.NONE`` (default).
    set_diag
        Whether to set the diagonal of the adjacency matrix to ``1.0``.
    percentile
        If not ``None``, prune edges whose distance exceeds this percentile.

    See Also
    --------
    squidpy.gr.neighbors.KNN : k-nearest neighbors builder.
    squidpy.gr.neighbors.Delaunay : Delaunay triangulation builder.
    squidpy.gr.neighbors.Radius : radius-based builder.
    squidpy.gr.neighbors.Grid : grid ring-expansion builder.
    """

    def __init__(
        self,
        transform: Transform = Transform.NONE,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        self.transform = transform
        self.set_diag = set_diag
        self.percentile = percentile

    def build(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        """Build spatial graph from coordinates.

        Parameters
        ----------
        coords
            Array of shape ``(n_points, n_dims)`` with spatial coordinates.

        Returns
        -------
        Tuple of ``(adjacency, distances)`` sparse matrices.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseEfficiencyWarning)
            Adj, Dst = self._build_graph(coords)

        self._apply_percentile(Adj, Dst)

        Adj.eliminate_zeros()
        Dst.eliminate_zeros()

        Adj = self._apply_transform(Adj)
        return Adj, Dst

    @abstractmethod
    def _build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        """Construct raw adjacency and distance matrices.

        Must be implemented by subclasses.
        """

    def _apply_percentile(self, Adj: csr_matrix, Dst: csr_matrix) -> None:
        if self.percentile is not None:
            threshold = np.percentile(Dst.data, self.percentile)
            Adj[Dst > threshold] = 0.0
            Dst[Dst > threshold] = 0.0

    def _apply_transform(self, Adj: csr_matrix) -> csr_matrix:
        if self.transform == Transform.SPECTRAL:
            return _transform_a_spectral(Adj)
        if self.transform == Transform.COSINE:
            return _transform_a_cosine(Adj)
        if self.transform == Transform.NONE:
            return Adj
        raise NotImplementedError(f"Transform `{self.transform}` is not yet implemented.")


class KNNBuilder(GraphBuilder):
    """Build a graph using k-nearest neighbors.

    Parameters
    ----------
    n_neighs
        Number of nearest neighbors.
    radius_bounds
        If a ``(min, max)`` tuple, prune edges outside this distance interval
        after building the k-NN graph.
    transform
        Adjacency matrix transform.
    set_diag
        Whether to set the diagonal to ``1.0``.
    percentile
        Percentile distance threshold for pruning.

    Examples
    --------
    >>> import numpy as np
    >>> from squidpy.gr.neighbors import KNN
    >>> coords = np.random.default_rng(0).random((100, 2))
    >>> adj, dst = KNN(n_neighs=10).build(coords)
    >>> adj.shape
    (100, 100)
    """

    def __init__(
        self,
        n_neighs: int = 6,
        radius_bounds: tuple[float, float] | None = None,
        transform: Transform = Transform.NONE,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        super().__init__(transform=transform, set_diag=set_diag, percentile=percentile)
        self.n_neighs = n_neighs
        self.radius_bounds = radius_bounds

    def _build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        N = coords.shape[0]
        tree = NearestNeighbors(n_neighbors=self.n_neighs, metric="euclidean")
        tree.fit(coords)

        dists, col_indices = tree.kneighbors()
        dists, col_indices = dists.reshape(-1), col_indices.reshape(-1)
        row_indices = np.repeat(np.arange(N), self.n_neighs)

        Adj = csr_matrix(
            (np.ones_like(row_indices, dtype=np.float32), (row_indices, col_indices)),
            shape=(N, N),
        )
        Dst = csr_matrix((dists, (row_indices, col_indices)), shape=(N, N))

        Adj.setdiag(1.0 if self.set_diag else Adj.diagonal())
        Dst.setdiag(0.0)

        if self.radius_bounds is not None:
            self._apply_radius_bounds(Adj, Dst)

        return Adj, Dst

    def _apply_radius_bounds(self, Adj: csr_matrix, Dst: csr_matrix) -> None:
        minn, maxx = sorted(self.radius_bounds)[:2]
        mask = (Dst.data < minn) | (Dst.data > maxx)
        a_diag = Adj.diagonal()
        Dst.data[mask] = 0.0
        Adj.data[mask] = 0.0
        Adj.setdiag(a_diag)


class DelaunayBuilder(GraphBuilder):
    """Build a graph from Delaunay triangulation.

    Parameters
    ----------
    radius_bounds
        If a ``(min, max)`` tuple, prune edges outside this distance interval
        after triangulation.
    transform
        Adjacency matrix transform.
    set_diag
        Whether to set the diagonal to ``1.0``.
    percentile
        Percentile distance threshold for pruning.

    Examples
    --------
    >>> import numpy as np
    >>> from squidpy.gr.neighbors import Delaunay
    >>> coords = np.random.default_rng(0).random((50, 2))
    >>> adj, dst = Delaunay().build(coords)
    >>> adj.shape
    (50, 50)
    """

    def __init__(
        self,
        radius_bounds: tuple[float, float] | None = None,
        transform: Transform = Transform.NONE,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        super().__init__(transform=transform, set_diag=set_diag, percentile=percentile)
        self.radius_bounds = radius_bounds

    def _build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        N = coords.shape[0]
        tri = _Delaunay(coords)
        indptr, indices = tri.vertex_neighbor_vertices
        Adj = csr_matrix(
            (np.ones_like(indices, dtype=np.float32), indices, indptr), shape=(N, N)
        )

        # fmt: off
        dists = np.array(list(chain(*(
            euclidean_distances(
                coords[indices[indptr[i] : indptr[i + 1]], :],
                coords[np.newaxis, i, :],
            )
            for i in range(N)
            if len(indices[indptr[i] : indptr[i + 1]])
        )))).squeeze()
        # fmt: on
        Dst = csr_matrix((dists, indices, indptr), shape=(N, N))

        Adj.setdiag(1.0 if self.set_diag else Adj.diagonal())
        Dst.setdiag(0.0)

        if self.radius_bounds is not None:
            minn, maxx = sorted(self.radius_bounds)[:2]
            mask = (Dst.data < minn) | (Dst.data > maxx)
            a_diag = Adj.diagonal()
            Dst.data[mask] = 0.0
            Adj.data[mask] = 0.0
            Adj.setdiag(a_diag)

        return Adj, Dst


class RadiusBuilder(GraphBuilder):
    """Build a graph using radius-based neighbors.

    Parameters
    ----------
    radius
        Neighborhood radius.
    n_neighs
        Number of neighbors (used to initialize the tree; all points within
        ``radius`` are returned).
    transform
        Adjacency matrix transform.
    set_diag
        Whether to set the diagonal to ``1.0``.
    percentile
        Percentile distance threshold for pruning.

    Examples
    --------
    >>> import numpy as np
    >>> from squidpy.gr.neighbors import Radius
    >>> coords = np.random.default_rng(0).random((100, 2))
    >>> adj, dst = Radius(radius=0.3).build(coords)
    >>> adj.shape
    (100, 100)
    """

    def __init__(
        self,
        radius: float = 1.0,
        n_neighs: int = 6,
        transform: Transform = Transform.NONE,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        super().__init__(transform=transform, set_diag=set_diag, percentile=percentile)
        self.radius = radius
        self.n_neighs = n_neighs

    def _build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        N = coords.shape[0]
        tree = NearestNeighbors(
            n_neighbors=self.n_neighs, radius=self.radius, metric="euclidean"
        )
        tree.fit(coords)

        dists, col_indices = tree.radius_neighbors()
        row_indices = np.repeat(np.arange(N), [len(x) for x in col_indices])
        dists = np.concatenate(dists)
        col_indices = np.concatenate(col_indices)

        Adj = csr_matrix(
            (np.ones_like(row_indices, dtype=np.float32), (row_indices, col_indices)),
            shape=(N, N),
        )
        Dst = csr_matrix((dists, (row_indices, col_indices)), shape=(N, N))

        Adj.setdiag(1.0 if self.set_diag else Adj.diagonal())
        Dst.setdiag(0.0)

        return Adj, Dst


class GridBuilder(GraphBuilder):
    """Build a graph on grid coordinates with ring expansion.

    Parameters
    ----------
    n_neighs
        Number of neighboring tiles in the base grid.
    n_rings
        Number of rings of neighbors to include.
    delaunay
        Whether to use Delaunay triangulation for the base grid connectivity.
    transform
        Adjacency matrix transform.
    set_diag
        Whether to set the diagonal to ``1.0``.
    percentile
        Percentile distance threshold for pruning.

    Examples
    --------
    >>> import numpy as np
    >>> from squidpy.gr.neighbors import Grid
    >>> coords = np.column_stack([
    ...     np.repeat(np.arange(10), 10),
    ...     np.tile(np.arange(10), 10),
    ... ]).astype(float)
    >>> adj, dst = Grid(n_neighs=6, n_rings=2).build(coords)
    >>> adj.shape
    (100, 100)
    """

    def __init__(
        self,
        n_neighs: int = 6,
        n_rings: int = 1,
        delaunay: bool = False,
        transform: Transform = Transform.NONE,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        super().__init__(transform=transform, set_diag=set_diag, percentile=percentile)
        self.n_neighs = n_neighs
        self.n_rings = n_rings
        self.delaunay = delaunay

    def _build_graph(self, coords: NDArrayA) -> tuple[csr_matrix, csr_matrix]:
        Adj = self._base_connectivity(coords)

        if self.n_rings > 1:
            Adj = self._expand_rings(Adj)
            Dst = Adj.copy()
            Adj.data[:] = 1.0
        else:
            Dst = Adj.copy()

        Dst.setdiag(0.0)
        return Adj, Dst

    def _base_connectivity(self, coords: NDArrayA) -> csr_matrix:
        """Build single-ring grid connectivity with median-distance correction."""
        N = coords.shape[0]
        if self.delaunay:
            tri = _Delaunay(coords)
            indptr, indices = tri.vertex_neighbor_vertices
            Adj = csr_matrix(
                (np.ones_like(indices, dtype=np.float32), indices, indptr),
                shape=(N, N),
            )
        else:
            tree = NearestNeighbors(n_neighbors=self.n_neighs, metric="euclidean")
            tree.fit(coords)
            dists, col_indices = tree.kneighbors()
            dists, col_indices = dists.reshape(-1), col_indices.reshape(-1)
            row_indices = np.repeat(np.arange(N), self.n_neighs)

            dist_cutoff = np.median(dists) * 1.3
            mask = dists < dist_cutoff
            row_indices, col_indices, dists = (
                row_indices[mask],
                col_indices[mask],
                dists[mask],
            )
            Adj = csr_matrix(
                (np.ones_like(row_indices, dtype=np.float32), (row_indices, col_indices)),
                shape=(N, N),
            )

        set_diag_val = True if self.n_rings > 1 else self.set_diag
        Adj.setdiag(1.0 if set_diag_val else Adj.diagonal())
        return Adj

    def _expand_rings(self, Adj: csr_matrix) -> csr_matrix:
        """Expand adjacency to include higher-order ring neighbors."""
        Res, Walk = Adj, Adj
        for i in range(self.n_rings - 1):
            Walk = Walk @ Adj
            Walk[Res.nonzero()] = 0.0
            Walk.eliminate_zeros()
            Walk.data[:] = i + 2.0
            Res = Res + Walk
        Res.setdiag(float(self.set_diag))
        Res.eliminate_zeros()
        return Res
