"""Developer API for spatial graph construction.

Provides typed builder classes that expose only the relevant parameters
for each graph construction strategy, as an alternative to the monolithic
:func:`squidpy.gr.spatial_neighbors` function.

Each builder encapsulates a single graph construction method and is
configured at instantiation time.  Calling :meth:`~GraphBuilder.build`
with a coordinate array returns ``(adjacency, distances)`` sparse matrices.

Quick reference
---------------

==============  =========================================================
Builder         When to use
==============  =========================================================
``KNN``         Fixed number of nearest neighbors (default ``n_neighs=6``)
``Delaunay``    Delaunay triangulation of the point set
``Radius``      All neighbors within a fixed radius
``Grid``        Regular grid / Visually tiled data with ring expansion
==============  =========================================================

Examples
--------
>>> import numpy as np
>>> from squidpy.gr.neighbors import KNN, Delaunay
>>>
>>> coords = np.random.default_rng(0).random((100, 2))
>>>
>>> adj, dst = KNN(n_neighs=10).build(coords)
>>> adj, dst = Delaunay().build(coords)
"""

from squidpy._constants._constants import Transform
from squidpy.gr.neighbors._builders import (
    DelaunayBuilder,
    GraphBuilder,
    GridBuilder,
    KNNBuilder,
    RadiusBuilder,
)

KNN = KNNBuilder
Delaunay = DelaunayBuilder
Grid = GridBuilder
Radius = RadiusBuilder

__all__ = [
    "GraphBuilder",
    "KNN",
    "KNNBuilder",
    "Delaunay",
    "DelaunayBuilder",
    "Grid",
    "GridBuilder",
    "Radius",
    "RadiusBuilder",
    "Transform",
]
