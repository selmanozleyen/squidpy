"""The graph module."""

from __future__ import annotations

from squidpy.gr._build import mask_graph, spatial_neighbors
from squidpy.gr._ligrec import ligrec
from squidpy.gr._nhood import centrality_scores, interaction_matrix, nhood_enrichment
from squidpy.gr._niche import calculate_niche
from squidpy.gr._ppatterns import co_occurrence, spatial_autocorr
from squidpy.gr._ripley import ripley
from squidpy.gr._sepal import sepal
