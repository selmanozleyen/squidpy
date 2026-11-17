from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from squidpy.experimental.tl import label_components, neighbourhood_distances


def _adata(coords, labels, library=None) -> AnnData:
    obs = pd.DataFrame({"pert": pd.Categorical(labels)}, index=[f"c{i}" for i in range(len(labels))])
    if library is not None:
        obs["lib"] = pd.Categorical(library)
    return AnnData(obs=obs, obsm={"spatial": np.asarray(coords, dtype=float)})


def test_label_components_splits_by_distance_and_label():
    # two A blobs far apart, one B blob next to the first A blob, one unlabelled cell
    coords = [[0, 0], [1, 0], [2, 0], [100, 0], [101, 0], [3, 0], [4, 0], [50, 50]]
    labels = ["A", "A", "A", "A", "A", "B", "B", None]
    adata = _adata(coords, labels)
    comp = label_components(adata, "pert", radius=1.5, copy=True)
    assert comp.iloc[:3].nunique() == 1
    assert comp.iloc[3:5].nunique() == 1
    assert comp.iloc[0] != comp.iloc[3]  # same label, not connected
    assert comp.iloc[5] != comp.iloc[2]  # connected in space, different label
    assert pd.isna(comp.iloc[7])
    assert comp.iloc[0].startswith("A:")


def test_label_components_min_cells_and_library():
    coords = [[0, 0], [1, 0], [0, 0], [1, 0], [10, 10]]
    labels = ["A", "A", "A", "A", "A"]
    adata = _adata(coords, labels, library=["s1", "s1", "s2", "s2", "s2"])
    label_components(adata, "pert", radius=1.5, min_cells=2, library_key="lib")
    comp = adata.obs["pert_component"]
    assert comp.iloc[0] == comp.iloc[1]
    assert comp.iloc[2] == comp.iloc[3]
    assert comp.iloc[0] != comp.iloc[2]  # identical coordinates, different sections
    assert pd.isna(comp.iloc[4])  # singleton dropped by min_cells


def test_neighbourhood_distances_matches_brute_force():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(300, 2))
    labels = np.where(rng.uniform(size=300) < 0.2, rng.choice(["A", "B"], size=300), None)
    adata = _adata(coords, labels)
    label_components(adata, "pert", radius=8.0)
    res = neighbourhood_distances(adata, "pert_component", max_distance=15.0)
    comp = adata.obs["pert_component"]
    free = np.flatnonzero(comp.isna().to_numpy())
    expected = {}
    for c in comp.cat.categories:
        src = coords[(comp == c).to_numpy()]
        d = np.sqrt(((coords[free][:, None, :] - src[None, :, :]) ** 2).sum(-1)).min(1)
        for i, di in zip(free, d, strict=True):
            if 0 < di <= 15.0:
                expected[(c, adata.obs_names[i])] = di
    got = {(c, o): d for c, o, d in res.itertuples(index=False)}
    assert got.keys() == expected.keys()
    np.testing.assert_allclose([got[k] for k in expected], [expected[k] for k in expected])


def test_neighbourhood_distances_respects_library():
    coords = [[0, 0], [1, 0], [0, 0], [1, 0]]
    adata = _adata(coords, ["A", None, None, None], library=["s1", "s1", "s2", "s2"])
    label_components(adata, "pert", radius=1.0, library_key="lib")
    res = neighbourhood_distances(adata, "pert_component", max_distance=5.0, library_key="lib")
    assert set(res["obs_name"]) == {"c1"}  # c2/c3 are in another section


@pytest.mark.parametrize("bad", [0, -1])
def test_validation(bad):
    adata = _adata([[0, 0], [1, 0]], ["A", "A"])
    with pytest.raises(ValueError, match="radius"):
        label_components(adata, "pert", radius=bad)
    label_components(adata, "pert", radius=2)
    with pytest.raises(ValueError, match="max_distance"):
        neighbourhood_distances(adata, "pert_component", max_distance=bad)
