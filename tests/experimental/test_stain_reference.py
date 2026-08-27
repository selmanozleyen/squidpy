from __future__ import annotations

import numpy as np
import pytest

from squidpy.experimental.im._stain._constants import RUIFROK_HE
from squidpy.experimental.im._stain._reference import validate_stain_reference
from squidpy.experimental.types import StainReference

# Tests construct stain matrices and background estimates by hand; there is
# no library-wide pure-white default to lean on.
_TEST_BACKGROUND = np.array([245.0, 250.0, 240.0])


def _ref(**fields: object) -> StainReference:
    """Build and validate a reference the way the fit functions do."""
    return validate_stain_reference(fields)  # type: ignore[arg-type]


def _ruifrok_matrix() -> np.ndarray:
    third = np.cross(RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"])
    third /= np.linalg.norm(third)
    return np.column_stack([RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"], third])


def test_macenko_basic() -> None:
    ref = _ref(
        method="macenko",
        stain_matrix=_ruifrok_matrix(),
        white_point=_TEST_BACKGROUND,
    )
    assert ref["method"] == "macenko"
    assert ref["stain_matrix"].shape == (3, 3)
    assert ref["mu"] is None and ref["sigma"] is None
    np.testing.assert_array_equal(ref["white_point"], _TEST_BACKGROUND)


def test_reinhard_basic() -> None:
    ref = _ref(method="reinhard", mu=np.array([1.0, 0.5, -0.2]), sigma=np.array([0.1, 0.1, 0.1]))
    assert ref["method"] == "reinhard"
    assert ref["stain_matrix"] is None
    assert ref["white_point"] is None


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        _ref(method="not-a-method")  # type: ignore[arg-type]


def test_decomposition_requires_stain_matrix() -> None:
    with pytest.raises(ValueError, match="requires stain_matrix"):
        _ref(method="macenko", white_point=_TEST_BACKGROUND)


def test_decomposition_requires_white_point() -> None:
    with pytest.raises(ValueError, match="requires white_point"):
        _ref(method="macenko", stain_matrix=_ruifrok_matrix())


def test_decomposition_forbids_mu_sigma() -> None:
    with pytest.raises(ValueError, match="forbids mu/sigma"):
        _ref(
            method="macenko",
            stain_matrix=_ruifrok_matrix(),
            white_point=_TEST_BACKGROUND,
            mu=np.zeros(3),
            sigma=np.ones(3),
        )


def test_reinhard_requires_mu_and_sigma() -> None:
    with pytest.raises(ValueError, match="requires both mu and sigma"):
        _ref(method="reinhard", mu=np.zeros(3))


def test_reinhard_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        _ref(method="reinhard", mu=np.zeros(3), sigma=np.array([1.0, 0.0, 1.0]))


def test_reinhard_forbids_stain_matrix() -> None:
    with pytest.raises(ValueError, match="forbids stain_matrix"):
        _ref(
            method="reinhard",
            mu=np.zeros(3),
            sigma=np.ones(3),
            stain_matrix=_ruifrok_matrix(),
        )


def test_reinhard_forbids_white_point() -> None:
    with pytest.raises(ValueError, match="forbids white_point"):
        _ref(
            method="reinhard",
            mu=np.zeros(3),
            sigma=np.ones(3),
            white_point=_TEST_BACKGROUND,
        )


def test_bad_white_point() -> None:
    with pytest.raises(ValueError, match="white_point"):
        _ref(
            method="macenko",
            stain_matrix=_ruifrok_matrix(),
            white_point=np.array([255.0, -1.0, 255.0]),
        )


def test_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"stain_matrix must have shape"):
        _ref(
            method="macenko",
            stain_matrix=np.zeros((2, 3)),
            white_point=_TEST_BACKGROUND,
        )


def test_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match=r"mu contains non-finite values"):
        _ref(
            method="reinhard",
            mu=np.array([np.nan, 0.0, 0.0]),
            sigma=np.ones(3),
        )


def test_validated_copy_is_normalised() -> None:
    # every key present (None where the method does not use it), arrays coerced to
    # float64, and the caller's mapping left untouched
    raw = {"method": "reinhard", "mu": [1, 2, 3], "sigma": [1, 1, 1]}
    ref = validate_stain_reference(raw)  # type: ignore[arg-type]
    assert set(ref) == set(StainReference.__annotations__)
    assert ref["stain_matrix"] is None and ref["white_point"] is None
    assert ref["mu"].dtype == np.float64
    assert raw["mu"] == [1, 2, 3]


def test_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="Unknown reference key"):
        _ref(method="reinhard", mu=np.zeros(3), sigma=np.ones(3), bogus=1)
