from __future__ import annotations

import pytest

from squidpy._utils import memory_budget


@pytest.mark.parametrize(
    ("max_mem", "expected"),
    [
        (1000, 1000),
        ("1000", 1000),
        ("16GB", 16 * 10**9),
        ("512MiB", 512 * 2**20),
        ("1.5TB", 15 * 10**11),
        (" 2 gib ", 2 * 2**30),
    ],
)
def test_memory_budget_parses(max_mem, expected):
    assert memory_budget(max_mem, floor=1) == expected


@pytest.mark.parametrize("max_mem", ["16 gigs", "GB", "-1GB", "0", 0, -5])
def test_memory_budget_rejects(max_mem):
    with pytest.raises(ValueError, match="max_mem"):
        memory_budget(max_mem, floor=1)


def test_memory_budget_default_is_at_least_the_floor():
    assert memory_budget(None, floor=123) >= 123
