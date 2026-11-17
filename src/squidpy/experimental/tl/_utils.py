"""Resolve the reference and query containers of the experimental tools that take both."""

from __future__ import annotations

from collections.abc import Sequence

from anndata import AnnData
from spatialdata import SpatialData

from squidpy._validators import assert_non_empty_sequence

__all__ = ["resolve_ref_query"]


def resolve_ref_query(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None,
    *,
    genes: str | Sequence[str],
    table_key: str | Sequence[str | None] | None,
    layer: str | Sequence[str | None] | None,
) -> tuple[AnnData, AnnData, list[str], list[str], str | None, str | None]:
    """Resolve the reference and query tables of an imputation, split its layers, and validate ``genes``.

    Return the reference, the query, the genes to impute, the genes they share, and their layers.
    """
    ref_table, query_table = resolve_pair(table_key, name="table_key")
    ref_layer, query_layer = resolve_pair(layer, name="layer")
    data_query = query_of(
        data_ref, data_query, ref_address=(ref_table,), query_address=(query_table,), key_name="table_key"
    )
    ref = resolve_table(data_ref, ref_table, side="reference")
    query = resolve_table(data_query, query_table, side="query")
    for side, adata in (("reference", ref), ("query", query)):
        if not adata.var_names.is_unique:
            raise ValueError(f"The {side} has duplicate gene names; make them unique with `var_names_make_unique`.")

    genes = assert_non_empty_sequence(genes, name="genes")
    if missing := [g for g in genes if g not in ref.var_names]:
        raise KeyError(f"Genes not found in the reference: {missing}.")
    shared = query.var_names.intersection(ref.var_names, sort=False).tolist()
    if not shared:
        raise ValueError("The reference and the query have no genes in common.")
    return ref, query, genes, shared, ref_layer, query_layer


def resolve_pair(value: str | Sequence[str | None] | None, *, name: str) -> tuple[str | None, str | None]:
    """Normalize a single key, applying to both sides, or a ``(ref, query)`` pair to a pair."""
    if value is None or isinstance(value, str):
        return value, value
    if isinstance(value, tuple | list) and len(value) == 2:
        return value[0], value[1]
    raise ValueError(f"`{name}` must be a single key or a `(ref, query)` pair, got {value!r}.")


def query_of(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None,
    *,
    ref_address: tuple[str | None, ...],
    query_address: tuple[str | None, ...],
    key_name: str,
) -> AnnData | SpatialData:
    """The container holding the query sample, validating the single-SpatialData form."""
    if data_query is not None:
        return data_query
    if not isinstance(data_ref, SpatialData):
        raise ValueError("`data_query` is required unless `data_ref` is a SpatialData holding both samples.")
    if ref_address == query_address:
        raise ValueError(
            f"With a single SpatialData, pass `{key_name}=(ref, query)` to say which of its elements holds each sample."
        )
    return data_ref


def resolve_table(container: AnnData | SpatialData, table_key: str | None, *, side: str) -> AnnData:
    """Resolve a container plus ``table_key`` to the AnnData holding the data."""
    if isinstance(container, AnnData):
        if table_key is not None:
            raise ValueError(
                f"`table_key` was given for the {side}, but the {side} is an AnnData, which has no tables. "
                f"Pass `table_key=None` for it (use a `(ref, query)` pair to address mixed containers)."
            )
        return container
    if not isinstance(container, SpatialData):
        raise TypeError(f"Expected the {side} to be an AnnData or SpatialData, got {type(container).__name__}.")
    if table_key is None:
        raise ValueError(
            f"The {side} is a SpatialData, which may hold several tables; pass `table_key` to say "
            f"which one holds the data. Available: {sorted(container.tables)}."
        )
    if table_key not in container.tables:
        raise KeyError(
            f"`table_key={table_key!r}`: no such table in the {side}. Available: {sorted(container.tables)}."
        )
    return container.tables[table_key]
