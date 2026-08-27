# Experimental Types

Every parameter mapping and result container of {mod}`squidpy.experimental`, on one page.

The `*Params` types are {class}`~typing.TypedDict` declarations with `total=False`: pass any
subset of the keys as a plain `dict`, and the absent ones fall back to the defaults listed
below.

```{eval-rst}
.. currentmodule:: squidpy.experimental

.. autosummary::

    types.BackgroundDetectionParams
    types.FelzenszwalbParams
    types.WekaParams
    types.ReinhardParams
    types.MacenkoParams
    types.VahadaneParams
    types.TilingQCParams
    types.StitchParams
    im.StainReference
```

## Tissue detection

```{eval-rst}
.. autoclass:: squidpy.experimental.types.BackgroundDetectionParams
    :members:
    :undoc-members:

.. autoclass:: squidpy.experimental.types.FelzenszwalbParams
    :members:
    :undoc-members:

.. autoclass:: squidpy.experimental.types.WekaParams
    :members:
    :undoc-members:
```

## Stain normalization

```{eval-rst}
.. autoclass:: squidpy.experimental.types.ReinhardParams
    :members:
    :undoc-members:

.. autoclass:: squidpy.experimental.types.MacenkoParams
    :members:
    :undoc-members:

.. autoclass:: squidpy.experimental.types.VahadaneParams
    :members:
    :undoc-members:

.. autoclass:: squidpy.experimental.im.StainReference
    :members:
    :undoc-members:
```

## Tiling

```{eval-rst}
.. autoclass:: squidpy.experimental.types.TilingQCParams
    :members:
    :undoc-members:

.. autoclass:: squidpy.experimental.types.StitchParams
    :members:
    :undoc-members:
```
