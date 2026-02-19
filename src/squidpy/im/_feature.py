from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pandas as pd
from anndata import AnnData
from scanpy import logging as logg
from tqdm.auto import tqdm

from squidpy._constants._constants import ImageFeature
from squidpy._docs import d, inject_docs
from squidpy.gr._utils import _save_data
from squidpy.im._container import ImageContainer

__all__ = ["calculate_image_features"]


@d.dedent
@inject_docs(f=ImageFeature)
def calculate_image_features(
    adata: AnnData,
    img: ImageContainer,
    layer: str | None = None,
    library_id: str | Sequence[str] | None = None,
    features: str | Sequence[str] = ImageFeature.SUMMARY.s,
    features_kwargs: Mapping[str, Mapping[str, Any]] = MappingProxyType({}),
    key_added: str = "img_features",
    copy: bool = False,
    show_progress_bar: bool = True,
    **kwargs: Any,
) -> pd.DataFrame | None:
    """
    Calculate image features for all observations in ``adata``.

    Parameters
    ----------
    %(adata)s
    %(img_container)s
    %(img_layer)s
    %(img_library_id)s
    features
        Features to be calculated. Valid options are:

        - `{f.TEXTURE.s!r}` - summary stats based on repeating patterns
          :meth:`squidpy.im.ImageContainer.features_texture`.
        - `{f.SUMMARY.s!r}` - summary stats of each image channel
          :meth:`squidpy.im.ImageContainer.features_summary`.
        - `{f.COLOR_HIST.s!r}` - counts in bins of image channel's histogram
          :meth:`squidpy.im.ImageContainer.features_histogram`.
        - `{f.SEGMENTATION.s!r}` - stats of a cell segmentation mask
          :meth:`squidpy.im.ImageContainer.features_segmentation`.
        - `{f.CUSTOM.s!r}` - extract features using a custom function
          :meth:`squidpy.im.ImageContainer.features_custom`.

    features_kwargs
        Keyword arguments for the different features that should be generated, such as
        ``{{ {f.TEXTURE.s!r}: {{ ... }}, ... }}``.
    key_added
        Key in :attr:`anndata.AnnData.obsm` where to store the calculated features.
    %(copy)s
    show_progress_bar
        Whether to show a progress bar.
    kwargs
        Keyword arguments for :meth:`squidpy.im.ImageContainer.generate_spot_crops`.

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` where columns correspond to the calculated features.

    Otherwise, modifies the ``adata`` object with the following key:

        - :attr:`anndata.AnnData.uns` ``['{{key_added}}']`` - the above mentioned dataframe.

    Raises
    ------
    ValueError
        If a feature is not known.
    """
    layer = img._get_layer(layer)
    if isinstance(features, str | ImageFeature):
        features = [features]
    features = sorted({ImageFeature(f).s for f in features})

    start = logg.info(f"Calculating features `{list(features)}`")

    features_list = []
    obs_ids = list(adata.obs_names)
    for crop in tqdm(
        img.generate_spot_crops(adata, library_id=library_id, return_obs=False, as_array=False, **kwargs),
        total=len(obs_ids),
        disable=not show_progress_bar,
    ):
        if TYPE_CHECKING:
            assert isinstance(crop, ImageContainer)
        crop = crop.compute(layer)

        features_dict = {}
        for feature in features:
            feature = ImageFeature(feature)
            feature_kwargs = features_kwargs.get(feature.s, {})

            if feature == ImageFeature.TEXTURE:
                res = crop.features_texture(layer=layer, **feature_kwargs)
            elif feature == ImageFeature.COLOR_HIST:
                res = crop.features_histogram(layer=layer, **feature_kwargs)
            elif feature == ImageFeature.SUMMARY:
                res = crop.features_summary(layer=layer, **feature_kwargs)
            elif feature == ImageFeature.SEGMENTATION:
                res = crop.features_segmentation(intensity_layer=layer, **feature_kwargs)
            elif feature == ImageFeature.CUSTOM:
                res = crop.features_custom(layer=layer, **feature_kwargs)
            else:
                raise NotImplementedError(f"Feature `{feature}` is not yet implemented.")

            features_dict.update(res)
        features_list.append(features_dict)

    res = pd.concat([pd.DataFrame([fd]) for fd in features_list], ignore_index=True)
    res.index = obs_ids

    if copy:
        logg.info("Finish", time=start)
        return res

    _save_data(adata, attr="obsm", key=key_added, data=res, time=start)
