# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2023 Stefan Bender
#
# This module is part of pymipas_noxy.
# pymipas_noxy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Correlation histogram utility functions

Utility functions to correlate trace gases against each other.
Initially designed to correlate NOy and CH4 to separate
"background" from "EPP-induced" NOy.
"""
from __future__ import absolute_import, division, print_function

from logging import info, debug

import numpy as np
import xarray as xr

from scipy.stats import gaussian_kde


# %%
def histogram2d_colwise(ds, x, y, x_edges, y_edges, density=False):
    """2-D Histogram with column-wise 1-D histograms

    Assembles a 2-D histogram of the data by grouping "column-wise"
    1-D histograms.

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset.
    x: str
        Name of the x-variable in the dataset.
    y: str
        Name of the y-variable in the dataset.
    x_edges: array-like (M,)
        Bin edges of the x-variable.
    y_edges: array-like (N,)
        Bin edges of the y-variable.
    density: bool, optional (default False)
        Whether or not to normalize the individual histograms
        to unit density.

    Returns
    -------
    ret: xarray.Dataset
    """
    x_ctr = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_ctr = 0.5 * (y_edges[:-1] + y_edges[1:])
    y_binv = y + "_bins"
    ret = ds.groupby_bins(x, x_edges, labels=x_ctr).apply(
        lambda d: xr.DataArray(
            np.histogram(d[y], bins=y_edges, density=density)[0],
            dims=(y_binv,),
            coords={y_binv: y_ctr},
        )
    )
    return ret


# %%
def histogram2d(ds, x, y, x_edges, y_edges, density=False):
    """2-D Histogram with `numpy.histogram2d`

    Produces a 2-D histogram using numpy's `histogram2d()` function.

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset.
    x: str
        Name of the x-variable in the dataset.
    y: str
        Name of the y-variable in the dataset.
    x_edges: array-like (M,)
        Bin edges of the x-variable.
    y_edges: array-like (N,)
        Bin edges of the y-variable.
    density: bool, optional (default False)
        Whether or not to normalize the 2-D histogram to unit density.

    Returns
    -------
    ret: xarray.DataArray
        DataArray containing the histogram with dimensions and
        coordinates set accordingly.

    See Also
    --------
    numpy.histogram2d
    """
    x_ctr = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_ctr = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_binv = x + "_bins"
    y_binv = y + "_bins"

    _hist = np.histogram2d(
        ds[x].values.ravel(),
        ds[y].values.ravel(),
        bins=[x_edges, y_edges],
        density=density,
    )
    ret = xr.DataArray(
        _hist[0],
        dims=(x_binv, y_binv,),
        coords={x_binv: x_ctr, y_binv: y_ctr}
    )
    return ret


# %%
def histogram2d_kde(
    ds, x, y, x_edges, y_edges,
    dims=("altitude", "geo_id",),
    **kwargs,
):
    """2-D Histogram using Gaussian kernel density estimate

    Calculates a 2-D histogram using scipy's Gaussian kernel density.

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset.
    x: str
        Name of the x-variable in the dataset.
    y: str
        Name of the y-variable in the dataset.
    x_edges: array-like (M,)
        Bin edges of the x-variable.
    y_edges: array-like (N,)
        Bin edges of the y-variable.
    dim: str or tuple
        Dimension name(s) over which to aggregate the histogram.
    **kwargs: optional
        Keyword arguments to be passed to `scipy.gaussian_kde()`

    Returns
    -------
    ret: xarray.DataArray
        DataArray containing the histogram with dimensions and
        coordinates set accordingly.

    See Also
    --------
    scipy.gaussian_kde
    """
    x_ctr = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_ctr = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_binv = x + "_bins"
    y_binv = y + "_bins"

    pX, pY = np.meshgrid(x_ctr, y_ctr)
    poss = np.vstack([pX.ravel(), pY.ravel()])

    gkde = gaussian_kde(
        ds[[x, y]].stack(_id=dims).dropna("_id").to_array(),
        **kwargs,
    )
    info("kde bandwidth factor: %f", gkde.factor)
    kde_eval = gkde(poss)

    ret = xr.DataArray(
        kde_eval.reshape(pX.shape).T,
        dims=(x_binv, y_binv,),
        coords={x_binv: x_ctr, y_binv: y_ctr}
    )
    return ret


# %%
def xy_interpolate_to_zero(ds, bin_var, **kwargs):
    """Interpolate the mean, mode, or median NOy to zero CH4

    Interpolates the determined NOy per CH4 to zero CH4 vmr
    for interpolation and to calculate the background NOy from it.
    """
    _ds = ds.sel({bin_var: slice(-1e-12, None)})
    mask = np.isfinite(_ds)
    mi = np.where(mask)

    if np.all(_ds[mi].coords[bin_var] > 0.):
        # Add zero CH4 point for interpolation
        _zero_pt = xr.zeros_like(
            _ds.isel({bin_var: [0]})
        ).assign_coords({bin_var: [0.]})
        _ds = xr.concat([_zero_pt, _ds], dim=bin_var)
    # interpolate (over nans)
    ret = _ds.interpolate_na(dim=bin_var, **kwargs)
    return ret


# %%
def hist_mean(hist_da, x_var, y_var, min_hpts=50):
    _h_mean = (hist_da[y_var].data * hist_da).sum(y_var) / hist_da.sum(y_var)
    _hist_mean = xr.where(hist_da.sum(y_var) >= min_hpts, _h_mean, np.nan)
    debug("mean: %s", _hist_mean)
    return xy_interpolate_to_zero(_hist_mean, x_var, method="linear")


# %%
def hist_median(hist_da, x_var, y_var, min_hpts=50):
    _hcsum = hist_da.cumsum(y_var)
    _ix = []
    for _i in range(_hcsum.shape[0]):
        _ix.append(np.searchsorted(_hcsum[_i], 0.5 * hist_da.sum(y_var)[_i]))
    _h_median = hist_da.isel({y_var: _ix})[y_var]
    _h_median[x_var] = (y_var, hist_da[x_var].data)
    _h_median = _h_median.swap_dims({y_var: x_var})

    _hist_median = xr.where(hist_da.sum(y_var) >= min_hpts, _h_median, np.nan)
    debug("median: %s", _hist_median)
    return xy_interpolate_to_zero(_hist_median, x_var, method="linear")


# %%
def hist_mode(hist_da, x_var, y_var, min_hpts=50):
    _h_mode = hist_da.isel({y_var: hist_da.argmax(y_var)})[y_var]
    _hist_mode = xr.where(hist_da.sum(y_var) >= min_hpts, _h_mode, np.nan)
    debug("mode: %s", _hist_mode)
    return xy_interpolate_to_zero(_hist_mode, x_var, method="linear")
