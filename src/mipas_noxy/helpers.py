# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8:et
#
# Copyright (c) 2023 Stefan Bender
#
# This module is part of pymipas_noxy.
# pymipas_noxy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Helper functions for EPP-NOy calculation

Provides helper functions to select and smooth data of NOy, CH4, and CO
to separate EPP-NOy from stratospheric background NOy.
Might be useful for other tracer-tracer correlations too.
"""
from __future__ import absolute_import, division, print_function

# %%
import logging

# %%
import numpy as np
import xarray as xr

# %%
from astropy import constants as ac, units as au

# %%
from .correlate import histogram2d, hist_stats_ds
from .units import *

# %%
logging.basicConfig(
    level=logging.INFO,
    format=
        "[%(levelname)-8s] (%(asctime)s) "
        "%(filename)s:%(lineno)d:%(funcName)s() %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
)

# %%
logger = logging.getLogger()


# %%
def select_common_data(noy, ch4, co, on="geo_id"):
    _ids = sorted(set(noy[on].values) & set(ch4[on].values) & set(co[on].values))

    noy_id = noy.swap_dims({"time": on}).sel({on: _ids})
    ch4_id = ch4.swap_dims({"time": on}).sel({on: _ids})
    co_id = co.swap_dims({"time": on}).sel({on: _ids})
    return noy_id, ch4_id, co_id


# %%
def g_weights(xs, μ, σ):
    logger.debug("xs: %s, μ: %s, σ: %s", xs, μ, σ)
    ws = np.exp(-0.5 * (xs - μ)**2 / σ**2)
    dx = np.gradient(xs, axis=0)
    return ws * dx / (ws * dx).sum(axis=0)


# %%
def idl_smooth(d, dim, n):
    if not n % 2:
        n = n + 1
    ret = d.rolling(center=True, min_periods=n, **{dim: n}).mean(dim).fillna(d)
    return ret


# %%
def smooth_targets(a, b, smooth_vr=None, variable="target"):
    # cap altitude range to the one common to both
    # aa = a.sel(altitude=slice(0, np.minimum(a.altitude.max(), b.altitude.max())))
    aa = a
    vr_a = aa.vr_row
    vr_b = b.vr_row.interp(
        altitude=aa.altitude,
        kwargs=dict(bounds_error=False, fill_value=0.),
    )
    # In the IDL code, both resolution vectors are filtered through
    # a boxcar running average of 4 points before the convolution.
    # For IDL compatibility, pass `smooth_vr=4`.
    if smooth_vr is not None:
        vr_a = idl_smooth(vr_a, "altitude", smooth_vr)
        vr_b = idl_smooth(vr_b, "altitude", smooth_vr)
    _vr_diffsq = (vr_a.fillna(0.)**2 - vr_b.fillna(0.)**2)
    # _vr_diffsq = (vr_a**2)
    vr_diff = xr.where(_vr_diffsq > 0., np.sqrt(_vr_diffsq), 1e-12)
    # Construct convolution matrix for resolution reduction
    smooth_mat = xr.apply_ufunc(
        g_weights,
        b.altitude,
        aa.rename({"altitude": "out_alt"}).out_alt,
        # convert FWHM to sigma
        vr_diff.rename({"altitude": "out_alt"}) / (2 * np.sqrt(2 * np.log(2))),
        # input_core_dims=[["altitude"], ["out_alt"], ["out_alt"]],
        # output_core_dims=[["out_alt", "altitude", "geo_id"]],
        join="outer",
    )
    smooth_b = smooth_mat.dot(b[variable].fillna(0.), dims="altitude")
    smooth_b = smooth_b.rename({"out_alt": "altitude"})
    smooth_b.attrs = b[variable].attrs
    return smooth_b


# %%
def potential_temperature(pressure, temperature):
    tpot = temperature / ((pressure / (1000. * au.hPa))**(0.286))
    tpot = tpot.to_unit("K")
    tpot.attrs.update({
        "long_name": "Potential temperature",
        "standard_name": "air_potential_temperature",
        "reference_pressure": "1000 hPa",
    })
    return tpot


# %%
def calc_noy_bg_epp(
    ds, ch4_var, co_var, noy_var,
    ch4_bin_edges, noy_bin_edges,
    alt_range=(None, None), lat_range=(-91., 91.),
    akm_thresh=None,
    ch4_thresh=None,
    co_thresh=None,
    tpot_thresh=None,
    min_pts=0,
    min_tot=40,
    smooth_hist=None,
    copy_vars=["latitude", "lat_bnds", "longitude", "lon_bnds", "time_bnds"],
):
    corr_alts = alt_range
    corr_lats = lat_range
    # select altitude
    _ds = ds.sel(altitude=slice(*corr_alts))
    # select latitude
    _ds = _ds.where(
        (_ds.latitude > corr_lats[0])
        & (_ds.latitude < corr_lats[1]),
        drop=True,
    )
    if ch4_thresh is not None:
        logger.info("selecting by CH4 >= %g ppm", ch4_thresh)
        _ds = _ds.where((_ds[ch4_var] >= ch4_thresh), drop=True)
    if co_thresh is not None:
        logger.info("selecting by CO <= %g ppm", co_thresh)
        _ds = _ds.where((_ds[co_var] <= co_thresh), drop=True)
    if akm_thresh is not None:
        logger.info("selecting by AKM diag > %g", akm_thresh)
        _ds = _ds.where((_ds["akm_diagonal"] > akm_thresh), drop=True)
    if tpot_thresh is not None:
        logger.info("selecting by potential temperature > %g", tpot_thresh)
        if "T_pot" in ds.data_vars:
            _tpot = ds["T_pot"]
        else:
            _tpot = potential_temperature(_ds.pressure, _ds.temperature)
        _ds = _ds.where((_tpot > tpot_thresh), drop=True)
    for v in copy_vars:
        _ds[v] = ds[v] 
        _ds[v].attrs = ds[v].attrs 

    hist_da = histogram2d(_ds, ch4_var, noy_var, ch4_bin_edges, noy_bin_edges, density=False)
    # hist_da = histogram2d_kde(_ds, ch4_var, noy_var, ch4_bin_edges, noy_bin_edges, dims=("altitude", "geo_id"))
    _hist_da = xr.where(hist_da >= min_pts, hist_da, 0.)
    if smooth_hist is not None:
        # In the IDL code, the histogram is filtered (3 times) through
        # a boxcar running average of 4 points along the NOy axis.
        # For IDL compatibility, pass `smooth_hist=4`.
        for _ in range(3):
            _hist_da = idl_smooth(_hist_da, noy_var + "_bins", smooth_hist)
            logger.debug("smoothed hist: %s", _hist_da)

    logger.info("min %d histogram points", min_tot)
    ## %%
    hist_sds = hist_stats_ds(_hist_da, ch4_var, noy_var, min_pts=0, min_tot=min_tot)
    hist_sds.attrs.update(ds.attrs)
    hist_sds.attrs.update({
        "Altitude range [km]": corr_alts,
        "Latitude range [degrees_north]": corr_lats,
        "CH4 threshold used [ppm]": ch4_thresh or "none",
        "CO threshold used [ppm]": co_thresh or "none",
        "AKM diagonal threshold used [1]": akm_thresh or "none",
        "Potential temperature threshold used [K]": tpot_thresh or "none",
        "Histogramm boxcar size [NOy pts]": smooth_hist or "none",
    })
    hist_sds = hist_sds.expand_dims(latitude=[np.nanmean(corr_lats)])
    hist_sds.latitude.attrs = ds.latitude.attrs
    logger.debug("histogram ds: %s", hist_sds)
    return hist_sds


# %%
def sub_bg_noy(
    ds, h_sds,
    ch4_var, co_var, noy_var,
    co_thresh=None,
):
    ch4_binv = f"{ch4_var}_bins"
    _hist_mean = h_sds["mean"]
    logger.debug(_hist_mean)

    _noy_bg = _hist_mean.rename(
        {ch4_binv: ch4_var}
    ).interp(
        # interpolate from histogram CH4 to values from the dataset
        {ch4_var: ds[ch4_var]},
        kwargs=dict(fill_value="extrapolate"),
    )
    logger.debug(_noy_bg[ch4_var])
    _noy_bg = _noy_bg.drop(ch4_var)

    # %%
    # "background" NOy = all NOy where CO <= threshold,
    # in those places there should be no EPP effect.
    noy_bg = xr.where(
        (ds[co_var].to_unit("ppm") <= co_thresh),
        ds[noy_var],
        _noy_bg,
    )

    bg_name = noy_var + "_bg"
    epp_name = noy_var + "_epp"
    ret = ds.copy()
    ret[bg_name] = noy_bg
    ret[bg_name].attrs.update({
        "long_name": "volume mixing ratio of background NOy",
    })
    ret[epp_name] = ret[noy_var] - ret[bg_name]
    ret[epp_name].attrs.update({
        "long_name": "volume mixing ratio of EPP NOy",
    })
    ret.attrs.update(h_sds.attrs)
    return ret
