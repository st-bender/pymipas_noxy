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
"""Zonal mean functions

Provides helper functions to calculate zonal means of a data set.
"""
from __future__ import absolute_import, division, print_function

# %%
import numpy as np
import xarray as xr

# %%
from astropy import constants as ac, units as au

# %%
from .units import *


# %%
def weighted_zm(ds, dim="time", variable="target", weight_var="weights"):
    # In some cases the weights can have different dimensions than the data
    # and simply summing the weights does not normalize them correctly.
    # Broadcasts them to consistent shapes and normalization with `.sum()`.
    # NaNs in the data should also propagate this way and get zero weights.
    weights = ds[weight_var].where(ds[variable].notnull())
    weights = weights.fillna(0.)
    weights.attrs = {"long_name": "weights", "units": "1"}
    ww = weights / weights.sum(dim)
    return (ds[variable] * ww).sum(dim)


# %%
def calc_zms(
    ds,
    dlat=5.0,
    dim="geo_id",
    variable="target",  # for `ncount`
):
    # zonal means
    lat_edges = np.arange(-90, 90 + 0.5 * dlat, dlat)
    lat_cntrs = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    # skip time and string variables for zonal mean averaging.
    # zm_vars = [_v for _v in ds.data_vars if ds[_v].dtype.char not in "MS"]
    zm_vars = list(filter(lambda _v: ds[_v].dtype.char not in "MSOU", ds.data_vars))
    ds = ds.set_coords(("latitude",))  # set as coordinate for binning

    weights = np.cos(ds["latitude"].to_unit(au.radian))
    weights.attrs = {"long_name": "cosine of latitude", "units": "1"}
    weights = weights.where(ds[zm_vars].notnull())
    weights = weights.fillna(0.)
    # weighted data sum per bin
    zm_wdsum = (ds[zm_vars] * weights).groupby_bins(
        "latitude", lat_edges, labels=lat_cntrs
    ).sum(dim=dim)
    # weight sum per bin
    zm_wsum = (weights).groupby_bins("latitude", lat_edges, labels=lat_cntrs).sum(dim=dim)
    zm_ds = zm_wdsum / zm_wsum

    # Weight sum and normal count in bin
    zm_ds["wsum"] = zm_wsum[variable]
    zm_ds["wsum"].attrs = {"long_name": "sum of weights in bin", "units": "1"}
    zm_ds["ncount"] = ds[variable].groupby_bins(
        "latitude", lat_edges, labels=lat_cntrs,
    ).count(dim=dim)
    zm_ds["ncount"].attrs = {
        "long_name": "number of data points in bin", "units": "1",
    }

    if "latitude" in zm_ds:
        # rename old latitudes first
        zm_ds = zm_ds.rename({"latitude": "lat_orig"})
    zm_ds = zm_ds.rename({"latitude_bins": "latitude"})
    zm_ds.latitude.attrs = ds.latitude.attrs
    if "lat_orig" in zm_ds:
        # copy attributes from the "original" latitudes
        zm_ds.latitude.attrs = zm_ds.lat_orig.attrs

    if "time" in dim:
        zm_ds["time"] = ds.time.mean("time").dt.round("1ms")
        zm_ds = zm_ds.set_coords("time")
        # zm_ds = zm_ds.expand_dims(time=[ds.time.mean("time").dt.round("1ms").values])

    # Note: the IDL code exports standard deviation and standard error
    # from which one can obtain the number of data points.
    ncnts = (weights[variable] > 0.0).groupby_bins(
        "latitude", lat_edges, labels=lat_cntrs,
    ).sum(dim=dim)
    ncnts.attrs = {"long_name": "number of points in bin", "units": "1"}
    # weighted standard deviations
    zm_ds_wm = zm_ds[variable].drop("time").sel(latitude=ds.latitude, method="nearest")
    wstds2_num1 = ((ds[variable] - zm_ds_wm)**2 * weights[variable]).groupby_bins(
        "latitude", lat_edges, labels=lat_cntrs
    ).sum(dim=dim)
    wstds = np.sqrt(wstds2_num1 * ncnts / (zm_wsum[variable] * (ncnts - 1)))
    wstds = wstds.rename({"latitude_bins": "latitude"})
    # wstds = wstds.rename({v: v + "_std" for v in wstds.data_vars})
    wstds = wstds.rename(variable + "_std")
    zm_ds = xr.merge([zm_ds, wstds])

    return zm_ds


# %%
def calc_Ntot(
    ds,
    vaxis=-1,  # vertical/radial axis for dr
    **kwargs  # passed to `calc_zms()`
):
    #p_unit = getattr(au, ds.pressure.attrs["units"])
    #T_unit = getattr(au, ds.temperature.attrs["units"])
    # ndens = ds.pressure.data * p_unit / (ac.R * ds.temperature.data * T_unit)
    ndens = ds.pressure / ac.R / ds.temperature
    # convert to number densities
    nd_ds = ds.copy()
    nd_ds["ndens"] = ndens
    nd_ds["ndens"].attrs.update({"long_name": "number density of molecules in air"})
    for _v in filter(lambda _v: _v.startswith("vmr_"), nd_ds.data_vars):
        nd_name = "nd_" + _v[4:]
        nd_ds[nd_name] = nd_ds["ndens"] * nd_ds[_v]
        nd_ds[nd_name].attrs.update({
            "long_name":
                nd_ds[_v].attrs["long_name"].replace(
                    "volume mixing ratio", "number density"
                ),
            "standard_name":
                nd_ds[_v].attrs["standard_name"].replace(
                    "mole_fraction", "mole_concentration"
                ),
        })
    # zonal means
    zm_ds = calc_zms(nd_ds, **kwargs)
    # volume element
    alt_unit = au.Unit(ds.altitude.attrs["units"])
    # dlambda = (np.cos(np.radians(zm_ds.latitude)) * np.radians(dlat)).values
    dlambda = np.diff(np.sin(np.radians(xr.plot.utils._infer_interval_breaks(zm_ds.latitude))))
    dr = np.abs(np.gradient(zm_ds.altitude, axis=vaxis)) * alt_unit
    rr = ac.R_earth + zm_ds.altitude.values * alt_unit
    dvol = 2 * np.pi * (rr**2 * dr) * dlambda[:, None]
    dvol = dvol.to("m3")
    dvol_vdim = zm_ds.altitude.dims[vaxis]
    zm_ds["dvol"] = (
        ("latitude", dvol_vdim), dvol,
        {"long_name": "volume element", "standard_name": "air_volume", "units": dvol.unit}
    )
    for _v in filter(lambda _v: _v.startswith("nd_"), zm_ds.data_vars):
        ntot_name = "Ntot_" + _v[3:]
        zm_ds[ntot_name] = (zm_ds[_v] * dvol).to_unit("mol")
        zm_ds[ntot_name].attrs.update({
            "long_name":
                zm_ds[ntot_name].attrs["long_name"].replace(
                    "number density", "moles"
                ),
            "standard_name":
                zm_ds[ntot_name].attrs["standard_name"].replace(
                    "mole_concentration", "atmosphere_moles"
                ),
        })
    return zm_ds
