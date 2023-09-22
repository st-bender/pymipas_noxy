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
"""Background for EPP-NOy, and column integration

Provides functions to calculate and subtract background NOy,
previously derived from correlating NOy, CH4, and CO.
Might be useful for other tracer-tracer correlations too.
"""
from __future__ import absolute_import, division, print_function

# %%
import logging

# %%
import numpy as np
import xarray as xr

# %%
from .helpers import calc_Ntot

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
def tpot_at_noych4_min(ds, arange=(22, 44)):
    """Find potential temperature at NOy/CH4 minimum in range
    """
    _ds_sel = ds.sel(altitude=slice(*arange))
    _min_ai = _ds_sel.noy_vs_ch4.argmin("altitude")
    min_ai = np.clip(_min_ai, 0, _ds_sel.altitude.size - 2)
    return _ds_sel.T_pot.isel(altitude=min_ai + 1)


# %%
def epp_noy_single(
    ds,
    corr_ds,
    ch4_var, co_var, noy_var,
    co_high=0.7,
    co_low=0.07,
    co_ch4_min=0.0175,
    ch4_fac=0.026,
    ch4_const=0.012,
    dim="geo_id",
    tpot_limits=None,
):
    """Background and EPP NOy calculation for a single profile

    Separates the background and EPP-NOy contents based on the
    correlations with CH4 and on thresholds based on the CO.
    This function operates on a single profile, useful with
    `<ds>.groupby(dim).map(...)`.

    Parameters
    ----------
    ds: `xarray.Dataset`
        Combined dataset containing the trace gas data,
        including the vmrs of CO, CH4, and NOy.
    corr_ds: `xarray.Dataset`
        Dataset containing the NOy/CH4 correlation values.
    ch4_var: str
        Name of the dataset variable containing the CH4 vmr.
    co_var: str
        Name of the dataset variable containing the CO vmr.
    noy_var: str
        Name of the dataset variable containing the NOy vmr.
    co_high: float, optional (default: 0.7)
        Threshold of CO vmr (in ppm) above which all NOy in
        the volume is considered EPP-NOy.
    co_low: float, optional (default: 0.07)
        Threshold of CO vmr (in ppm) below which all NOy in
        the volume is considered background NOy.
    co_ch4_min: float, optional (default: 0.0175)
        Threshold for the product of CO and CH4 vmr below which
        all NOy in the volume is considered background NOy.
    ch4_const: float, optional (default: 0.012)
        Intercept (a) of the linear CH4 threshold to consider all
        NOy in the volume as EPP-NOy, i.e. if NOy > a + b * CH4,
        see also `ch4_fac`.
    ch4_fac: float, optional (default: 0.026)
        Slope (b) of the linear CH4 threshold to consider all NOy
        in the volume as EPP-NOy, i.e. if NOy > a + b * CH4,
        see also `ch4_const`.
    dim: str, optional (default: "geo_id")
        The dimension along which the profiles are "stacked".
        Only used to select the same correlation point.
    tpot_limits: list of float, optional (default: None)
        List of monthly potential temperature limits in the SH
        to separate EPP-NOy from background NOy. For the NH,
        the list is rotated by 6 months.

    Returns
    -------
    epp_noy: `xarray.DataArray`
        The EPP-NOy content (vmr).

    See Also
    --------
    epp_noy_multi, process_day_single
    """
    _lat = ds.latitude
    _id = ds[dim].values
    _noy_ch4 = corr_ds.sel({dim: _id})
    _mv8_ch4 = ds[ch4_var]
    _mv8_co = ds[co_var]
    _mv8_noy = ds[noy_var]
    _mv8_coch4 = _mv8_co * _mv8_ch4
    _mv8_tpot = ds.T_pot
    try:
        # check if it is a named DataArray
        _name = corr_ds.name
    except AttributeError:
        # Otherwise probably a Dataset, use the default "mean"
        _name = "mean"
    # `.reset_coords()` converts the unused coordinates to variables
    # returning a Dataset.
    # Converts it back to a DataArray by selecting the named variable,
    # either from the DataArray passed to the function, or using the
    # default name ("mean").
    _bg_noy = _noy_ch4.interp(
        {ch4_var: _mv8_ch4}, kwargs=dict(fill_value="extrapolate"),
    ).reset_coords()[_name]
    _epp_noy0 = _mv8_noy - _bg_noy
    if tpot_limits is None:
        _potn = tpot_at_noych4_min(ds, arange=(22, 44))
        _pots = _potn
    else:
        _mv8_mnth = ds.time.dt.month.values
        _potn = tpot_limits[(_mv8_mnth + 6 - 1) % 12]
        _pots = tpot_limits[_mv8_mnth - 1]
    _epp_noy = xr.zeros_like(_mv8_noy)
    # high CO => all NOy is considered from EPP
    cond1 = (_mv8_co > co_high) | (_mv8_noy > (ch4_const + _mv8_ch4 * ch4_fac))
    _epp_noy += xr.where(cond1, _epp_noy0, 0.)
    if _lat <= 0:
        ggs = np.where(_mv8_tpot >= _pots)[0][0:5]
        _ratio_s = np.maximum(0, (_epp_noy0[ggs] / _mv8_co[ggs]).mean())
        _epp_noy += xr.where(
            ~cond1 & (_mv8_co > co_low) & (_mv8_coch4 > co_ch4_min),
            xr.where(
                (_mv8_tpot < _pots) & ((_epp_noy0 / _mv8_co) > _ratio_s),
                _mv8_co * _ratio_s,
                _epp_noy0
            ),
            0.
        )
    else:
        ggn = np.where(_mv8_tpot >= _potn)[0][0:5]
        _ratio_n = np.maximum(0, (_epp_noy0[ggn] / _mv8_co[ggn]).mean())
        _epp_noy += xr.where(
            ~cond1 & (_mv8_co > co_low) & (_mv8_coch4 > co_ch4_min),
            xr.where(
                (_mv8_tpot < _potn) & ((_epp_noy0 / _mv8_co) > _ratio_n),
                _mv8_co * _ratio_n,
                _epp_noy0
            ),
            0.
        )
    _epp_noy = xr.where(ds.altitude > 20., _epp_noy, 0.)
    _epp_noy.attrs.update({
        "long_name": "volume mixing ratio of EPP NOy",
    })
    logger.debug(_epp_noy)
    return _epp_noy


# %%
def _noy_co_ratio(ds, tpot_thr, eppnoy, co, dim="geo_id"):
    tdim = dim
    if not hasattr(dim, "__getitem__") or isinstance(dim, str):
        # convert single items or strings to tuple for concatenation
        tdim = (dim,)
    dims = tuple(tdim) + ("altitude",)
    tpot_x = ds.T_pot.transpose(*dims)
    eppnoy_x = eppnoy.transpose(*dims)
    co_x = co.transpose(*dims)
    _tp_wh = np.where(tpot_x >= tpot_thr)
    _iix = np.where(np.diff(_tp_wh[0]) > 0)
    _iixp = np.concatenate([[0], _iix[0] + 1])
    _gi = _tp_wh[0][_iixp]
    _ai = _tp_wh[1][_iixp]
    _rrv = np.array([
        eppnoy_x.values[_gi, _ai + _i] / co_x.values[_gi, _ai + _i]
        for _i in range(5)
    ])
    _rr = np.nanmean(_rrv, axis=0)
    return np.maximum(0., _rr) * xr.ones_like(ds[dim], dtype=float)


# %%
def epp_noy_multi(
    ds, corr_ds,
    ch4_var, co_var, noy_var,
    co_high=0.7,
    co_low=0.07,
    co_ch4_min=0.0175,
    ch4_fac=0.026,
    ch4_const=0.012,
    dim="geo_id",
    tpot_limits=None,
):
    """Background and EPP NOy calculation for multiple profiles

    Separates the background and EPP-NOy contents based on the
    correlations with CH4 and on thresholds based on the CO.
    This function operates on multiple profiles at once.

    Parameters
    ----------
    ds: `xarray.Dataset`
        Combined dataset containing the trace gas data,
        including the vmrs of CO, CH4, and NOy.
    corr_ds: `xarray.Dataset`
        Dataset containing the NOy/CH4 correlation values.
    ch4_var: str
        Name of the dataset variable containing the CH4 vmr.
    co_var: str
        Name of the dataset variable containing the CO vmr.
    noy_var: str
        Name of the dataset variable containing the NOy vmr.
    co_high: float, optional (default: 0.7)
        Threshold of CO vmr (in ppm) above which all NOy in
        the volume is considered EPP-NOy.
    co_low: float, optional (default: 0.07)
        Threshold of CO vmr (in ppm) below which all NOy in
        the volume is considered background NOy.
    co_ch4_min: float, optional (default: 0.0175)
        Threshold for the product of CO and CH4 vmr below which
        all NOy in the volume is considered background NOy.
    ch4_const: float, optional (default: 0.012)
        Intercept (a) of the linear CH4 threshold to consider all
        NOy in the volume as EPP-NOy, i.e. if NOy > a + b * CH4,
        see also `ch4_fac`.
    ch4_fac: float, optional (default: 0.026)
        Slope (b) of the linear CH4 threshold to consider all NOy
        in the volume as EPP-NOy, i.e. if NOy > a + b * CH4,
        see also `ch4_const`.
    dim: str, optional (default: "geo_id")
        The dimension along which the profiles are "stacked".
        Only used to select the same correlation point.
    tpot_limits: list of float, optional (default: None)
        List of monthly potential temperature limits in the SH
        to separate EPP-NOy from background NOy. For the NH,
        the list is rotated by 6 months.

    Returns
    -------
    epp_noy: `xarray.DataArray`
        The EPP-NOy content (vmr).

    See Also
    --------
    epp_noy_single, process_day_multi1, process_day_multi2
    """
    _lat = ds.latitude
    _mv8_ch4 = ds[ch4_var]
    _mv8_co = ds[co_var]
    _mv8_noy = ds[noy_var]
    _mv8_coch4 = _mv8_co * _mv8_ch4
    _mv8_tpot = ds.T_pot
    _bg_noy = corr_ds
    _epp_noy0 = _mv8_noy - _bg_noy

    if tpot_limits is None:
        _potn = tpot_at_noych4_min(ds, arange=(22, 44))
        _pots = _potn
    else:
        _mv8_mnth = ds.time.dt.month.values[0]
        _potn = tpot_limits[(_mv8_mnth + 6 - 1) % 12]
        _pots = tpot_limits[_mv8_mnth - 1]
    _ratio_n = _noy_co_ratio(ds, _potn, _epp_noy0, _mv8_co, dim=dim)
    _ratio_s = _noy_co_ratio(ds, _pots, _epp_noy0, _mv8_co, dim=dim)

    _epp_noy = xr.zeros_like(_mv8_noy)
    # large CO vmr ~ all NOy is from EPP
    cond1 = (_mv8_co > co_high) | (_mv8_noy > (ch4_const + _mv8_ch4 * ch4_fac))
    _epp_noy += xr.where(cond1, _epp_noy0, 0.)
    
    # SH data
    _epp_noy += xr.where(
        ~cond1 & (_lat <= 0.) & (_mv8_co > co_low) & (_mv8_coch4 > co_ch4_min),
        xr.where(
            (_mv8_tpot < _pots) & ((_epp_noy0 / _mv8_co) > _ratio_s),
            _mv8_co * _ratio_s,
            _epp_noy0
        ),
        0.
    )
    # NH data
    _epp_noy += xr.where(
        ~cond1 & (_lat > 0.) & (_mv8_co > co_low) & (_mv8_coch4 > co_ch4_min),
        xr.where(
            (_mv8_tpot < _potn) & ((_epp_noy0 / _mv8_co) > _ratio_n),
            _mv8_co * _ratio_n,
            _epp_noy0
        ),
        0.
    )
    _epp_noy = xr.where(ds.altitude > 20., _epp_noy, 0.)
    _epp_noy.attrs.update({
        "long_name": "volume mixing ratio of EPP NOy",
    })
    logger.debug(_epp_noy)
    return _epp_noy


# %%
def process_day_single(
    ds, corr_ds,
    ch4_var, co_var, noy_var,
    dim="geo_id",
    **kwargs,
):
    """Daily background and EPP NOy calculation using a single profiles

    Calculate daily EPP-NOy contents based on the correlations
    of NOy with CH4 and on thresholds based on the CO.
    Combines the correlation selection and background/EPP handling
    for easier use. Subtracts and fine-tunes the background-NOy
    by iterating over all profiles separately.

    Parameters
    ----------
    ds: `xarray.Dataset`
        Combined dataset containing the trace gas data,
        including the vmrs of CO, CH4, and NOy.
    corr_ds: `xarray.Dataset`
        Dataset containing the NOy/CH4 correlation values.
    ch4_var: str
        Name of the dataset variable containing the CH4 vmr.
    co_var: str
        Name of the dataset variable containing the CO vmr.
    noy_var: str
        Name of the dataset variable containing the NOy vmr.
    dim: str, optional (default: "geo_id")
        The dimension along which the profiles are iterated,
        e.g. "time" or "geo_id".
    **kwargs: dict
        Keyword arguments passed to `epp_noy_single()`.

    Returns
    -------
    epp_noy: `xarray.DataArray`

    See Also
    --------
    epp_noy_single
    """
    corr_ds_sel = corr_ds.sel(
        time=ds.time.dt.floor("D"),
        latitude=ds.latitude,
        method="nearest",
    )
    if ch4_var + "_bins" in corr_ds_sel.dims:
        corr_ds_sel = corr_ds_sel.rename({ch4_var + "_bins": ch4_var})
    epp_noy_da = ds.groupby(dim).map(
        epp_noy_single,
        args=(corr_ds_sel, ch4_var, co_var, noy_var),
        dim=dim,
        **kwargs,
    )
    return epp_noy_da


# %%
def process_day_multi1(
    ds, corr_ds,
    ch4_var, co_var, noy_var,
    dim="geo_id",
    **kwargs,
):
    """Daily background and EPP NOy calculation using all profiles

    Calculate daily EPP-NOy contents based on the correlations
    of NOy with CH4 and on thresholds based on the CO.
    Combines the correlation selection and background/EPP handling
    for easier use. Subtracts and fine-tunes the background-NOy
    by processing all profiles simultaneously.
    Variant 1 by interpolating the background correlated NOy amount
    per individual profile.

    Parameters
    ----------
    ds: `xarray.Dataset`
        Combined dataset containing the trace gas data,
        including the vmrs of CO, CH4, and NOy.
    corr_ds: `xarray.Dataset`
        Dataset containing the NOy/CH4 correlation values.
    ch4_var: str
        Name of the dataset variable containing the CH4 vmr.
    co_var: str
        Name of the dataset variable containing the CO vmr.
    noy_var: str
        Name of the dataset variable containing the NOy vmr.
    dim: str, optional (default: "geo_id")
        The dimension along which the profiles are iterated,
        e.g. "time" or "geo_id".
    **kwargs: dict
        Keyword arguments passed to `epp_noy_single()`.

    Returns
    -------
    epp_noy: `xarray.DataArray`

    See Also
    --------
    epp_noy_multi, process_day_multi2
    """
    corr_ds_sel = corr_ds.sel(
        time=ds.time.dt.floor("D"),
        latitude=ds.latitude,
        method="nearest",
    )
    if ch4_var + "_bins" in corr_ds_sel.dims:
        corr_ds_sel = corr_ds_sel.rename({ch4_var + "_bins": ch4_var})
    corr_ds_i = corr_ds_sel.groupby(dim).map(
        lambda _ds: _ds.interp({
            ch4_var: ds[ch4_var].sel({dim: _ds[dim]})
        }, kwargs=dict(fill_value="extrapolate")).reset_coords()
    )
    epp_noy_da = epp_noy_multi(ds, corr_ds_i, ch4_var, co_var, noy_var, **kwargs)
    return epp_noy_da


# %%
def process_day_multi2(
    ds, corr_ds,
    ch4_var, co_var, noy_var,
    **kwargs,
):
    """Daily background and EPP NOy calculation using a single profiles

    Calculate daily EPP-NOy contents based on the correlations
    of NOy with CH4 and on thresholds based on the CO.
    Combines the correlation selection and background/EPP handling
    for easier use. Subtracts and fine-tunes the background-NOy
    by processing all profiles simultaneously.
    Variant 2 by interpolating the background correlated NOy amount
    per hemisphere.

    Parameters
    ----------
    ds: `xarray.Dataset`
        Combined dataset containing the trace gas data,
        including the vmrs of CO, CH4, and NOy.
    corr_ds: `xarray.Dataset`
        Dataset containing the NOy/CH4 correlation values.
    ch4_var: str
        Name of the dataset variable containing the CH4 vmr.
    co_var: str
        Name of the dataset variable containing the CO vmr.
    noy_var: str
        Name of the dataset variable containing the NOy vmr.
    dim: str, optional (default: "geo_id")
        The dimension along which the profiles are iterated,
        e.g. "time" or "geo_id".
    **kwargs: dict
        Keyword arguments passed to `epp_noy_single()`.

    Returns
    -------
    epp_noy: `xarray.DataArray`

    See Also
    --------
    epp_noy_multi, process_day_multi1
    """
    _ti = np.unique(ds.time.dt.floor("D"))
    corr_ds_sel = corr_ds.sel(time=_ti[0], method="nearest")
    if ch4_var + "_bins" in corr_ds_sel.dims:
        corr_ds_sel = corr_ds_sel.rename({ch4_var + "_bins": ch4_var})
    if "time" in corr_ds_sel.coords:
        corr_ds_sel = corr_ds_sel.drop("time")
    # _ds = ds.swap_dims({"geo_id": "time"}).reset_coords()
    _ds = ds.copy()
    corr_ds_i = xr.where(
        _ds.latitude <= 0,
        corr_ds_sel.sel(latitude=-45, method="nearest").drop("latitude").interp(
            {ch4_var: _ds[ch4_var]}, kwargs=dict(fill_value="extrapolate"),
        ),
        corr_ds_sel.sel(latitude=45, method="nearest").drop("latitude").interp(
            {ch4_var: _ds[ch4_var]}, kwargs=dict(fill_value="extrapolate"),
        ),
    )
    epp_noy_da = epp_noy_multi(_ds, corr_ds_i, ch4_var, co_var, noy_var, **kwargs)
    return epp_noy_da


# %%
def epp_tot_day(
    ds, corr_ds,
    ch4_var, co_var, noy_var,
    dlat=5.0, name="vmr_noy_epp",
    **kwargs,
):
    epp_noy = process_day_multi2(ds, corr_ds, ch4_var, co_var, noy_var, **kwargs)
    mrg_ds = xr.merge([ds, epp_noy.drop(ch4_var).rename(name)])
    dzm_tds = calc_Ntot(mrg_ds, dlat=dlat)
    return dzm_tds


# %%
def integrate_eppnoy(
    ds,
    arange=(25, 70), lrange=(-90, 0),  # for integration
    asearch_max=44,
    lsearch_num=3,
    co_thresh=None,
    co_var="vmr_co",
    method="minimum",  # or "gradient" for the (minimum) gradient
    ntot_var="Ntot_noy_epp",
    dims=("altitude", "latitude"),
):
    """Integrates zonal mean columns for total number of molecules

    """
    # all-nan array to return in case there is not enough data
    empty = np.nan * xr.ones_like(
        ds[ntot_var].isel(altitude=[0], latitude=[0], drop=False),
    )
    empty = empty.assign_coords(
        altitude=[np.nanmean(arange)],
        latitude=[np.nanmean(lrange)],
    )

    _ds_reg = ds.sel(altitude=slice(*arange), latitude=slice(*lrange))
    if (np.isnan(_ds_reg[ntot_var])).sum() > 0.5 * _ds_reg[ntot_var].count():
        return empty
    if np.mean(lrange) <= 0:
        lslice = slice(0, lsearch_num)
    else:
        lslice = slice(-lsearch_num, -1)
    if co_thresh is None:
        # select by noy/ch4 minimum altitude
        amin = arange[0]
        amax = asearch_max
        _noy_ch4_sel = _ds_reg.noy_vs_ch4.isel(
            latitude=lslice, drop=True,
        ).sel(altitude=slice(amin, amax))
        # fill negative values with the median for finding the minimum
        _noy_ch4_sel = xr.where(_noy_ch4_sel > 0, _noy_ch4_sel, _noy_ch4_sel.median())
        try:
            if method == "minimum":
                _min_ais = _noy_ch4_sel.argmin("altitude")
            elif method == "gradient":
                _min_ais = np.gradient(_noy_ch4_sel, axis=1).argmin(axis=1)
            else:
                logger.warn("unsupported minimum method: %s", method)
                return empty
            _min_ai = _min_ais.mean().round().astype(int)
        except ValueError:
            # couldn't find the index for the minimum
            return empty
        _min_alt = _ds_reg.altitude.isel(altitude=_min_ai).values
        logger.info("integrating from %g km to top", _min_alt)
        eppnoy_sel = _ds_reg[ntot_var].sel(altitude=slice(_min_alt, 70))
    else:
        # select by CO threshold
        eppnoy_sel = _ds_reg[ntot_var].where(_ds_reg[co_var].to_unit("ppm") > co_thresh)
    ret = np.maximum(0., eppnoy_sel).sum(dims)
    ret = ret.expand_dims(
        altitude=[np.nanmean(arange)],
        latitude=[np.nanmean(lrange)],
    )
    ret["altitude"].attrs = ds.altitude.attrs
    ret["latitude"].attrs = ds.latitude.attrs
    logger.debug(ret)
    return ret
