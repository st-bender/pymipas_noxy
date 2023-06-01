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
def trans_tpot(ds, arange=(22, 44)):
    """Find potential temperature at NOy/CH4 minimum in range
    """
    _ds_sel = ds.sel(altitude=slice(*arange))
    _min_ai = _ds_sel.noy_vs_ch4.argmin("altitude")
    min_ai = np.clip(_min_ai, 0, _ds_sel.altitude.size - 2)
    return _ds_sel.T_pot.isel(altitude=min_ai + 1)


# %%
def epp_noy_single(
    _mv8_sel,
    corr_ds,
    co_high=0.7,
    co_low=0.07,
    co_ch4_min=0.0175,
    ch4_fac=0.026,
    ch4_const=0.012,
    dim="geo_id",
    ivar="vmr_ch4",
    tpot_limits=None,
):
    """Background and EPP NOy calculation for a single profile
    """
    _lat = _mv8_sel.latitude
    _id = _mv8_sel[dim].values
    _noy_ch4 = corr_ds.sel({dim: _id})
    _mv8_ch4 = _mv8_sel[ivar]
    _mv8_co = _mv8_sel.vmr_co
    _mv8_noy = _mv8_sel.vmr_noy
    _mv8_coch4 = _mv8_co * _mv8_ch4
    _mv8_tpot = _mv8_sel.T_pot
    _bg_noy = _noy_ch4.interp({ivar: _mv8_ch4}).reset_coords()["mean"]
    _epp_noy0 = _mv8_noy - _bg_noy
    if tpot_limits is None:
        _potn = trans_tpot(_mv8_sel, arange=(22, 44))
        _pots = _potn
    else:
        _mv8_mnth = _mv8_sel.time.dt.month.values
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
    _epp_noy = xr.where(_mv8_sel.altitude > 20., _epp_noy, 0.)
    _epp_noy.attrs.update({
        "long_name": "volume mixing ratio of EPP-NOY",
    })
    logger.debug(_epp_noy)
    return _epp_noy


# %%
def _noy_co_ratio(ds, tpot_thr, eppnoy, co, dim="geo_id"):
    tpot_x = ds.T_pot.transpose(dim, "altitude")
    eppnoy_x = eppnoy.transpose(dim, "altitude")
    co_x = co.transpose(dim, "altitude")
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
    _mv8_sel, corr_ds,
    co_high=0.7,
    co_low=0.07,
    co_ch4_min=0.0175,
    ch4_fac=0.026,
    ch4_const=0.012,
    dim="geo_id",
    ivar="vmr_ch4",
    tpot_limits=None,
):
    """Background and EPP NOy calculation for multiple profiles
    """
    _lat = _mv8_sel.latitude
    _mv8_ch4 = _mv8_sel[ivar]
    _mv8_co = _mv8_sel.vmr_co
    _mv8_noy = _mv8_sel.vmr_noy
    _mv8_coch4 = _mv8_co * _mv8_ch4
    _mv8_tpot = _mv8_sel.T_pot
    _bg_noy = corr_ds["mean"]
    _epp_noy0 = _mv8_noy - _bg_noy

    if tpot_limits is None:
        _potn = trans_tpot(_mv8_sel, arange=(22, 44))
        _pots = _potn
    else:
        _mv8_mnth = _mv8_sel.time.dt.month.values[0]
        _potn = tpot_limits[(_mv8_mnth + 6 - 1) % 12]
        _pots = tpot_limits[_mv8_mnth - 1]
    _ratio_n = _noy_co_ratio(_mv8_sel, _potn, _epp_noy0, _mv8_co, dim=dim)
    _ratio_s = _noy_co_ratio(_mv8_sel, _pots, _epp_noy0, _mv8_co, dim=dim)

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
    _epp_noy = xr.where(_mv8_sel.altitude > 20., _epp_noy, 0.)
    _epp_noy.attrs.update({
        "long_name": "volume mixing ratio of EPP-NOY",
    })
    logger.debug(_epp_noy)
    return _epp_noy


# %%
def process_day_single(ds, corr_ds, dim="geo_id", ivar="vmr_ch4", **kwargs):
    corr_ds_sel = corr_ds.sel(
        time=ds.time.dt.floor("D"),
        latitude=ds.latitude,
        method="nearest",
    )
    if ivar + "_bins" in corr_ds_sel.dims:
        corr_ds_sel = corr_ds_sel.rename({ivar + "_bins": ivar})
    epp_noy_da = ds.groupby(dim).map(
        epp_noy_single,
        args=(corr_ds_sel,),
        dim=dim, ivar=ivar,
        **kwargs,
    )
    return epp_noy_da


# %%
def process_day_multi1(ds, corr_ds, dim="geo_id", ivar="vmr_ch4", **kwargs):
    corr_ds_sel = corr_ds.sel(
        time=ds.time.dt.floor("D"),
        latitude=ds.latitude,
        method="nearest",
    )
    if ivar + "_bins" in corr_ds_sel.dims:
        corr_ds_sel = corr_ds_sel.rename({ivar + "_bins": ivar})
    corr_ds_i = corr_ds_sel.groupby(dim).map(
        lambda _ds: _ds.interp({
            ivar: ds[ivar].sel({dim: _ds[dim]})
        }).reset_coords()
    )
    epp_noy_da = epp_noy_multi(ds, corr_ds_i, ivar=ivar, **kwargs)
    return epp_noy_da


# %%
def process_day_multi2(ds, corr_ds, ivar="vmr_ch4", **kwargs):
    _ti = np.unique(ds.time.dt.floor("D"))
    corr_ds_sel = corr_ds.sel(time=_ti[0])
    if ivar + "_bins" in corr_ds_sel.dims:
        corr_ds_sel = corr_ds_sel.rename({ivar + "_bins": ivar})
    # _ds = ds.swap_dims({"geo_id": "time"}).reset_coords()
    _ds = ds.copy()
    corr_ds_i = xr.where(
        _ds.latitude <= 0,
        corr_ds_sel.sel(latitude=-45).interp({ivar: _ds[ivar]}),
        corr_ds_sel.sel(latitude=45).interp({ivar: _ds[ivar]}),
    )
    epp_noy_da = epp_noy_multi(_ds, corr_ds_i, ivar=ivar, **kwargs)
    return epp_noy_da


# %%
def epp_tot_day(ds, corr_ds, dlat=5.0, ivar="vmr_ch4", name="vmr_noy_epp", **kwargs):
    epp_noy = process_day_multi2(ds, corr_ds, iver=ivar, **kwargs)
    mrg_ds = xr.merge([ds, epp_noy.drop(ivar).rename(name)])
    dzm_tds = calc_Ntot(mrg_ds, dlat=dlat)
    return dzm_tds


# %%
def integrate_eppnoy(
    ds,
    arange=(25, 70), lrange=(-90, 0),  # for integration
    asearch_max=44,
    lsearch_num=3,
    co_thresh=0.025,
    method="minimum",  # or "gradient" for the (minimum) gradient
    ntot_var="Ntot_noy_epp",
):
    """Integrates zonal mean columns for total number of molecules

    """
    # all-nan array to return in case there is not enough data
    empty = np.nan * xr.ones_like(
        ds[ntot_var].isel(altitude=0, latitude=0, drop=True),
    )
    _ds_reg = ds.sel(altitude=slice(*arange), latitude=slice(*lrange))
    if (np.isnan(_ds_reg[ntot_var])).sum() > 0.5 * _ds_reg[ntot_var].count():
        return empty
    if co_thresh is None:
        # select by noy/ch4 minimum altitude
        amin = arange[0]
        amax = asearch_max
        _noy_ch4_sel = _ds_reg.noy_vs_ch4.isel(
            latitude=slice(0, lsearch_num), drop=True,
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
        logger.debug(_min_alt)
        eppnoy_sel = _ds_reg[ntot_var].sel(altitude=slice(_min_alt, 70))
    else:
        # select by CO threshold
        eppnoy_sel = _ds_reg[ntot_var].where(_ds_reg.vmr_co > co_thresh)
    ret = np.maximum(0., eppnoy_sel).sum()
    logger.debug(ret)
    return ret
