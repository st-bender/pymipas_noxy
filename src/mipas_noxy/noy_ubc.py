# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8:et
#
# Copyright (c) 2024 Stefan Bender
#
# This module is part of pymipas_noxy.
# pymipas_noxy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""NOy Upper-Boundary-Condition-Model functions

Provides functions for the NOy upper boundary condition model as
described in Funke et al., 2016 [1]_.

.. [1] Funke et al., Atmos. Chem. Phys., 16, 8667--8693, 2016
   doi: 10.5194/acp-16-8667-2016, https://www.atmos-chem-phys.net/16/8667/2016/
"""
from __future__ import absolute_import, division, print_function

# %%
import logging

# %%
import numpy as np
import pandas as pd
import xarray as xr

from numpy.polynomial.polynomial import polyval

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


def days_since_Jul01(ts):
    """ Days since July 01 of the given year

    Used to center the NOy seasonal distributions for the NH.
    SH starts on Jan 01 and `dayofyear` can be used as-is.

    Parameters
    ----------
    ts: `xr.DataArray` of datetime
        The times for which to calculate the number of days
        since July 01 of the respective year.

    Returns
    -------
    da: `xr.DataArray`
        The number of whole days since July 1 with the same
        dimensions as `ts`.
    """
    return xr.where(
        ts.dt.month < 7,
        ts.dt.dayofyear + 184,
        (
            # time difference is in nanoseconds
            ts - pd.to_datetime([f"{y.values}-07-01" for y in ts.dt.year])
        ).astype(int) // 86_400_000_000_000 + 1,  # convert to days.
    )


def Green_func_F16(t, Γ, Δ):
    """ "Inverse Gaussian" Green's function

    "Inverse Gaussian" type Green's function as described in
    Eq. (8) in Funke et al., 2016.

    Parameters
    ----------
    t: float of array_like
        The time(s) for which to calculate the Green's function,
        same units as Γ.
    Γ: float
        The mean (transport) time of the distribution.
    Δ: float
        The width of the distribution.

    Returns
    -------
    G: float or array_like
        The "Inverse Gaussian" Green's function with the given parameters
        evaluated at `t`.
    """
    nrm = np.sqrt(Γ**3 / (4.0 * np.pi * Δ**2 * t**3))
    nexpon = Γ * (t - Γ)**2 / (4.0 * Δ**2 * t)
    ret = nrm * np.exp(-nexpon)
    return np.where(t > 0.0, ret, 0.0)


def Green_filter_F16(ts, Γ, Δ, axis=-1):
    r""" Normalized "Inverse Gaussian" Green's function filter

    Normalized "Inverse Gaussian" type Green's function filter such that
    \int_{-\infty}^{t} G(t') dt' = 1.

    See Eq. (7) in Funke et al., 2016.

    Parameters
    ----------
    ts: float or array_like
        The time(s) for which to calculate the Green's function,
        same units as Γ. Should be equally spaced for correct normalizaion.
    Γ: float
        The mean (transport) time of the distribution.
    Δ: float
        The width of the distribution.
    axis: int, optional (default: -1)
        In case `t` is multi-dimensional, indicate the axis that should be
        used for filtering and hence normalization.

    Returns
    -------
    Gs: float or array_like
        The normalized "Inverse Gaussian" Green's function filter with the
        given parameters evaluated at `ts`.
    """
    f = Green_func_F16(ts, Γ, Δ)
    n = np.nansum(f, axis=axis)
    return f / n


def Green_filter_ubc(ts, Γ, Δ, axis=-1):
    r""" Normalized "Inverse Gaussian" Green's function filter

    Normalized "Inverse Gaussian" type Green's function filter
    as implemented in the UBC IDL code, therein with
    Δ = (\sqrt(0.7 * Γ) + 6.0) / \sqrt(2) = \sqrt(0.35 * Γ) + 4.24

    See Eq. (7) in Funke et al., 2016.

    Parameters
    ----------
    ts: float or array_like
        The time(s) for which to calculate the Green's function,
        same units as Γ. Should be equally spaced for correct normalizaion.
    Γ: float
        The mean (transport) time of the distribution.
    Δ: float
        The width of the distribution.
    axis: int, optional (default: -1)
        In case `t` is multi-dimensional, indicate the axis that should be
        used for filtering and hence normalization.

    Returns
    -------
    Gs: float or array_like
        The normalized "Inverse Gaussian" Green's function filter with the
        given parameters evaluated at `ts`.
    """
    f = ts**-1.5 * np.exp(-(ts - Γ)**2 / (4. * Δ**2 * ts / Γ))
    f = np.where(ts > 0.0, f, 0.0)
    n = np.nansum(f, axis=axis)
    return f / n


def Nm_func_F16(t, Nm, wm, tm):
    """ Time-variable NOy amount

    Time-dilated NOy amount from maximum value with finite lifetime.

    See Eq. (9) in Funke et al., 2016.

    Parameters
    ----------
    t: float or array_like
        The time(s) for which to calculate the NOy amount for,
        day of year for SH, days since 01. Jul for NH.
    Nm: float
        The maximum NOy amount per season.
    wm: float
        The width of the NOy distribution.
    tm: float
        The occurence day (of year) of the seasonal NOy maximum.

    Returns
    -------
    noy: float or array_like
        The distribution of the daily NOy.
    """
    expon = np.exp(-wm * (t - tm))
    return 4.0 * Nm * expon / (1 + expon)**2


def time_matrix(ts, dw=250):
    """Time-matrix for convolution

    The result matrix contains dw columns for consecutive days (backwards)
    for each t in ts (rows).

    Parameters
    ----------
    ts: float or array_like (N,)
        The time(s) for which the window times should be returned.

    Returns
    -------
    tm: `xr.DataArray` of shape (N, dw)
        The time matrix for selecting the Ap indices going back in time
        `dw` days. The first dimension is the original time dimension,
        the second the shifted times for the filter.
    """
    dl0 = np.arange(dw, dtype=float)
    tm = xr.DataArray(
        ts[:, None] - pd.to_timedelta(dl0, unit="d").to_numpy(),
        # use underscore to not confuse with an existing `time` dimension
        dims=["_time", "filter"],
        coords={"_time": ts},
    )
    return tm


def noy_ese(t, p, nn, wn, tn, ap_da, avtype="daily", dw=250, xtype="dens"):
    """Elevated-Stratopause parametrization of NOy

    Parameters
    ----------
    t: float or array_like
        The time(s) of the ESE.
    p: float or array_like
        The pressure level(s) for which to calculate the NOy amount for.
    nn: float
        The maximum NOy amount per NH season.
    wn: float
        The width of the NH NOy distribution.
    tn: float
        The occurence day (since Jul 01) of the NH NOy maximum.
    ap_da: `xr.DataArray`
        DataArray containing the Ap time series.
    avtype: str, optional (default: daily)
        Type of averaging, oprions: "daily" or "average".
    dw: int, optional (default: 250)
        Window size (in days) for Ap averaging.
    xtype: str, optional (default: "dens")
        Type of variable, "dens" for density and "flux" for flux.

    Returns
    -------
    ese: array_like
        The hemispheric NOy amount in units of `nn` (typically Gmol / km);
        variable length from onset day to 324 days after July 1.
    """
    #; vertical time lag variation
    tm_poly = [ 62.7637, 23.3374, 3.34175, 0.2589, 0.0106088]
    #; vertical flux variation
    fm_poly = [ 0.357087, -0.239236, 0.00420932, 0.0105685, 0.00107633 ]
    #; vertical wbar variation
    wm_poly = [ -1.69674, -0.493714, +0.151089, +0.00082302, -0.0139315, -0.000871843, +0.000161791 ]

    dn = days_since_Jul01(t).values.astype(int)
    dl = np.arange(dw, dtype=float)
    dl[0] = 0.5
    ies = 0
    lp = np.log(p)             #; log pressure levels
    tm = polyval(lp, tm_poly)  #; vertical time lag variation
    #; variation at equinox transition
    tm = np.minimum(tm + np.exp((tm + dn - 279.) / 4.), 270.)
    #; seasonal dependence of amount at source region
    # Funke et al., 2016, (21) with 0.0075 * 4 = 0.03
    xu = Nm_func_F16(dn, 0.0075, 0.046, 173.)
    #; seasonal dependence of ESE wbar at source region
    # Funke et al., 2016, (21) with 1.25 * 4 = 5
    wu = Nm_func_F16(dn, 1.25, 0.043, 173.)
    fm = polyval(lp, fm_poly)  #; vertical flux variation
    #; scale with source region amount*wbar, consider equinox transition
    fm = np.maximum(fm / (1. + np.exp((tm + dn - 273.) / 8.)) * xu * wu, 0.)
    wm = np.exp(polyval(lp, wm_poly))  #; vertical wbar variation
    #; scale with source region wbar, consider equinox transition
    wm = wm / (1. + np.exp((tm + dn - 280) / 9.)) * wu
    xb = Nm_func_F16(dn + tm.astype(int), nn, wn, tn)
    we = 0.15
    if xtype == "dens": nne = fm / wm - xb
    if xtype == "flux": nne = fm - xb
    if avtype != "average":
        filtere = Green_filter_F16(dl, tm, (np.sqrt(0.7 * tm) + 6.0) / np.sqrt(2))
    xe = np.zeros(324 - dn, dtype=float)
    tl = dn + tm
    it = 0
    while (dn + it < 324): #  and it < n_elements(tim)):         #; convolve with Ap
        if it - ies < tm:
            rfac = ((it - ies) / tm)**0.3
        else:
            rfac = 1.  #;fade in after ESE onset
        if dn + it > 304:
             rfac = rfac * ((324 - dn - it)/20.)**0.5      #; fade out after 1st May
        sease = rfac * Nm_func_F16(dn + it, nne, we, tl)
        if avtype == "average":
            xe[it] = sease * ap_da.sel(time=t.values)
        else:
           apts = time_matrix((t + pd.to_timedelta(it, unit="D")).values, dw=dw)
           aph = ap_da.sel(time=apts).values.T
           xe[it] = xe[it] + (sease * filtere.dot(aph))[0]
        it = it + 1
    return xe
