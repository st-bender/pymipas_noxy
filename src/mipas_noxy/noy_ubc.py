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
    ds: `xr.DataArray`
    """
    return xr.where(
        ts.dt.month < 7,
        ts.dt.dayofyear + 184,
        (
            # time difference is in nanoseconds
            ts - pd.to_datetime([f"{y.values}-07-01" for y in ts.dt.year])
        ).astype(float) / 86_400_000_000_000 + 1,  # convert to days.
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
    nrm = np.sqrt(Γ**3 / (4 * np.pi * Δ**2 * t**3))
    nexpon = 0.25 * Γ * (t - Γ)**2 / (Δ**2 * t)
    return nrm * np.exp(-nexpon)


def Green_filter_F16(ts, Γ, Δ, axis=-1):
    """ Normalized "Inverse Gaussian" Green's function filter

    Normalized "Inverse Gaussian" type Green's function filter such that
    \int_{-\infty}^{t} G(t') dt' = 1.

    See Eq. (7) in Funke et al., 2016.

    Parameters
    ----------
    ts: float of array_like
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
    """ Normalized "Inverse Gaussian" Green's function filter

    Normalized "Inverse Gaussian" type Green's function filter
    as implemented in the UBC IDL code, therein with
    Δ = (\sqrt(0.7 * Γ) + 6.0) / \sqrt(2) = \sqrt(0.35 * Γ) + 4.24

    See Eq. (7) in Funke et al., 2016.

    Parameters
    ----------
    ts: float of array_like
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
    n = np.nansum(f, axis=axis)
    return f / n


def Nm_func_F16(t, Nm, wm, tm):
    """ Time-variable NOy amount

    Time-dilated NOy amount from maximum value with finite lifetime.

    See Eq. (9) in Funke et al., 2016.

    Parameters
    ----------
    t: float of array_like
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
