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
"""Plotting helper functions for zonal means and correlations

Provides plotting functions to show the NOy, CH4, and CO
histograms and zonal means.
"""
from __future__ import absolute_import, division, print_function

# %%
import numpy as np
import xarray as xr

# %%
import matplotlib.pyplot as plt


# %%
def plot_corr_hist(hist_sds, ch4_var, noy_var, cmap="jet", min_pts=0):
    # ch4_binv = ch4_var + "_bins"
    noy_binv = noy_var + "_bins"
    # contour levels
    lx = np.exp(np.arange(21) * 0.25 + 15) / np.exp(20) * 0.3

    hist_da = hist_sds["histogram"]
    _hist_da = xr.where(hist_da >= min_pts, hist_da, 0.)
    _hist_mean = hist_sds["mean"]
    _hist_median = hist_sds["median"]
    _hist_mode = hist_sds["mode"]

    hist_da2 = xr.where(
        _hist_da.sum(noy_binv) > 20.,
        _hist_da / _hist_da.sum(noy_binv) * 2.,
        0.,
    )
    hist_da2.attrs = hist_da.attrs
    hist_da2.attrs["long_name"] = "fraction of data points"
    lmaxmin = np.log(max(hist_da2.min().values, 0.02))
    lmax = np.log(min(hist_da2.max().values, 0.25))
    lx = np.exp(
        np.arange(21) * (lmax - lmaxmin) / 20
        + lmaxmin
    )

    fig, ax = plt.subplots()
    (hist_da2).where(hist_da2 > 0.0001).plot.contourf(
        y=noy_binv,
        ax=ax,
        cmap=cmap,
        levels=lx,
        robust=True,
    )
    _cs = (hist_da2).plot.contour(
        y=noy_binv,
        ax=ax,
        colors="k",
        levels=np.arange(20) * 0.1 + 0.1,
        linewidths=1.0,
        robust=True,
    )
    ax.clabel(_cs)
    _hist_mean.plot(ax=ax, color="C0", ls="-", label="mean")
    # _hist_mean.plot(ax=ax, color="w", ls=":")
    _hist_mode.plot(ax=ax, color="C1", ls="-", label="mode")
    # _hist_mode.plot(ax=ax, color="w", ls=":")
    _hist_median.plot(ax=ax, color="C2", ls="-", label="median")
    # _hist_median.plot(ax=ax, color="w", ls=":")
    # _hist_std = hist_sds["std"]
    #ax.fill_between(
    #    _hist_std[ch4_binv],
    #    _hist_mean - _hist_std,
    #    _hist_mean + _hist_std,
    #    color="C0",
    #    alpha=0.333,
    #)
    ax.set_xlim((-0.05, 1.6))
    ax.set_ylim((-0.005, 0.03))
    ax.minorticks_on()
    ax.grid(False, which="minor")
    ax.legend();
    if "latitude" in hist_da.coords:
        # mean just in case there are more than one
        if hist_da.latitude.mean() <= 0:
            ax.set_title("SH")
        else:
            ax.set_title("NH")
    return fig


# %%
def plot_day_zms(
    noy_epp, noy_bg, co, ch4,
    co_thresh=None,
    aslice=(None, None), lslice=(None, None), cmap="jet",
):
    def plot_co_line(ax, thresh=co_thresh):
        if thresh is None:
            return
        co_sel = co.sel(altitude=slice(*aslice), latitude=slice(*lslice))
        co_sel.plot.contour(
            y="altitude",
            ax=ax,
            colors="w",
            levels=[thresh],
            linewidths=1.0,
        )
        (-co_sel).plot.contour(
            y="altitude",
            ax=ax,
            colors="k",
            levels=[-thresh],
            linewidths=1.0,
        )
        return

    fig, axs = plt.subplots(1, 4, sharey="row", figsize=(14, 4))

    # EPP NOy
    ax = axs[0]
    noy_epp.sel(
        altitude=slice(*aslice),
        latitude=slice(*lslice)
    ).plot.contourf(
        y="altitude",
        levels=np.arange(0, 0.0155, 0.0005) * 1e3,
        #levels=np.arange(-0.0015, 0.0155, 0.0005) * 1e3,
        cmap=cmap,
        ax=ax,
    )
    plot_co_line(ax, thresh=co_thresh)
    ax.set_title("EPP NOy")

    # background NOy
    ax = axs[1]
    noy_bg.sel(
        altitude=slice(*aslice),
        latitude=slice(*lslice)
    ).plot.contourf(
        y="altitude",
        levels=np.arange(0, 0.0155, 0.0005) * 1e3,
        #levels=np.arange(-0.0015, 0.0155, 0.0005) * 1e3,
        cmap=cmap,
        ax=ax,
    )
    plot_co_line(ax, thresh=co_thresh)
    ax.set_title("BG NOy")
    ax.set_ylabel("")

    # CO on log-scale
    ax = axs[2]
    co.sel(
        altitude=slice(*aslice),
        latitude=slice(*lslice)
    ).plot.contourf(
        y="altitude",
        norm=plt.matplotlib.colors.LogNorm(0.01, 1),
        levels=np.logspace(-2, 0, 33),
        cbar_kwargs=dict(format="%g", ticks=[0.01, 0.1, 1]),
        cmap=cmap,
        ax=ax,
    )
    plot_co_line(ax, thresh=co_thresh)
    ax.set_title("CO")
    ax.set_ylabel("")

    # CH4
    ax = axs[3]
    ch4.sel(
        altitude=slice(*aslice),
        latitude=slice(*lslice)
    ).plot.contourf(
        y="altitude",
        levels=np.arange(0, 0.51, 0.01),
        # cbar_kwargs=dict(ticks=plt.matplotlib.ticker.MultipleLocator(0.1)),
        cbar_kwargs=dict(ticks=np.arange(0, 0.51, 0.1)),
        cmap=cmap,
        ax=ax,
    )
    plot_co_line(ax, thresh=co_thresh)
    ax.set_title("CH$_4$")
    ax.set_ylabel("")

    # fig.suptitle(date)
    return fig

