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
"""MIPAS EPP-NOy calculator command line interface
"""

# %%
from os import path

# %%
import logging

# %%
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages as ppdf

# %%
import numpy as np
import pandas as pd
import xarray as xr

# %%
import click
import toml

# %%
from .epp_noy import integrate_eppnoy, process_day_multi2
from .units import convert_to_ppm
from .util import (
    read_mv8_species_v1,
    compat_drop_vars,
    fixup_altitudes,
    fixup_target_name,
    get_nc_filename,
)
from .helpers import (
    calc_Ntot,
    calc_noy_bg_epp,
    calc_zms,
    plot_corr_hist,
    plot_day_zms,
    potential_temperature,
    select_common_data,
    smooth_targets,
)

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
def _file_suffix(config):
    akm_thresh = config.get("akm_thresh", None)
    ch4_thresh = config.get("ch4_thresh", None)
    co_thresh = config.get("co_thresh", None)
    tpot_thresh = config.get("tpot_thresh", None)
    suffix = ""
    if co_thresh is not None:
        suffix += f"_co{co_thresh * 100:03.0f}"
    if akm_thresh is not None:
        suffix += f"_ak{akm_thresh * 1000:04.0f}"
    if ch4_thresh is not None:
        suffix += f"_ch4{ch4_thresh * 1000:04.0f}"
    if tpot_thresh is not None:
        suffix += f"_tpot{tpot_thresh:04.0f}"
    return suffix


# %%
def _setup_bins(cbin_conf):
    # cbin_conf = corr_conf.get("bins", {})
    # CH4 bins
    ch4_bin_width = cbin_conf.get("ch4_bin_width", 3e-8 * 1e6)
    ch4_bin_min = cbin_conf.get("ch4_bin_min", -1e-7 * 1e6)
    ch4_nbins = cbin_conf.get("ch4_nbins", 47)
    # correction factor (num / den)
    ch4_bin_den = cbin_conf.get("ch4_bin_den", 1.0)
    ch4_bin_num = cbin_conf.get("ch4_bin_num", 1.0)
    # NOy bins
    noy_bin_width = cbin_conf.get("noy_bin_width", 5e-10 * 1e6)
    noy_bin_min = cbin_conf.get("noy_bin_min", -2e-8 * 1e6)
    noy_nbins = cbin_conf.get("noy_nbins", 100)
    # correction factor (num / den)
    noy_bin_den = cbin_conf.get("noy_bin_den", 1.0)
    noy_bin_num = cbin_conf.get("noy_bin_num", 1.0)
    # IDL's findgen starts at 0 as well
    ch4_bin_edges = np.arange(ch4_nbins + 1) * ch4_bin_width + ch4_bin_min
    ch4_bin_edges *= ch4_bin_num / ch4_bin_den
    noy_bin_edges = np.arange(noy_nbins + 1) * noy_bin_width + noy_bin_min
    noy_bin_edges *= noy_bin_num / noy_bin_den
    return ch4_bin_edges, noy_bin_edges


# %%
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], show_default=True)


# %%
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-c",
    "--config-file",
    default=None,
    help="Configuration file to use, takes precedence over the argument.",
)
@click.argument("configfile", required=False)
@click.option(
    "-m", "--month", default=None, type=int, help="month to process, needs also `year`."
)
@click.option(
    "-y", "--year", default=None, type=int, help="year to process, needs also `month`."
)
# Do not write output file
@click.option(
    "-n", "--dry-run", default=False, is_flag=True, help="Dry run, no output written."
)
# logging options
@click.option(
    "-v", "--verbose", "loglevel", flag_value="INFO", help="Sets 'loglevel' to 'INFO'."
)
@click.option(
    "-q", "--quiet", "loglevel", flag_value="ERROR", help="Sets 'loglevel' to 'ERROR'."
)
@click.option(
    "-l",
    "--loglevel",
    default="WARN",
    help="Sets 'loglevel' directly to the given level.",
    type=click.Choice(["DEBUG", "INFO", "WARN", "ERROR"], case_sensitive=False),
)
def main(
    config_file,
    configfile,
    month,
    year,
    dry_run,
    loglevel,
):
    try:
        xr.set_options(keep_attrs=True)
        xr.set_options(display_max_rows=1000)
    except ValueError:
        pass
    np.set_printoptions(precision=6)

    # %%
    xr.set_options(display_width=100)
    xr.set_options(cmap_divergent="PuOr_r", cmap_sequential="cividis")

    # %%
    logger = logging.getLogger()
    debug = logger.debug
    info = logger.info
    warn = logger.warning
    error = logger.error
    info("setting loglevel: %s", loglevel)
    logger.setLevel(getattr(logging, loglevel.upper(), None))
    warn("new loglevel: %s", logging.getLogger().getEffectiveLevel())

    if not config_file:
        if not configfile:
            error(
                "No config file specified, "
                "use `--config-file` or pass as argument."
            )
            return -1
        else:
            config_file = configfile

    with open(config_file, mode="rb") as _fp:
        config = toml.loads(_fp.read().decode())
    debug("config: %s", config)

    # Time setup, command line takes precedence,
    # but works only if both year and month are given.
    if year and month:
        tymss = [(year, month)]
    else:
        times = config.get("times", [])
        tyms = [
            set((_t.year, _t.month) for _t in pd.date_range(**t))
            for t in times
        ]
        if not tyms:
            error(
                "No times specified, either on the cmmand line "
                "via `--year` and `--month`, or by using "
                "`[[times]]` entries in the configuration file."
            )
            return -2
        else:
            tymss = sorted(set.union(*tyms))
    info("year(s)/month(s): %s", tymss)

    # Short-cut config dictionaries
    inp_conf = config.get("input", {})
    out_conf = config.get("output", {})
    out_path = out_conf.get("path", ".")
    out_targets = out_conf.get("targets", [])
    if not hasattr(out_targets, "__getitem__") or isinstance(out_targets, str):
        out_targets = [out_targets]
    out_files = out_conf.get("datasets", [])

    debug("input config: %s", inp_conf)
    debug("output config: %s", out_conf)
    info("output targets: %s", out_targets)

    corr_conf = config.get("correlation", {})
    cthr_conf = corr_conf.get("thresholds", {})
    cbin_conf = corr_conf.get("bins", {})
    ch4_bin_edges, noy_bin_edges = _setup_bins(cbin_conf)
    debug("CH4 bins: %s", ch4_bin_edges)
    debug("NOy bins: %s", noy_bin_edges)

    regions = corr_conf.get("region", [])
    if not hasattr(regions, "__getitem__") or isinstance(regions, str):
        regions = [regions]

    # plot config
    plot_conf = config.get("figures", {})
    fig_fmt = plot_conf.get("format", "pdf")
    fig_path = plot_conf.get("path", ".")
    fig_suff = plot_conf.get("suffix", "")
    fig_type = plot_conf.get("types", [])
    if fig_suff == "auto":
        fig_suff = _file_suffix(cthr_conf)
    # zonal means
    zm_alts = plot_conf.get("plot_alts", (None, None))
    zm_lats = plot_conf.get("plot_lats", (-90, 90))
    zm_cmap = plot_conf.get("cmap", "jet")
    zm_co_thresh = plot_conf.get("co_thresh", None)
    zm_dlat = plot_conf.get("zm_dlat", 5.0)
    plt.style.use(plot_conf.get("style", None))

    for (year, month) in tymss:
        date = f"{year:04d}{month:02d}"
        for out_target in out_targets:
            info("Processing: %s %s %s", out_target, year, month)
            out_target_conf = config.get(out_target, {})

            # read species netcdfs into list
            mv8_noy_l = read_mv8_species_v1(config, out_target, year, month)
            debug("MIPAS v8 input list: %s", mv8_noy_l)
            if not mv8_noy_l:
                continue

            mv8_noy, mv8_ch4, mv8_co = mv8_noy_l
            mv8_noy = mv8_noy.isel(species=0)
            mv8_ch4 = mv8_ch4.isel(species=0)
            mv8_co = mv8_co.isel(species=0)
            # drop unneeded variables
            mv8_noy = compat_drop_vars(
                mv8_noy,
                ["weights", "retrieval_in_logarithmic_parameter_space"],
            )
            mv8_noy_id, mv8_ch4_id, mv8_co_id = select_common_data(mv8_noy, mv8_ch4, mv8_co)

            smooth_vr = inp_conf[out_target].get("smooth_vr", None)
            info("vertical resolution smoothing: %s.", smooth_vr)
            smooth_ch4 = smooth_targets(mv8_noy_id, mv8_ch4_id, smooth_vr=smooth_vr)
            # smooth_ch4 = mv8_ch4_id.target.interp(altitude=mv8_noy_id.altitude)
            smooth_ch4 = smooth_ch4.rename("ch4_vmr")
            smooth_ch4.attrs = mv8_ch4.target.attrs
            debug("smooth_ch4: %s", smooth_ch4)
            # interpolate CH4 averaging kernel diagonal for data selection
            smooth_ch4akd = mv8_ch4_id.akm_diagonal.interp(
                altitude=mv8_noy_id.altitude,
                kwargs=dict(bounds_error=False, fill_value="extrapolate"),
            )
            smooth_ch4akd = smooth_ch4akd.rename("ch4_akd")
            smooth_ch4akd.attrs = mv8_ch4.akm_diagonal.attrs
            debug("smooth_ch4akd: %s", smooth_ch4akd)

            # smooth_co = smooth_targets(mv8_noy_id, mv8_co_id)
            smooth_co = mv8_co_id.target.interp(
                altitude=mv8_noy_id.altitude,
                kwargs=dict(bounds_error=False, fill_value="extrapolate"),
            )
            smooth_co = smooth_co.rename("co_vmr")
            smooth_co.attrs = mv8_co.target.attrs
            debug("smooth_co: %s", smooth_co)

            combined = mv8_noy_id.copy()
            combined["vmr_ch4"] = smooth_ch4
            combined["akd_ch4"] = smooth_ch4akd
            combined["vmr_co"] = smooth_co
            combined = combined.rename({"target": "vmr_noy"})
            combined["vmr_noy"].attrs["standard_name"] = "mole_fraction_of_noy_expressed_as_nitrogen_in_air"
            combined = convert_to_ppm(combined)
            combined["noy_vs_ch4"] = combined["vmr_noy"] / combined["vmr_ch4"]
            combined["noy_vs_ch4"].attrs = {"long_name": "NOy to CH4 ratio", "units": "1"}
            # Use p and T from the CH4 data set
            combined["pressure"] = mv8_ch4_id.pressure.interp(
                altitude=mv8_noy_id.altitude,
                kwargs=dict(bounds_error=False),
            ).fillna(mv8_noy_id.pressure)
            combined["temperature"] = mv8_ch4_id.temperature.interp(
                altitude=mv8_noy_id.altitude,
                kwargs=dict(bounds_error=False),
            ).fillna(mv8_noy_id.temperature)
            combined["T_pot"] = potential_temperature(
                combined.pressure,
                combined.temperature,
            )
            debug("combined: %s", combined)
            if "combined" in out_files:
                cnc_fname = f"{out_target}_combined_mipasv8_{date}.nc"
                cnc_fpname = path.join(out_path, cnc_fname)
                combined.time.encoding["units"] = "days since 2000-01-01"
                combined.to_netcdf(cnc_fpname) #, unlimited_dims=["geo_id"])
                info("Combined data set saved to: %s", cnc_fpname)

            bg_file = out_target_conf.get("bg_file", None)
            if bg_file is None:
                info("Calculating (monthly) NOy/CH4 background correlation.")
                h_dsl = []
                for reg in regions:
                    info("region: %s", reg)
                    ch4_noy_hist = calc_noy_bg_epp(
                        combined,
                        "vmr_ch4", "vmr_co", "vmr_noy",
                        ch4_bin_edges, noy_bin_edges,
                        **reg,
                        **cthr_conf,
                        copy_vars=[],
                    )
                    h_ds = ch4_noy_hist.expand_dims(time=[combined.time.mean().values])
                    h_dsl.append(h_ds)
                hh_ds = xr.merge(h_dsl)
            else:
                info("Using NOy/CH4 background correlation from: %s", bg_file)
                hh_ds = xr.open_dataset(bg_file).load()

            # Subtract the background NOy
            noy_bg_epp_da = combined.groupby("time.date").apply(
                process_day_multi2,
                args=(hh_ds["mean"], "vmr_ch4", "vmr_co", "vmr_noy",),
                **out_target_conf.get("sub", {}),
            )
            if "vmr_ch4" in noy_bg_epp_da.coords:
                noy_bg_epp_da = noy_bg_epp_da.drop("vmr_ch4")
            if "latitude" in noy_bg_epp_da.coords:
                noy_bg_epp_da = noy_bg_epp_da.drop("latitude")

            if "epp_noy" in out_files:
                mv8_noy1 = mv8_noy_id.copy()
                mv8_noy1["target"] = noy_bg_epp_da
                mv8_noy1 = mv8_noy1.swap_dims({"geo_id": "time"}).reset_coords()
                # Fixup to make the dataset compliant with
                # the original v8 netcdf data sets.
                # Use the "name" if set.
                oname = out_target_conf.get(
                    "name",
                    "".join(_c for _c in out_target if _c not in "_-")
                )
                mv8_noy1 = fixup_target_name(
                    mv8_noy1,
                    inp_conf[out_target]["targets"][0],
                    oname.upper(),
                )
                mv8_noy1 = fixup_altitudes(mv8_noy1)
                mv8_noy1.attrs.update(out_target_conf.get("attrs", {}))
                mv8_noy1.attrs.update({
                    "date_created": pd.Timestamp.utcnow().isoformat()
                })
                debug("MIPAS v8 EPP-NOy ds fixed: %s", mv8_noy1)

                # Construct file/path name for output species
                out_path1 = get_nc_filename(
                    out_conf.get("path", "."),
                    config.get("resolution", "R"),
                    oname.upper(),  # upper case in filename
                    year,
                    month,
                    version=inp_conf.get(out_target, {}).get("versions", [])[0],
                )
                mv8_noy1.to_netcdf(out_path1)
                info("EPP-NOy data set saved to: %s", out_path1)

            # Combine with the original
            noy_bg_epp = xr.merge([
                combined,
                noy_bg_epp_da.rename("vmr_noy_epp"),
                (combined["vmr_noy"] - noy_bg_epp_da).rename("vmr_noy_bg"),
            ])
            debug("noy_bg_epp: %s", noy_bg_epp)

            # Figures
            # Histogram only if calculated
            if bg_file is None and "hist" in fig_type:
                ph_ds = hh_ds.isel(time=0).reset_coords()
                for ir, _ in enumerate(regions):
                    hist_fig = plot_corr_hist(
                        ph_ds.isel(latitude=ir),
                        "vmr_ch4", "vmr_noy",
                        cmap=zm_cmap,
                        min_pts=cthr_conf.get("min_pts", 0),
                    )
                    hist_fig.suptitle("MIPAS v8" + " " + date)
                    hist_fname = f"{out_target}_hist_mipasv8_{date}_{zm_cmap}{fig_suff}_reg{ir}.{fig_fmt}"
                    hist_fpname = path.join(fig_path, hist_fname)
                    hist_fig.savefig(hist_fpname)
                    info("Region %d histogram saved to: %s", ir, hist_fpname)

            # (Monhtly) zonal mean
            if "zm" in fig_type:
                zms = calc_zms(
                    noy_bg_epp,
                    dlat=zm_dlat,
                    dim="geo_id",
                    variable="vmr_noy",
                )
                fig = plot_day_zms(
                    zms["vmr_noy_epp"].to_unit("ppb"),
                    zms["vmr_noy_bg"].to_unit("ppb"),
                    zms["vmr_co"], zms["vmr_ch4"],
                    aslice=zm_alts,
                    lslice=zm_lats,
                    cmap=zm_cmap,
                    co_thresh=zm_co_thresh,
                )
                fig.suptitle("MIPAS v8" + " " + date)
                zm_fname = f"{out_target}_zm{zm_dlat:02.0f}_mipasv8_{date}_{zm_cmap}{fig_suff}.{fig_fmt}"
                zm_fpname = path.join(fig_path, zm_fname)
                fig.savefig(zm_fpname)
                info("Lat-Alt zonal mean saved to: %s", zm_fpname)

            if bg_file is None and "hist" in out_files:
                hnc_fname = f"{out_target}_hist_mipasv8_{date}{fig_suff}.nc"
                hnc_fpname = path.join(out_path, hnc_fname)
                hh_ds.time.encoding["units"] = "days since 2000-01-01"
                hh_ds.to_netcdf(hnc_fpname, unlimited_dims=["time"])
                info("Histogram statsitics saved to: %s", hnc_fpname)

            if "epp_noy_tot" in out_files:
                ntot_ds = noy_bg_epp.swap_dims({"geo_id": "time"}).resample(time="1d").apply(
                    calc_Ntot,
                    dlat=zm_dlat,
                    dim="time",
                )
                debug("daily NOy content ds: %s", ntot_ds)
                epp_noy_tot = ntot_ds.groupby("time").map(
                    integrate_eppnoy,
                    **out_target_conf.get("sum", {}),
                ).T
                epp_noy_tot = epp_noy_tot.to_unit("Gmol")
                info("Daily hemispheric EPP-NOy: %s", epp_noy_tot)
                tnc_fname = f"{out_target}_Ntot_mipasv8_{date}{fig_suff}.nc"
                tnc_fpname = path.join(out_path, tnc_fname)
                # convert to dataset for netcdf
                epp_noy_tot_ds = epp_noy_tot.to_dataset()
                epp_noy_tot_ds.time.encoding["units"] = "days since 2000-01-01"
                epp_noy_tot_ds.attrs["config"] = str(config)
                epp_noy_tot_ds.transpose("time", "altitude", "latitude").to_netcdf(
                    tnc_fpname, unlimited_dims=["time"],
                )
                info("Daily hemispheric EPP-NOy saved to: %s", tnc_fpname)

    return 0


if __name__ == "__main__":
    main()
