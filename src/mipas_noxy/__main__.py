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
"""MIPAS NOxy calculator command line interface
"""
import logging

import numpy as np
import xarray as xr

import click
import tomli

from .util import *

logging.basicConfig(
    level=logging.INFO,
    format=
        "[%(levelname)-8s] (%(asctime)s) "
        "%(filename)s:%(lineno)d:%(funcName)s() %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
)


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], show_default=True)


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
    xr.set_options(keep_attrs=True)
    xr.set_options(display_max_rows=1000, display_width=96)
    np.set_printoptions(precision=3)

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
        config = tomli.load(_fp)
    debug("config: %s", config)

    # Time setup, command line takes precedence,
    # but works only if both year and month are given.
    if year and month:
        tymss = [(year, month)]
    else:
        times = config.get("times", [])
        tyms = [
            set((_t.year, _t.month) for _t in xr.date_range(**t))
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
    out_targets = out_conf.get("targets")
    if not hasattr(out_targets, "__getitem__"):
        out_targets = [out_targets]

    debug("input config: %s", inp_conf)
    debug("output config: %s", out_conf)
    info("output targets: %s", out_targets)

    for (year, month) in tymss:
        for out_target in out_targets:
            info("Processing: %s %s %s", out_target, year, month)

            # read species netcdfs into list
            mv8_noy_l = read_mv8_species_v1(config, out_target, year, month)
            debug("MIPAS v8 input list: %s", mv8_noy_l)

            # combine into one data set
            mv8_noy_ds = combine_NOxy(mv8_noy_l)
            debug("MIPAS v8 combined ds: %s", mv8_noy_ds)

            # sum/average to NOx/NOy
            mv8_noy = calculate_NOxy(mv8_noy_ds)
            debug("MIPAS v8 NOx/NOy ds: %s", mv8_noy)

            # drop unneeded variables
            mv8_noy1 = mv8_noy.drop_vars(
                ["weights", "retrieval_in_logarithmic_parameter_space"]
            )

            # Fixup to make the dataset comliant with
            # the original v8 netcdf data sets.
            mv8_noy1 = fixup_target_name(
                mv8_noy1,
                inp_conf[out_target]["targets"][0],
                out_target.upper(),
            )
            mv8_noy1 = fixup_altitudes(mv8_noy1)
            info("MIPAS v8 NOx/NOy ds fixed: %s", mv8_noy1)

            # Construct file/path name for output species
            out_path = get_nc_filename(
                out_conf.get("path", "."),
                config.get("resolution", "R"),
                out_target.upper(),  # upper case in filename
                year,
                month,
                version=inp_conf.get(out_target, {}).get("versions", [])[0],
            )
            info("Saving to: %s", out_path)

            if dry_run:
                info("Dry run, writing skipped.")
            else:
                mv8_noy1.to_netcdf(out_path)
                info("Saved.")

    return 0


if __name__ == "__main__":
    main()
