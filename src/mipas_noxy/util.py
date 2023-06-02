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
"""MIPAS NOx and NOy calculator utility functions

Utility functions to read MIPAS level-2 netcdf files and
to calculate NOx or NOy species from the individual species.
"""
from __future__ import absolute_import, division, print_function

from logging import info, debug
from os import path

import h5netcdf.legacyapi as nc4
import numpy as np
import xarray as xr

from scipy.io import netcdf_file


def get_nc_filename(l2_path, resolution, species, year, month, ext="nc", version="61"):
    """Infer the full path for the level-2 netcdf files
    """
    species_subdir = "V8{resolution}_{species}_{version}".format(
        resolution=resolution,
        species=species,
        version=version,
    )
    file = "MIPAS-E_IMK.{year:04d}{month:02d}.V8{resolution}_{species}_{version}.{ext}".format(
        year=year,
        month=month,
        resolution=resolution,
        species=species,
        version=version,
        ext=ext,
    )
    full_path = path.join(
        l2_path,
        species_subdir,
        file
    )
    return full_path


def open_mipas_l2(file, **kwargs):
    """Open and harmonize MIPAS level-2 netcdf file

    Opens the MIPAS level-2 netcdf file and harmonizes the
    altitude coordinates so that the dataset can be easily
    processed with `xarray`.
    """
    def _open_nc34(file):
        try:
            # try netcdf4 with h5netcdf first
            _ds = nc4.Dataset(file, mode="r")
        except (IOError, OSError):
            # apparently a netcdf3 file, use scipy.io
            _ds = netcdf_file(file, mmap=False)
        return _ds

    def _get_attrs(v):
        if hasattr(v, "attrs"):
            # h5netcdf netcdf4
            attrs = v.attrs
        else:
            # netCDF4 just in case
            attrs = v.__dict__
            if "_attributes" in attrs:
                # scipy.io netcdf3
                attrs = attrs["_attributes"]
        # at least scipy.io converts the attribute values to bytes,
        # converts them back to be consistent with the others.
        attrs.update({
            k: _v.decode() for k, _v in attrs.items()
            if type(_v) == bytes
        })
        return attrs

    # Drops the "altitude" variable since `xarray` can't handle
    # dimensions and variables with the same name but different dimensions
    mipv8_ds = xr.open_dataset(file, drop_variables=["altitude"], **kwargs)
    if "altgrid" in mipv8_ds.dims:
        mipv8_ds = mipv8_ds.rename({"altgrid": "altitude"})
    if "timegrid" in mipv8_ds.dims:
        mipv8_ds = mipv8_ds.swap_dims({"timegrid": "time"})
    # harmonize dtypes to avoid big/little endian mixup
    _dtype = mipv8_ds.target.dtype
    # Fix altitude naming and coordinates
    nc4_ds = _open_nc34(file)
    _alt = xr.DataArray(
        nc4_ds.variables["altitude"][:].astype(_dtype),
        dims=("altitude", "time"),
        coords={
            "altitude": nc4_ds.variables["altitude"][:, 0].astype(_dtype),
            "time": mipv8_ds.time
        },
    )
    # Re-creates the original "altitude" variable but with a different name
    mipv8_ds["alt"] = _alt
    # Copies the attributes for the altitudes (now both of them)
    alt_attrs = _get_attrs(nc4_ds.variables["altitude"])
    mipv8_ds["alt"].attrs = alt_attrs
    mipv8_ds["altitude"].attrs = alt_attrs
    return mipv8_ds


def interp_altitude(ds, interp_error=False, **kwargs):
    ret = ds.interp(**kwargs)
    if interp_error:
        # overwrite `target_noise_error` with the squared interpolation
        ret["target_noise_error"] = np.sqrt(
            (ds.target_noise_error**2).interp(**kwargs)
        )
    return ret


def _read_mipas_species(file, _s, _w, interp_alts=None, load_kwargs=None):
    debug("%s %s", file, path.exists(file))
    with open_mipas_l2(file, **load_kwargs) as _ds:
        _ds = _ds.expand_dims("species").assign_coords(species=[_s])
        _ds["weights"] = ("species", [_w])
        log_retr = (
            _ds.attrs["retrieval_in_logarithmic_parameter_space"].lower()
            in ["1", "true", "yes"]
        )
        _ds["retrieval_in_logarithmic_parameter_space"] = ("species", [log_retr])
    return _ds


def read_mv8_species_v1(config, target, year, month, load_kwargs=None):
    load_kwargs = load_kwargs or {}
    _res = config.get("resolution", "R")

    input_conf = config.get("input", {})
    input_path = input_conf.get("path")

    spec_conf = input_conf[target]
    targets = spec_conf.get("targets", [])
    versions = spec_conf.get("versions", [])
    weights = spec_conf.get("weights", [])

    # iterate over the targets
    dsl = []
    for _s, _v, _w in zip(targets, versions, weights):
        info("species: %s, version: %s, weight: %s", _s, _v, _w)
        _file = get_nc_filename(input_path, _res, _s, year, month, version=_v)
        if not path.exists(_file):
            return None
        _ds = _read_mipas_species(_file, _s, _w, load_kwargs=load_kwargs)
        dsl.append(_ds)
    return dsl


def combine_NOxy(dsl, species=None):
    # order "species" as in the list, or as given as an argument
    if species is None:
        species = [_d.species.values[0] for _d in dsl]
    _ds = xr.merge(dsl)
    _ds = _ds.sel(species=species)
    _dsb = _ds.dropna(dim="time", how="any", subset=["geo_id"])

    geo_id_set = set(_ds.geo_id.values[0])
    for gids in _ds.geo_id.values[1:]:
        geo_id_set = geo_id_set.intersection(set(gids))
    # Filters out NaNs by type "float" since using `np.nan` didn't work.
    # Ensures that the `byte` list can be sorted and compared
    # to assert that we have indeed all the available geo_ids
    # across all data sets in the combined data set.
    geo_id_set = list(filter(lambda o: type(o) != float, geo_id_set))
    assert (sorted(geo_id_set) == _dsb.geo_id).all()
    # fix dtypes
    for _v in dsl[0].data_vars:
        _dsb[_v] = _dsb[_v].astype(dsl[0][_v].dtype)
    return _dsb


def calculate_NOxy(ds, dim="species", keep_attrs=True):
    scaled_vars = ["target"]
    squared_vars = ["target_noise_error"]
    dof_vars = ["chi2", "rms"]
    akm_vars = ["akm_diagonal", "vr_akdiag", "vr_col", "vr_row"]

    _dsvars = set(ds.data_vars)

    # copy first dataset and update later
    mv8_noxy = ds.isel({dim: 0}).drop(dim).copy()

    # overwrite degrees of freedom with the sum
    mv8_noxy["dof"] = ds.dof.sum(dim=dim)

    # update scaled (weighted) variables with the weighted (scaled) sum
    # usually: weights = number of N (nitrogen) atoms in molecule
    for _v in _dsvars & set(scaled_vars):
        mv8_noxy[_v] = (ds[_v] * ds.weights).sum(dim=dim)

    # update "squared" variables with the sqrt(sum of squares)
    # e.g. error variables
    for _v in _dsvars & set(squared_vars):
        mv8_noxy[_v] = np.sqrt(((ds[_v] * ds.weights)**2).sum(dim=dim))

    # update dof-weighted variables with the dof-weighted average
    for _v in _dsvars & set(dof_vars):
        mv8_noxy[_v] = (ds[_v] * ds.dof).sum(dim=dim) / mv8_noxy.dof

    # update akm-like variables with the target-weighted average
    ak_ww = ds.weights * np.abs(ds.target)
    ak_w = ak_ww / ak_ww.sum(dim=dim)
    for _v in _dsvars & set(akm_vars):
        mv8_noxy[_v] = (ds[_v] * ak_w).sum(dim=dim)

    for _v in mv8_noxy.data_vars:
        # fix dtype
        mv8_noxy[_v] = mv8_noxy[_v].astype(ds[_v].dtype)
        if keep_attrs:
            # copy attributes
            mv8_noxy[_v].attrs = ds[_v].attrs

    return mv8_noxy


# from v81, 25 Apr 2023 of
# https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html
STD_NAMES = {
	"NOX": "mole_fraction_of_nox_expressed_as_nitrogen_in_air",
	"NOY": "mole_fraction_of_noy_expressed_as_nitrogen_in_air",
}


def fixup_target_name(ds, from_name, to_name):
    # We are going to overwrite things, so make a copy first.
    ds = ds.copy()
    # replace in variable attributes
    for _v in ds.data_vars:
        _attrs = ds[_v].attrs
        _ln = _attrs.get("long_name", None)
        _sn = _attrs.get("standard_name", None)
        if _ln and from_name in _ln:
            _attrs["long_name"] = _ln.replace(from_name, to_name)
        if _sn and _sn.startswith("mole_fraction"):
            _attrs["standard_name"] = STD_NAMES.get(to_name.upper(), _sn)
    # replace in global attributes
    for _k, _v in ds.attrs.items():
        if from_name in _v:
            ds.attrs[_k] = _v.replace(from_name, to_name)
    return ds


def compat_drop_vars(ds, var):
    """Backwards-compatible dropping of vairbales in the dataset

    Just in case one or the other does not exist or gets removed
    from the API.
    """
    try:
        ds = ds.drop_vars(var)
    except AttributeError:
        ds = ds.drop(var)
    return ds


def fixup_altitudes(ds, alt_name="alt"):
    if alt_name not in ds.data_vars:
        info(
            "Variable '%s' not found, skip resetting altitude coordinates",
            alt_name
        )
        return ds
    # drops the "altitude" coordinate variable first
    # if there is already one in the dataset
    if "altitude" in ds.coords:
        ds = compat_drop_vars(ds, ["altitude"])
    # the variable in `alt_name` should hold the
    # "original" time-dependent altitude coordinates,
    # rename it to be consistent with the input nc files.
    ds = ds.rename({alt_name: "altitude"})
    return ds
