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
"""Unitful calculations (with xarray)

Provides unit conversions and calculations, and overloads `xarray.DataArray`
multiplication and division to keep track of the units.
"""
from __future__ import absolute_import, division, print_function

# %%
import xarray as xr

# %%
from astropy import units as au


# %%
def convert_to_ppm(ds):
    """Convert 'vmr' variables to `ppm`
    """
    ret = ds.copy()
    for v in ds.data_vars:
        if v.startswith("vmr"):
            ret[v] = ds[v].to_unit("ppm")
    return ret


# %%
def convert_to_molmol(ds):
    """Convert 'vmr' variables to `mol/mol`
    """
    ret = ds.copy()
    for v in ds.data_vars:
        if v.startswith("vmr"):
            ret[v] = ds[v].to_unit("mol/mol")
    return ret


# %%
def _get_unit(x):
    """Infer unit(s) from attributes
    """
    if hasattr(x, "unit"):
        return x.unit
    elif hasattr(x, "units"):
        return au.Unit(x.units)
    return au.dimensionless_unscaled


# %%
def mul_u(a, b):
    """Multiplication with units
    """
    # ret = a * b
    ret = a._binary_op(b, xr.core._typed_ops.operator.mul)
    ret_u = _get_unit(a) * _get_unit(b)
    ret.attrs["units"] = ret_u.si
    return ret


# %%
def div_u(a, b):
    """Division with units
    """
    # ret = a * b
    ret = a._binary_op(b, xr.core._typed_ops.operator.truediv)
    ret_u = _get_unit(a) / _get_unit(b)
    ret.attrs["units"] = ret_u.si
    return ret


# %%
def units_to_str(ds):
    ds = ds.copy()
    for v in ds.data_vars:
        if "units" in ds[v].attrs.keys():
            _u = ds[v].attrs["units"]
            if not isinstance(_u, str):
                _u = _u.to_string()
            ds[v].attrs["units"] = _u
    return ds


# %%
def add_method(cls):
    def decorator(func):
        # func needs to take `self` as the first argument
        setattr(cls, func.__name__, func)
        return func # returning func means func can still be used normally
    return decorator


# %%
@add_method(xr.DataArray)
def to_unit(a, u):
    ret = a * _get_unit(a).to(u)
    ret.attrs["units"] = u
    return ret


# %%
au.add_enabled_aliases({"degrees_north": au.degree})
au.add_enabled_aliases({"degrees_east": au.degree})
au.add_enabled_aliases({"degree_north": au.degree})
au.add_enabled_aliases({"degree_east": au.degree})
au.add_enabled_aliases({"ppm": 1e-6 * au.mol / au.mol})
au.add_enabled_aliases({"ppb": 1e-9 * au.mol / au.mol})

# %%
xr.DataArray.__mul__ = mul_u
xr.DataArray.__truediv__ = div_u
# make sure that attributes are kept, that's where we store the units
xr.set_options(keep_attrs=True)
