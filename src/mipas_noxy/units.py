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
from astropy import units as au
import xarray_simpleunits as xru


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
au.add_enabled_aliases({"degrees_north": au.degree})
au.add_enabled_aliases({"degrees_east": au.degree})
au.add_enabled_aliases({"degree_north": au.degree})
au.add_enabled_aliases({"degree_east": au.degree})
au.add_enabled_aliases({"ppm": 1e-6 * au.mol / au.mol})
au.add_enabled_aliases({"ppb": 1e-9 * au.mol / au.mol})

# %%
xru.init_units()
