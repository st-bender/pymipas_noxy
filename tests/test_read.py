# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
from os import sep as dsep

import pytest

from mipas_noxy.util import get_nc_filename


@pytest.mark.parametrize(
    "res, spec, v, name",
    [
        ["R", "NO", "261", "V8R_NO_261" + dsep + "MIPAS-E_IMK.200001.V8R_NO_261.nc"],
        ["H", "NO2", "61", "V8H_NO2_61" + dsep + "MIPAS-E_IMK.200001.V8H_NO2_61.nc"],
    ]
)
def test_get_nc_filename(res, spec, v, name):
    fpath = get_nc_filename(
        "", res, spec, 2000, 1, version=v,
    )
    assert fpath == name
