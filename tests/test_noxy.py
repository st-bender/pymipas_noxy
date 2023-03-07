# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
import numpy as np
import xarray as xr

from mipas_noxy.util import combine_NOxy, calculate_NOxy, fixup_target_name


def _prep_dss():
    coords={
        "time": np.array([
            "2000-01-01T00:00:01",
            "2000-01-01T01:00:01",
        ], dtype="M8[ns]"),
        "altitude": [50.],
    }
    target_attrs = dict(
        standard_name = "mole_fraction_of_nitrogen_monoxide_in_air",
        units = "1e-6",
        long_name = "volume mixing ratio of NO",
    )
    global_attrs = dict(
        retrieval_target_name = "NO",
        retrieval_in_logarithmic_parameter_space = "TRUE",
    )
    ds1 = xr.Dataset(
        data_vars={
            "geo_id": (["time"], [b"00001_20000101T000001Z", b"00002_20000101T010001Z"]),
            "dof": (["time"], [30, 30]),
            "chi2": (["time"], [1.1, 1.1]),
            "target": (["altitude", "time"], [[1., 2.]], target_attrs),
            "target_noise_error": (["altitude", "time"], [[1., 1.]]),
            "akm_diagonal": (["altitude", "time"], [[0.1, 0.2]]),
            "weights": ([], 1.0),
        },
        coords=coords,
        attrs=global_attrs,
    ).expand_dims(species=["NO"])
    ds2 = xr.Dataset(
        data_vars={
            "geo_id": (["time"], [b"00001_20000101T000001Z", b"00002_20000101T010001Z"]),
            "dof": (["time"], [40, 40]),
            "chi2": (["time"], [1.2, 1.2]),
            "target": (["altitude", "time"], [[3., 4.]], target_attrs),
            "target_noise_error": (["altitude", "time"], [[2., 2.]]),
            "akm_diagonal": (["altitude", "time"], [[0.2, 0.3]]),
            "weights": ([], 1.0),
        },
        coords=coords,
        attrs=global_attrs,
    ).expand_dims(species=["NO2"])
    ds3 = xr.Dataset(
        data_vars={
            "geo_id": (["time"], [b"00001_20000101T000001Z", b"00002_20000101T010001Z"]),
            "dof": (["time"], [50, 50]),
            "chi2": (["time"], [1.3, 1.3]),
            "target": (["altitude", "time"], [[-2., -1.]], target_attrs),
            "target_noise_error": (["altitude", "time"], [[1., 2.]]),
            "akm_diagonal": (["altitude", "time"], [[0.1, 0.2]]),
            "weights": ([], 2.0),
        },
        coords=coords,
        attrs=global_attrs,
    ).expand_dims(species=["N2O5"])
    return [ds1, ds2, ds3]


def test_noxy():
    dss = _prep_dss()
    dsc = combine_NOxy(dss)
    noxy = calculate_NOxy(dsc)
    np.testing.assert_allclose(noxy.dof, [120, 120])
    np.testing.assert_allclose(noxy.target, [[0., 4.]])
    np.testing.assert_allclose(noxy.target_noise_error, [[3., 4.582576]])
    np.testing.assert_allclose(noxy.chi2, [1.216666667, 1.216666667])
    np.testing.assert_allclose(noxy.akm_diagonal, [[.1375, 0.25]])


def test_fixup_target_name():
    noxy = _prep_dss()[0]
    noxy = fixup_target_name(noxy, "NO", "NOY")
    assert noxy.target.attrs["long_name"] == "volume mixing ratio of NOY"
    assert noxy.attrs["retrieval_target_name"] == "NOY"
