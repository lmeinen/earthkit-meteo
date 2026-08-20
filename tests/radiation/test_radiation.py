# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Tests that the high level radiation API dispatches to the backend implementations."""

import numpy as np
import pytest
from earthkit.utils.array import array_namespace
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo import radiation
from earthkit.meteo.utils.testing import NO_EKD, NO_XARRAY

DIFFUSE = [0.0, 120.5, 310.0]  # W/m2
DIRECT = [0.0, 45.25, 590.0]  # W/m2
TOTAL = [0.0, 165.75, 900.0]  # W/m2


def _signature(obj):
    if hasattr(obj, "dims") and hasattr(obj, "shape"):
        return ("xarray", tuple(obj.dims), tuple(obj.shape))

    xp = array_namespace(obj)
    arr = xp.asarray(obj)
    return ("array", tuple(arr.shape))


def _case_array(xp, device):
    import earthkit.meteo.radiation.array as impl

    return {
        "impl": impl,
        "args": (xp.asarray(DIFFUSE, device=device), xp.asarray(DIRECT, device=device)),
    }


def _case_xarray():
    import xarray as xr

    import earthkit.meteo.radiation.xarray as impl

    return {
        "impl": impl,
        "args": (xr.DataArray(np.asarray(DIFFUSE)), xr.DataArray(np.asarray(DIRECT))),
    }


BACKEND_PARAMS = [
    pytest.param(("array", xp, device), id=f"array-{xp._earthkit_array_namespace_name}-{device}")
    for xp, device in NAMESPACE_DEVICES
]
if not NO_XARRAY:
    BACKEND_PARAMS.append(pytest.param(("xarray", None, None), id="xarray"))


@pytest.fixture(params=BACKEND_PARAMS)
def backend_case(request):
    kind, xp, device = request.param
    if kind == "xarray":
        return _case_xarray()
    return _case_array(xp, device)


def test_highlevel_compatible_with_backend_api(backend_case):
    args = backend_case["args"]

    got = radiation.surface_downward_shortwave_radiation(*args)
    ref = backend_case["impl"].surface_downward_shortwave_radiation(*args)

    assert _signature(got) == _signature(ref)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref))
    np.testing.assert_allclose(np.asarray(got), np.array(TOTAL))


@pytest.mark.skipif(NO_EKD, reason="EKD is not installed")
def test_highlevel_dispatches_to_fieldlist():
    from earthkit.data import Field, FieldList

    def _fieldlist(values, variable):
        return FieldList.from_fields([
            Field.from_components(
                values=np.array(values),
                parameter={"variable": variable, "units": "W/m2"},
                vertical={"level": 0, "level_type": "surface"},
            )
        ])

    diffuse = _fieldlist(DIFFUSE, "ssrd_diffuse")
    direct = _fieldlist(DIRECT, "fdir")

    out = radiation.surface_downward_shortwave_radiation(diffuse, direct)

    assert len(out) == 1
    assert out.get("parameter.variable") == ["surface_downward_shortwave_radiation"]
    np.testing.assert_allclose(out[0].values, np.array(TOTAL))
