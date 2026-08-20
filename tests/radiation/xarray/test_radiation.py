# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest

from earthkit.meteo import radiation
from earthkit.meteo.utils.testing import NO_XARRAY

pytestmark = pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")

DIFFUSE = [0.0, 120.5, 310.0]  # W/m2
DIRECT = [0.0, 45.25, 590.0]  # W/m2


def _da(x, dims=None):
    import xarray as xr

    values = np.asarray(x)
    return xr.DataArray(values, dims=dims) if dims else xr.DataArray(values)


@pytest.mark.parametrize(
    "diffuse,direct",
    [
        (DIFFUSE, DIRECT),
        (120.5, 45.25),
    ],
)
def test_xr_downward_shortwave_radiation(diffuse, direct):
    out = radiation.surface_downward_shortwave_radiation(_da(diffuse), _da(direct))
    ref = radiation.array.surface_downward_shortwave_radiation(np.asarray(diffuse), np.asarray(direct))

    assert np.allclose(out.values, ref, equal_nan=True)
    if np.isscalar(diffuse) and np.isscalar(direct):
        assert out.ndim == 0


def test_xr_downward_shortwave_radiation_dims_and_attrs():
    diffuse = _da(DIFFUSE, dims=("point",))
    direct = _da(DIRECT, dims=("point",))

    out = radiation.surface_downward_shortwave_radiation(diffuse, direct)

    assert out.dims == ("point",)
    assert out.attrs["standard_name"] == "surface_downwelling_shortwave_flux_in_air"
    assert out.attrs["units"] == "W m-2"
