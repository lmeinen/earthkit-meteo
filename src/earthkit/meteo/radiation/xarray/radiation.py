# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import xarray as xr

from earthkit.meteo.utils.decorators import xarray_ufunc

from .. import array


def surface_downward_shortwave_radiation(diffuse: xr.DataArray, direct: xr.DataArray) -> xr.DataArray:
    r"""Compute the global downward shortwave radiation at the surface.

    Parameters
    ----------
    diffuse : xarray.DataArray
        Downward diffuse shortwave radiation on a horizontal plane (W/m2)
    direct : xarray.DataArray
        Downward direct shortwave radiation on a horizontal plane (W/m2).
        This is the direct beam projected onto the horizontal, not the
        direct normal irradiance.

    Returns
    -------
    xarray.DataArray
        Downward shortwave radiation (W/m2)


    The result is the sum of the two components:

    .. math::

        R_{sw} = R_{diffuse} + R_{direct}

    """
    return xarray_ufunc(array.surface_downward_shortwave_radiation, diffuse, direct).assign_attrs({
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "units": "W m-2",
    })
