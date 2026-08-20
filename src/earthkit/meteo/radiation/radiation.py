# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,  # noqa: F401
    TypeAlias,
    overload,
)

from earthkit.meteo.utils.decorators import dispatch

if TYPE_CHECKING:
    import xarray  # type: ignore[import]
    from earthkit.data import Field, FieldList  # type: ignore[import]


ArrayLike: TypeAlias = Any


@overload
def surface_downward_shortwave_radiation(
    diffuse: "ArrayLike",
    direct: "ArrayLike",
) -> "ArrayLike": ...


@overload
def surface_downward_shortwave_radiation(
    diffuse: "xarray.DataArray",
    direct: "xarray.DataArray",
) -> "xarray.DataArray": ...


@overload
def surface_downward_shortwave_radiation(
    diffuse: "FieldList",
    direct: "FieldList",
) -> "FieldList": ...


@overload
def surface_downward_shortwave_radiation(
    diffuse: "Field",
    direct: "Field",
) -> "Field": ...


def surface_downward_shortwave_radiation(
    diffuse: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    direct: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
) -> "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field":
    r"""Compute the global downward shortwave radiation at the surface.

    Parameters
    ----------
    diffuse : array-like | xarray.DataArray | FieldList | Field
        Downward diffuse shortwave radiation on a horizontal plane (W/m2)
    direct : array-like | xarray.DataArray | FieldList | Field
        Downward direct shortwave radiation on a horizontal plane (W/m2).
        This is the direct beam projected onto the horizontal, not the
        direct normal irradiance.

    Returns
    -------
    array-like | xarray.DataArray | FieldList | Field
        Downward shortwave radiation (W/m2)


    The result is the sum of the two components:

    .. math::

        R_{sw} = R_{diffuse} + R_{direct}


    Implementations
    ------------------------
    :func:`surface_downward_shortwave_radiation` calls one of the following implementations depending on
    the type of the input arguments:

    - :py:meth:`earthkit.meteo.radiation.array.surface_downward_shortwave_radiation` for array-like
    - :py:meth:`earthkit.meteo.radiation.xarray.surface_downward_shortwave_radiation`
      for xarray.DataArray
    - :py:meth:`earthkit.meteo.radiation.fieldlist.surface_downward_shortwave_radiation`
      for FieldList | Field

    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(surface_downward_shortwave_radiation, array=True)
    return dispatched(diffuse, direct)
