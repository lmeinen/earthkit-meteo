# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from __future__ import annotations

from earthkit.data import Field, FieldList  # type: ignore[import]

from earthkit.meteo.utils.decorators import fieldlist_ufunc

from .. import array


def surface_downward_shortwave_radiation(diffuse: FieldList | Field, direct: FieldList | Field) -> FieldList | Field:
    r"""Compute the global downward shortwave radiation at the surface.

    Parameters
    ----------
    diffuse : FieldList|Field
        Downward diffuse shortwave radiation on a horizontal plane (W/m2)
    direct : FieldList|Field
        Downward direct shortwave radiation on a horizontal plane (W/m2).
        This is the direct beam projected onto the horizontal, not the
        direct normal irradiance.

    Returns
    -------
    FieldList|Field
        Downward shortwave radiation (W/m2). The result has the same type as the
        input ``diffuse`` (FieldList or Field).


    The result is the sum of the two components:

    .. math::

        R_{sw} = R_{diffuse} + R_{direct}

    """
    fieldlist_ufunc_kwargs = {"default_variable": "surface_downward_shortwave_radiation"}

    return fieldlist_ufunc(
        array.surface_downward_shortwave_radiation, diffuse, direct, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )
