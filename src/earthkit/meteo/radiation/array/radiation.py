# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import Any, TypeAlias

from earthkit.utils.array import array_namespace

ArrayLike: TypeAlias = Any


def surface_downward_shortwave_radiation(diffuse: ArrayLike, direct: ArrayLike) -> ArrayLike:
    r"""Compute the global downward shortwave radiation at the surface.

    Parameters
    ----------
    diffuse : number or array-like
        Downward diffuse shortwave radiation on a horizontal plane (W/m2)
    direct : number or array-like
        Downward direct shortwave radiation on a horizontal plane (W/m2).
        This is the direct beam projected onto the horizontal, not the
        direct normal irradiance.

    Returns
    -------
    number or array-like
        Downward shortwave radiation (W/m2)


    The result is the sum of the two components:

    .. math::

        R_{sw} = R_{diffuse} + R_{direct}

    Both inputs must refer to the same plane and use the same units, so the
    function equally applies to time-integrated fluxes (J/m2).
    """
    xp = array_namespace(diffuse, direct)
    diffuse = xp.asarray(diffuse)
    direct = xp.asarray(direct)

    return diffuse + direct
