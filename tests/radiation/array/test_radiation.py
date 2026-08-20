# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Tests for the array level radiation functions."""

import numpy as np
import pytest
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo.radiation import array as radiation

DIFFUSE = [0.0, 120.5, 310.0]  # W/m2
DIRECT = [0.0, 45.25, 590.0]  # W/m2
TOTAL = [0.0, 165.75, 900.0]  # W/m2


@pytest.mark.parametrize("xp,device", NAMESPACE_DEVICES)
def test_downward_shortwave_radiation(xp, device):
    diffuse = xp.asarray(DIFFUSE, device=device)
    direct = xp.asarray(DIRECT, device=device)

    out = radiation.surface_downward_shortwave_radiation(diffuse, direct)

    assert out.shape == diffuse.shape
    np.testing.assert_allclose(np.asarray(out), np.array(TOTAL))


@pytest.mark.parametrize("xp,device", NAMESPACE_DEVICES)
def test_downward_shortwave_radiation_broadcast(xp, device):
    """A scalar component is broadcast against a gridded one."""
    diffuse = xp.asarray(DIFFUSE, device=device)

    out = radiation.surface_downward_shortwave_radiation(diffuse, 10.0)

    np.testing.assert_allclose(np.asarray(out), np.array(DIFFUSE) + 10.0)


def test_downward_shortwave_radiation_list_input():
    """Lists are converted to arrays rather than concatenated."""
    out = radiation.surface_downward_shortwave_radiation(DIFFUSE, DIRECT)

    np.testing.assert_allclose(np.asarray(out), np.array(TOTAL))


def test_downward_shortwave_radiation_nan_propagates():
    diffuse = np.array([1.0, np.nan])
    direct = np.array([2.0, 3.0])

    out = radiation.surface_downward_shortwave_radiation(diffuse, direct)

    np.testing.assert_allclose(out, np.array([3.0, np.nan]))
