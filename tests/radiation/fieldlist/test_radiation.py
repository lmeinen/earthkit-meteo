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

from earthkit.meteo.utils.testing import NO_EKD

pytestmark = pytest.mark.skipif(NO_EKD, reason="EKD is not installed")

DIFFUSE = [[0.0, 120.5], [310.0, 22.0]]  # W/m2
DIRECT = [[0.0, 45.25], [590.0, 8.0]]  # W/m2
TOTAL = [[0.0, 165.75], [900.0, 30.0]]  # W/m2

PARAMETERS = {
    "diffuse": {"variable": "ssrd_diffuse", "units": "W/m2"},
    "direct": {"variable": "fdir", "units": "W/m2"},
}


def _make_input_fieldlist(param, values, input_type="fieldlist"):
    from earthkit.data import Field, FieldList

    param_def = PARAMETERS[param]
    vertical = {"level": 0, "level_type": "surface"}

    fields = [
        Field.from_components(
            values=np.array(v),
            parameter={"variable": param_def["variable"], "units": param_def["units"]},
            vertical=vertical,
        )
        for v in values
    ]

    return fields[0] if input_type == "field" else FieldList.from_fields(fields)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_downward_shortwave_radiation(input_type):
    from earthkit.meteo.radiation.fieldlist import surface_downward_shortwave_radiation

    diffuse = _make_input_fieldlist("diffuse", DIFFUSE, input_type=input_type)
    direct = _make_input_fieldlist("direct", DIRECT, input_type=input_type)

    out = surface_downward_shortwave_radiation(diffuse, direct)

    assert isinstance(out, type(diffuse))

    if input_type == "fieldlist":
        assert len(out) == len(diffuse)
        assert out.get("parameter.variable") == ["surface_downward_shortwave_radiation"] * len(diffuse)
        for f, ref_vals in zip(out, TOTAL):
            np.testing.assert_allclose(f.values, np.array(ref_vals))
    else:
        assert out.get("parameter.variable") == "surface_downward_shortwave_radiation"
        np.testing.assert_allclose(out.values, np.array(TOTAL[0]))


def test_fieldlist_downward_shortwave_radiation_units():
    """The output parameter metadata is resolved from FIELD_PARAMS."""
    from earthkit.meteo.radiation.fieldlist import surface_downward_shortwave_radiation

    diffuse = _make_input_fieldlist("diffuse", DIFFUSE, input_type="field")
    direct = _make_input_fieldlist("direct", DIRECT, input_type="field")

    out = surface_downward_shortwave_radiation(diffuse, direct)

    assert str(out.get("parameter.units")) == str(diffuse.get("parameter.units"))
