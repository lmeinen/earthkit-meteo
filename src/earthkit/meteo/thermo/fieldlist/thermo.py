from earthkit.data import FieldList  # type: ignore[import]

from earthkit.meteo import constants


def potential_temperature(t: FieldList, p: FieldList) -> FieldList:
    r"""Compute the potential temperature.

    Parameters
    ----------
    t: FieldList
        Temperature (K)
    p: FieldList
        Pressure (Pa)

    Returns
    -------
    FieldList
        Potential temperature (K)


    The computation is based on the following formula [Wallace2006]_:

    .. math::

       \theta = t \left(\frac{10^{5}}{p}\right)^{\kappa}

    with :math:`\kappa = R_{d}/c_{pd}` (see :data:`earthkit.meteo.constants.kappa`).

    """
    res = t * (constants.p0 / p) ** constants.kappa
    return res.set({"parameter.variable": "pt"})
