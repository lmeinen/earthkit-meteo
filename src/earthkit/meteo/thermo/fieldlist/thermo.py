from earthkit.data import FieldList  # type: ignore[import]

from earthkit.meteo.utils.decorators import fieldlist_ufunc

from .. import array


def potential_temperature(t: FieldList, p: FieldList) -> FieldList:
    r"""Compute the potential temperature.

    Parameters
    ----------
    t: FieldList
        Temperature (K)
    p: FieldList or Iterable[float]
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
    fieldlist_ufunc_kwargs = {"default": "pt"}

    return fieldlist_ufunc(array.potential_temperature, t, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)
