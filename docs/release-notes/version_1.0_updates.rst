.. _release-notes-1.0:

Version 1.0 Updates
///////////////////////


Version 1.0.0
===============


Deprecated features
------------------------
The following deprecations have been introduced in this release:

-  :ref:`deprecated-hybrid-pressure-at-model-levels`
-  :ref:`deprecated-hybrid-relative-geopotential-thickness`
-  :ref:`deprecated-hybrid-pressure-at-height-levels`

Another change is that all these methods now need to be imported from the `earthkit.meteo.vertical.array` submodule. Previously they were available in the `earthkit.meteo.vertical` submodule.

Removed
----------------

The following functions have been removed:

- :py:func:`earthkit.meteo.thermo.kelvin_to_celsius`
- :py:func:`earthkit.meteo.thermo.celsius_to_kelvin`

They can be both replaced by using the newly added :py:attr:`~earthkit.meteo.constants.T_C2K` (273.15 K) constant.


High level interface
-----------------------
Added a high-level interface to most of the functions to support array, Xarray and earthkit-data FieldList/Field inputs. This offers automatic dispatching to the appropriate implementation based on the input type. This allows users to use the same function names and parameters regardless of the input type, while still benefiting from the optimized implementations for each type.

.. code-block:: python

    from earthkit.meteo import wind

    # For array-based inputs
    speed = wind.speed(u_array, v_array)

    # For Xarray-based inputs
    speed = wind.speed(u_xarray, v_xarray)


The actual implementations are now available in the ``array``, ``xarray`` and ``fieldlist``
submodules and can also be directly accessed bypassing the high level dispatch mechanism.

.. code-block:: python

    from earthkit.meteo.wind.array import wind as array_wind
    from earthkit.meteo.wind.xarray import wind as xarray_wind
    from earthkit.meteo.wind.fieldlist import wind as fieldlist_wind

    # For array-based inputs
    speed = array_wind.speed(u_array, v_array)

    # For Xarray-based inputs
    speed = xarray_wind.speed(u_xarray, v_xarray)

    # For FieldList-based inputs
    speed = fieldlist_wind.speed(u_fieldlist, v_fieldlist)


Examples of using the high-level interface with various inputs can be found in the following notebook example:

  - :ref:`/tutorials/input/input_formats.ipynb`


New vertical methods
------------------------------------------------

The following coordinate computing functions have been added to the :py:mod:`earthkit.meteo.vertical` submodule for array and FieldList inputs:

  - hybrid_level_parameters
  - pressure_on_hybrid_levels
  - relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta
  - relative_geopotential_thickness_on_hybrid_levels
  - geopotential_on_hybrid_levels
  - height_on_hybrid_levels

The following vertical interpolation functions have been added to the :py:mod:`earthkit.meteo.vertical` submodule for array and FieldList inputs:

  - interpolate_hybrid_to_pressure_levels
  - interpolate_hybrid_to_height_levels
  - interpolate_pressure_to_height_levels
  - interpolate_monotonic

The following vertical interpolation functions have been added to the :py:mod:`earthkit.meteo.vertical.xarray` submodule for Xarray inputs:

  - interpolate_monotonic
  - interpolate_to_pressure_levels
  - interpolate_sleve_to_coord_levels
  - interpolate_sleve_to_theta_levels


See the notebook examples:

- :ref:`vertical-tutorials`


Regimes
------------------------

Added new submodule :py:mod:`earthkit.meteo.regimes` with classes to define a weather regime classification based on patterns and functions to project anomaly fields onto these patterns to compute a regime index, following the approach of `Michel and Rivière (2011) <https://doi.org/10.1175/2011JAS3635.1>`_.

See the notebook example: :ref:`/tutorials/regimes/seven_weather_regimes.ipynb`


Bootstrap utilities
---------------------------------------

Added a new bootstrapping helper module :py:mod:`earthkit.meteo.score.bootstrap`
(:pr:`54`) providing functions for statistical resampling and bootstrap
confidence interval estimation. Both array and xarray inputs are supported.

Key functions:

- :py:func:`~earthkit.meteo.score.bootstrap.resample` — draw bootstrap samples
  from a dataset, with optional replacement
- :py:func:`~earthkit.meteo.score.bootstrap.bootstrap` — compute a statistic
  over bootstrap samples


``kge`` exposed at top-level score module
-------------------------------------------

:py:func:`~earthkit.meteo.score.kge` (Kling–Gupta Efficiency) is now importable
directly from :py:mod:`earthkit.meteo.score` (:pr:`162`).
