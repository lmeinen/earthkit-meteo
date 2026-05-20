# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import functools
import random
from collections.abc import Callable

import xarray as xr

from .. import array


def _unwrap(func):
    @functools.wraps(func)
    def unwrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 1:
            return result[0]
        return result

    return unwrapped


def resample(
    x: xr.DataArray,
    *args: xr.DataArray,
    dim: str = None,
    out_dim: str = "sample",
    n_iter: int = 100,
    n_samples: int | None = None,
    randrange: Callable[[int], int] = random.randrange,
) -> tuple[xr.DataArray, ...]:
    """Resample arrays for bootstrapping.

    Parameters
    ----------
    x, *args: xarray object
        Arrays to sample. Must have the same size along ``dim``
    dim: str
        Sample along this dimension
    out_dim: str
        Output dimension name for samples
    n_iter: int
        Number of bootstrapping iterations
    n_samples: int or None
        Number of samples for each iteration. If None, use the number of
        inputs (size of ``x`` along the sampling dimension)
    randrange: function (int -> int)
        Random generator for integers: `randrange(n)` should return an
        integer in `range(n)`

    Returns
    -------
    tuple
        Resampled arrays (one element per input array)
    """
    if dim is None:
        raise TypeError("resample with xarray arguments requires 'dim'")
    if out_dim is None:
        out_dim = "sample"
    n_arrays = len(args) + 1
    in_dims = [(dim,) for _ in range(n_arrays)]
    out_dims = [(dim, out_dim) for _ in range(n_arrays)]
    resampled = xr.apply_ufunc(
        functools.partial(
            _unwrap(array.resample),
            dim=-1,
            out_dim=-1,
            n_iter=n_iter,
            n_samples=n_samples,
            randrange=randrange,
        ),
        x,
        *args,
        input_core_dims=in_dims,
        exclude_dims={dim},
        output_core_dims=out_dims,
    )
    if n_arrays == 1:
        return (resampled,)
    return resampled


def bootstrap(
    func: Callable[..., xr.DataArray],
    *args: xr.DataArray,
    dim: str = None,
    out_dim: str = "sample",
    n_iter: int = 100,
    n_samples: int | None = None,
    randrange: Callable[[int], int] = random.randrange,
    **kwargs,
) -> xr.DataArray:
    """Run bootstrapping.

    Parameters
    ----------
    func: function ((array, ..., **kwargs) -> array)
        Function to bootstrap
    *args: xarray object
        Inputs to ``function``, sampled for bootstrapping. Must have the same
        size along ``dim``
    dim: str
        Sample along this dimension
    out_dim: str
        Output dimension name for samples
    n_iter: int
        Number of bootstrapping iterations
    n_samples: int or None
        Number of samples for each iteration. If None, use the number of
        inputs (size of the first array along the sampling dimension)
    randrange: function (int -> int)
        Random generator for integers: `randrange(n)` should return an
        integer in `range(n)`
    **kwargs
        Additional keyword arguments to ``func``

    Returns
    -------
    xarray object
        Aggregated results of the bootstrapping process
    """
    if dim is None:
        raise TypeError("bootstrap with xarray arguments requires 'dim'")
    if out_dim is None:
        out_dim = "sample"
    n_inputs = args[0].sizes[dim]
    assert all(arr.sizes[dim] == n_inputs for arr in args), (
        "Input arrays must have the same size along the sampling axis"
    )
    if n_samples is None:
        n_samples = n_inputs
    results = []
    for _ in range(n_iter):
        indices = [randrange(n_inputs) for _ in range(n_samples)]
        sampled = tuple(arr.isel({dim: indices}) for arr in args)
        results.append(func(*sampled, **kwargs))
    return xr.concat(results, out_dim)
