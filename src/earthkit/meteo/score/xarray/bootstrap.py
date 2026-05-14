import functools
import random

import xarray as xr

from .. import array


def resample(
    x,
    *args,
    dim=None,
    out_dim="sample",
    n_iter=100,
    n_samples=None,
    randrange=random.randrange,
):
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
    in_dims = [[dim] for _ in range(n_arrays)]
    out_dims = [[out_dim] for _ in range(n_arrays)]
    return xr.apply_ufunc(
        functools.partial(
            array.resample,
            sample_axis=-1,
            out_axis=-1,
            n_iter=n_iter,
            n_samples=n_samples,
            randrange=randrange,
        ),
        x,
        *args,
        input_core_dims=in_dims,
        output_core_dims=out_dims,
    )


def bootstrap(
    func,
    *args,
    dim=None,
    out_dim="sample",
    n_iter=100,
    n_samples=None,
    randrange=random.randrange,
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
        inputs (size of ``x`` along the sampling dimension)
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
