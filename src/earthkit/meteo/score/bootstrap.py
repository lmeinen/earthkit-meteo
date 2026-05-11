import functools
import random

try:
    import xarray as xr
except ModuleNotFoundError:
    xr = None

from . import array


def resample(x, *args, **kwargs):
    if xr is not None and isinstance(x, xr.DataArray):
        n_arrays = len(args) + 1
        dim = kwargs.get("dim", None)
        if dim is None:
            raise TypeError("resample with xarray arguments requires 'dim'")
        in_dims = [[dim] for _ in range(n_arrays)]
        sample_dim = kwargs.get("sample_dim", "sample")
        out_dims = [[sample_dim] for _ in range(n_arrays)]
        return xr.apply_ufunc(
            functools.partial(array.resample, sample_axis=-1, out_axis=-1, **kwargs),
            x,
            *args,
            input_core_dims=in_dims,
            output_core_dims=out_dims,
        )

    return array.resample(x, *args, **kwargs)


def _bootstrap_xarray(
    func,
    *args,
    dim=None,
    sample_dim="sample",
    n_iter=100,
    n_samples=None,
    randrange=random.randrange,
    **kwargs,
):
    assert xr is not None
    if dim is None:
        raise TypeError("bootstrap with xarray arguments requires 'dim'")
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
    return xr.concat(results, sample_dim)


def bootstrap(func, x, *args, **kwargs):
    if xr is not None and isinstance(x, xr.DataArray):
        return _bootstrap_xarray(func, x, *args, **kwargs)
    return array.bootstrap(func, x, *args, **kwargs)
