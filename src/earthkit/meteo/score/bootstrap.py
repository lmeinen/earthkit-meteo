# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import random

from ..utils.decorators import dispatch


def resample(
    x,
    *args,
    dim=None,
    out_dim=None,
    n_iter=100,
    n_samples=None,
    randrange=random.randrange,
    **kwargs,
):
    """Resample arrays for bootstrapping.

    Parameters
    ----------
    x, *args: xarray object or array-like
        Arrays to sample. Must have the same size along ``dim``
    dim: str or int or list of int
        Sample along this dimension (name or index/indices for array-like)
    out_dim: str or int
        Output dimension name (or index for array-like) for samples
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
    dispatched = dispatch(resample, fieldlist=False, array=True)
    return dispatched(
        x,
        *args,
        dim=dim,
        out_dim=out_dim,
        n_iter=n_iter,
        n_samples=n_samples,
        randrange=randrange,
        **kwargs,
    )


def bootstrap(
    func,
    x,
    *args,
    dim=None,
    out_dim="sample",
    n_iter=100,
    n_samples=None,
    randrange=random.randrange,
    **kwargs,
):
    """Run bootstrapping.

    Parameters
    ----------
    func: function ((array, ..., **kwargs) -> array)
        Function to bootstrap
    x, *args: xarray object or array-like
        Inputs to ``function``, sampled for bootstrapping. Must have the same
        size along ``dim``
    dim: str or int or list of int
        Sample along this dimension (name or index/indices for array-like)
    out_dim: str or int
        Output dimension name (or index for array-like) for samples
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
    xarray object or array-like
        Aggregated results of the bootstrapping process
    """
    dispatched = dispatch(bootstrap, match=1, fieldlist=False, array=True)
    return dispatched(
        func,
        x,
        *args,
        dim=dim,
        out_dim=out_dim,
        n_iter=n_iter,
        n_samples=n_samples,
        randrange=randrange,
        **kwargs,
    )
