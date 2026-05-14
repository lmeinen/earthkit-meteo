import random

from earthkit.utils.array import array_namespace


def iter_samples(x, *args, dim=0, n_iter=100, n_samples=None, randrange=random.randrange):
    """Iterate over resampled arrays for bootstrapping.

    Parameters
    ----------
    x, *args: array-like
        Arrays to sample. Must have the same size along ``dim``
    dim: int or list of int
        Sample along this dimension index (either same for all or one per argument)
    n_iter: int
        Number of bootstrapping iterations
    n_samples: int or None
        Number of samples for each iteration. If None, use the number of
        inputs (size of ``x`` along the sampling dimension)
    randrange: function (int -> int)
        Random generator for integers: `randrange(n)` should return an
        integer in `range(n)`

    Yields
    ------
    tuple
        Resampled arrays (one element per input array, one yield per iteration)
    """
    args = (x,) + args
    n_arrays = len(args)
    if isinstance(dim, int):
        dim = [dim for _ in range(n_arrays)]
    else:
        assert len(dim) == n_arrays, "dim must have one element per input array"
    xp = array_namespace(*args)
    arrays = tuple((xp.asarray(arr), axis) for arr, axis in zip(args, dim))
    n_inputs = x.shape[dim[0]]
    assert all(y.shape[axis] == n_inputs for y, axis in arrays), (
        "Input arrays must have the same size along the sampling dimension"
    )
    if n_samples is None:
        n_samples = n_inputs
    for _ in range(n_iter):
        indices = [randrange(n_inputs) for _ in range(n_samples)]
        sampled = tuple(xp.take(y, indices=indices, axis=axis) for y, axis in arrays)
        yield sampled


def resample(x, *args, dim=0, out_dim=0, n_iter=100, n_samples=None, randrange=random.randrange):
    """Resample arrays for bootstrapping.

    Parameters
    ----------
    x, *args: array-like
        Arrays to sample. Must have the same size along ``dim``
    dim: int or list of int
        Sample along this dimension index (either same for all or one per argument)
    out_dim: int
        Stack samples along this dimension index
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
        dim = 0
    if out_dim is None:
        out_dim = 0
    xp = array_namespace(x, *args)
    n_arrays = len(args) + 1
    samples = [[] for _ in range(n_arrays)]
    samples_it = iter_samples(x, *args, dim=dim, n_iter=n_iter, n_samples=n_samples, randrange=randrange)
    for sample in samples_it:
        for i in range(n_arrays):
            samples[i].append(sample[i])
    return tuple(xp.stack(sampled_arr, axis=out_dim) for sampled_arr in samples)


def bootstrap(
    func,
    x,
    *args,
    dim=0,
    out_dim=0,
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
    x, *args: array-like
        Inputs to ``function``, sampled for bootstrapping. Must have the same
        size along ``dim``
    dim: int or list of int
        Sample along this dimension index (either same for all or one per argument)
    out_dim: int
        Stack samples along this dimension index
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
    array-like
        Aggregated results of the bootstrapping process
    """
    xp = array_namespace(x, *args)
    samples = iter_samples(
        x,
        *args,
        dim=dim,
        n_iter=n_iter,
        n_samples=n_samples,
        randrange=randrange,
    )
    return xp.stack([func(*sampled, **kwargs) for sampled in samples], axis=out_dim)
