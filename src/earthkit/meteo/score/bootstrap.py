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
