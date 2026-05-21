from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from earthkit.meteo.score import xarray as bootstrap

if TYPE_CHECKING:
    import xarray  # type: ignore[import]

SEQ_5 = xr.DataArray([0, 0, 0, 2, 1, 2, 2, 4, 1, 4, 0, 4, 1, 3, 3], dims=["number"])
SEQ_10 = xr.DataArray([0, 1, 1, 5, 2, 4, 4, 9, 3, 9, 0, 9, 2, 6, 6], dims=["number"])

CSEQ_5 = xr.DataArray([0, 0, 0, 1, 4, 0, 0, 4, 0, 3, 1, 4, 1, 3, 4], dims=["number"])
CSEQ_10 = xr.DataArray([0, 0, 5, 1, 4, 5, 5, 9, 0, 8, 6, 9, 6, 3, 4], dims=["number"])


def _default_randrange(seed: int):
    random.seed(seed)


class _CustomRandRange:
    def __init__(self, seed: int):
        self.cur = seed
        self.fac = 75
        self.mod = 32769

    def __call__(self, n: int):
        self.cur = (self.cur * self.fac) % self.mod
        return self.cur % n


T = TypeVar("T")
U = TypeVar("U")


def _populate_kwarg(name: str, val: T | None, default: U, kwargs: dict) -> T | U:
    if val is None:
        val = default
    else:
        kwargs[name] = val
    return val


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_10), (_CustomRandRange, CSEQ_10)],
    ids=["default", "custom"],
)
def test_resample(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: "xarray.DataArray",
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 10, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = xr.DataArray(np.arange(10), dims=["number"])
    samples = bootstrap.resample(x, dim="number", **kwargs)
    assert len(samples) == 1
    assert samples[0].dims == ("sample", "number")
    assert samples[0].sizes["number"] == n_samples
    assert samples[0].sizes["sample"] == n_iter
    seq = seq.isel(number=slice(None, n_samples))
    xr.testing.assert_allclose(samples[0].sel(sample=0), seq)


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_10), (_CustomRandRange, CSEQ_10)],
    ids=["default", "custom"],
)
def test_resample_2(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: "xarray.DataArray",
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 10, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = xr.DataArray(np.arange(10), dims=["number"])
    y = 10 - x
    samples = bootstrap.resample(x, y, dim="number", **kwargs)
    assert len(samples) == 2
    assert samples[0].dims == ("sample", "number")
    assert samples[0].sizes["sample"] == n_iter
    assert samples[0].sizes["number"] == n_samples
    assert samples[1].dims == ("sample", "number")
    assert samples[1].sizes["sample"] == n_iter
    assert samples[1].sizes["number"] == n_samples
    seq = seq.isel(number=slice(None, n_samples))
    xr.testing.assert_allclose(samples[0].sel(sample=0), seq)
    xr.testing.assert_allclose(samples[1], 10 - samples[0])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_resample_2d(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: "xarray.DataArray",
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = xr.DataArray(np.arange(20).reshape(4, 5), dims=["point", "number"])
    samples = bootstrap.resample(x, dim="number", **kwargs)
    assert len(samples) == 1
    assert samples[0].dims == ("sample", "point", "number")
    assert samples[0].sizes["sample"] == n_iter
    assert samples[0].sizes["point"] == 4
    assert samples[0].sizes["number"] == n_samples
    seq = seq.isel(number=slice(None, n_samples))
    xr.testing.assert_allclose(
        samples[0].sel(sample=0),
        xr.concat([seq, seq + 5, seq + 10, seq + 15], "point"),
    )


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_resample_2diff(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: "xarray.DataArray",
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = xr.DataArray(np.arange(20).reshape(4, 5), dims=["point", "number"])
    y = xr.DataArray(np.arange(5), dims=["number"])
    samples = bootstrap.resample(x, y, dim="number", **kwargs)
    assert len(samples) == 2
    assert samples[0].dims == ("sample", "point", "number")
    assert samples[0].sizes["sample"] == n_iter
    assert samples[0].sizes["point"] == 4
    assert samples[0].sizes["number"] == n_samples
    assert samples[1].dims == ("sample", "number")
    assert samples[1].sizes["sample"] == n_iter
    assert samples[1].sizes["number"] == n_samples
    seq = seq.isel(number=slice(None, n_samples))
    xr.testing.assert_allclose(
        samples[0].sel(sample=0),
        xr.concat([seq, seq + 5, seq + 10, seq + 15], "point"),
    )
    xr.testing.assert_allclose(samples[1], samples[0].sel(point=0))


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_10), (_CustomRandRange, CSEQ_10)],
    ids=["default", "custom"],
)
def test_resample_out_dim(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: "xarray.DataArray",
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 10, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = xr.DataArray(np.arange(10), dims=["number"])
    samples = bootstrap.resample(x, dim="number", out_dim="iteration", **kwargs)
    assert len(samples) == 1
    assert samples[0].dims == ("iteration", "number")
    assert samples[0].sizes["number"] == n_samples
    assert samples[0].sizes["iteration"] == n_iter
    seq = seq.isel(number=slice(None, n_samples))
    xr.testing.assert_allclose(samples[0].sel(iteration=0), seq)


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_10), (_CustomRandRange, CSEQ_10)],
    ids=["default", "custom"],
)
def test_bootstrap(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: "xarray.DataArray",
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 10, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = xr.DataArray(np.arange(10), dims=["number"])
    values = bootstrap.bootstrap((lambda a: a.mean("number")), x, dim="number", **kwargs)
    assert values.dims == ("sample",)
    assert values.sizes["sample"] == n_iter
    seq = seq.isel(number=slice(None, n_samples))
    xr.testing.assert_allclose(values.sel(sample=0), seq.mean("number"))


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_10), (_CustomRandRange, CSEQ_10)],
    ids=["default", "custom"],
)
def test_bootstrap_2(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: "xarray.DataArray",
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 10, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = xr.DataArray(np.arange(10), dims=["number"])
    y = 10 - x
    values = bootstrap.bootstrap((lambda a, b: a.mean("number") - b.mean("number")), x, y, dim="number", **kwargs)
    assert values.dims == ("sample",)
    assert values.sizes["sample"] == n_iter
    seq = seq.isel(number=slice(None, n_samples))
    xr.testing.assert_allclose(values.sel(sample=0), 2 * seq.mean("number") - 10)


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_bootstrap_2d(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: "xarray.DataArray",
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = xr.DataArray(np.arange(20).reshape(4, 5), dims=["point", "number"])
    values = bootstrap.bootstrap((lambda a: a.mean("number")), x, dim="number", **kwargs)
    assert values.dims == ("sample", "point")
    assert values.sizes["sample"] == n_iter
    assert values.sizes["point"] == 4
    m = seq.isel(number=slice(None, n_samples)).mean("number")
    xr.testing.assert_allclose(values.sel(sample=0), xr.concat([m, m + 5, m + 10, m + 15], "point"))


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_bootstrap_2diff(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: "xarray.DataArray",
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = xr.DataArray(np.arange(20).reshape(4, 5), dims=["point", "number"])
    y = xr.DataArray(np.arange(5), dims=["number"])
    values = bootstrap.bootstrap((lambda a, b: a.mean("number") - b.mean("number")), x, y, dim="number", **kwargs)
    assert values.dims == ("sample", "point")
    assert values.sizes["sample"] == n_iter
    assert values.sizes["point"] == 4
    seq = seq.isel(number=slice(None, n_samples))
    xr.testing.assert_allclose(values.sel(sample=0), xr.DataArray([0, 5, 10, 15], dims=["point"]))


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_10), (_CustomRandRange, CSEQ_10)],
    ids=["default", "custom"],
)
def test_bootstrap_out_dim(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: "xarray.DataArray",
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 10, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = xr.DataArray(np.arange(10), dims=["number"])
    values = bootstrap.bootstrap((lambda a: a.mean("number")), x, dim="number", out_dim="iteration", **kwargs)
    assert values.dims == ("iteration",)
    assert values.sizes["iteration"] == n_iter
    seq = seq.isel(number=slice(None, n_samples))
    xr.testing.assert_allclose(values.sel(iteration=0), seq.mean("number"))
