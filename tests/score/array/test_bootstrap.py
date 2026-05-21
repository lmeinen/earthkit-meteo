# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import random
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import pytest

from earthkit.meteo.score import array as bootstrap

SEQ_5 = np.array([0, 0, 0, 2, 1, 2, 2, 4, 1, 4, 0, 4, 1, 3, 3])
SEQ_10 = np.array([0, 1, 1, 5, 2, 4, 4, 9, 3, 9, 0, 9, 2, 6, 6])

CSEQ_5 = np.array([0, 0, 0, 1, 4, 0, 0, 4, 0, 3, 1, 4, 1, 3, 4])
CSEQ_10 = np.array([0, 0, 5, 1, 4, 5, 5, 9, 0, 8, 6, 9, 6, 3, 4])


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
def test_iter_samples(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 10, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(10)
    samples = list(bootstrap.iter_samples(x, **kwargs))
    assert len(samples) == n_iter
    assert len(samples[0]) == 1
    assert len(samples[0][0]) == n_samples
    np.testing.assert_equal(samples[0][0], seq[:n_samples])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_10), (_CustomRandRange, CSEQ_10)],
    ids=["default", "custom"],
)
def test_iter_samples_2(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 10, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(10)
    y = 10 - x
    random.seed(2)
    samples = list(bootstrap.iter_samples(x, y, **kwargs))
    assert len(samples) == n_iter
    assert len(samples[0]) == 2
    assert len(samples[0][0]) == n_samples
    assert len(samples[0][1]) == n_samples
    np.testing.assert_equal(samples[0][0], seq[:n_samples])
    for sx, sy in samples:
        np.testing.assert_equal(sy, 10 - sx)


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_iter_samples_dim(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(20).reshape(4, 5)
    random.seed(2)
    samples = list(bootstrap.iter_samples(x, dim=1, **kwargs))
    assert len(samples) == n_iter
    assert len(samples[0]) == 1
    assert samples[0][0].shape == (4, n_samples)
    seq = seq[:n_samples]
    np.testing.assert_equal(
        samples[0][0],
        [seq, seq + 5, seq + 10, seq + 15],
    )


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_iter_samples_dim_2(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(20).reshape(4, 5)
    y = 20 - x
    random.seed(2)
    samples = list(bootstrap.iter_samples(x, y, dim=1, **kwargs))
    assert len(samples) == n_iter
    assert len(samples[0]) == 2
    assert samples[0][0].shape == (4, n_samples)
    assert samples[0][1].shape == (4, n_samples)
    seq = seq[:n_samples]
    np.testing.assert_equal(
        samples[0][0],
        [seq, seq + 5, seq + 10, seq + 15],
    )
    for sx, sy in samples:
        np.testing.assert_equal(sy, 20 - sx)


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_iter_samples_dim_2diff(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(20).reshape(4, 5)
    y = np.arange(5)
    random.seed(2)
    samples = list(bootstrap.iter_samples(x, y, dim=[1, 0], **kwargs))
    assert len(samples) == n_iter
    assert len(samples[0]) == 2
    assert samples[0][0].shape == (4, n_samples)
    assert samples[0][1].shape == (n_samples,)
    seq = seq[:n_samples]
    np.testing.assert_equal(
        samples[0][0],
        [seq, seq + 5, seq + 10, seq + 15],
    )
    for sx, sy in samples:
        np.testing.assert_equal(sy, sx[0])


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
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 10, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(10)
    samples = bootstrap.resample(x, **kwargs)
    assert len(samples) == 1
    assert samples[0].shape == (n_iter, n_samples)
    np.testing.assert_equal(samples[0][0], seq[:n_samples])


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
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 10, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(10)
    y = 10 - x
    random.seed(2)
    samples = bootstrap.resample(x, y, **kwargs)
    assert len(samples) == 2
    assert samples[0].shape == (n_iter, n_samples)
    assert samples[1].shape == (n_iter, n_samples)
    np.testing.assert_equal(samples[0][0], seq[:n_samples])
    np.testing.assert_equal(samples[1], 10 - samples[0])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_resample_dim(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(20).reshape(4, 5)
    random.seed(2)
    samples = bootstrap.resample(x, dim=1, **kwargs)
    assert len(samples) == 1
    assert samples[0].shape == (n_iter, 4, n_samples)
    seq = seq[:n_samples]
    np.testing.assert_equal(
        samples[0][0],
        [seq, seq + 5, seq + 10, seq + 15],
    )


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_resample_dim_2(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(20).reshape(4, 5)
    y = 20 - x
    random.seed(2)
    samples = bootstrap.resample(x, y, dim=1, **kwargs)
    assert len(samples) == 2
    assert samples[0].shape == (n_iter, 4, n_samples)
    assert samples[1].shape == (n_iter, 4, n_samples)
    seq = seq[:n_samples]
    np.testing.assert_equal(
        samples[0][0],
        [seq, seq + 5, seq + 10, seq + 15],
    )
    np.testing.assert_equal(samples[1], 20 - samples[0])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_resample_dim_2diff(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(20).reshape(4, 5)
    y = np.arange(5)
    random.seed(2)
    samples = bootstrap.resample(x, y, dim=[1, 0], **kwargs)
    assert len(samples) == 2
    assert samples[0].shape == (n_iter, 4, n_samples)
    assert samples[1].shape == (n_iter, n_samples)
    seq = seq[:n_samples]
    np.testing.assert_equal(
        samples[0][0],
        [seq, seq + 5, seq + 10, seq + 15],
    )
    np.testing.assert_equal(samples[1], samples[0][:, 0, :])


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
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 10, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(10)
    samples = bootstrap.resample(x, out_dim=1, **kwargs)
    assert len(samples) == 1
    assert samples[0].shape == (n_samples, n_iter)
    np.testing.assert_equal(samples[0][:, 0], seq[:n_samples])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_resample_dim_out_dim(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(20).reshape(4, 5)
    samples = bootstrap.resample(x, dim=1, out_dim=1, **kwargs)
    assert len(samples) == 1
    assert samples[0].shape == (4, n_iter, n_samples)
    np.testing.assert_equal(samples[0][0, 0, :], seq[:n_samples])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_10), (_CustomRandRange, CSEQ_10)],
    ids=["default", "custom"],
)
def test_bootstrap(n_iter, n_samples, rand, seq):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 10, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(10)
    values = bootstrap.bootstrap(np.mean, x, **kwargs)
    assert values.shape == (n_iter,)
    np.testing.assert_allclose(values[0], np.mean(seq[:n_samples]))


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
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 10, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(10)
    y = 10 - x
    random.seed(2)
    values = bootstrap.bootstrap((lambda a, b: np.mean(a) - np.mean(b)), x, y, **kwargs)
    assert values.shape == (n_iter,)
    np.testing.assert_allclose(values[0], 2 * np.mean(seq[:n_samples]) - 10)


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_bootstrap_dim(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(20).reshape(4, 5)
    random.seed(2)
    values = bootstrap.bootstrap((lambda a: np.mean(a, axis=1)), x, dim=1, **kwargs)
    assert values.shape == (n_iter, 4)
    m = np.mean(seq[:n_samples])
    np.testing.assert_allclose(values[0], [m, m + 5, m + 10, m + 15])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_bootstrap_dim_2(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(20).reshape(4, 5)
    y = 20 - x
    random.seed(2)
    values = bootstrap.bootstrap((lambda a, b: np.mean(a, axis=1) - np.mean(b, axis=1)), x, y, dim=1, **kwargs)
    assert values.shape == (n_iter, 4)
    m = np.mean(seq[:n_samples])
    np.testing.assert_allclose(values[0], [2 * m - 20, 2 * m - 10, 2 * m, 2 * m + 10])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_bootstrap_dim_2diff(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(20).reshape(4, 5)
    y = np.arange(5)
    random.seed(2)
    values = bootstrap.bootstrap((lambda a, b: np.mean(a, axis=1) - np.mean(b)), x, y, dim=[1, 0], **kwargs)
    assert values.shape == (n_iter, 4)
    np.testing.assert_allclose(values[0], [0, 5, 10, 15])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples", [None, 2, 6, 15])
@pytest.mark.parametrize(
    "rand, seq",
    [(_default_randrange, SEQ_5), (_CustomRandRange, CSEQ_5)],
    ids=["default", "custom"],
)
def test_bootstrap_dim_out_dim(
    n_iter: int | None,
    n_samples: int | None,
    rand: Callable[[int], Callable[[int], int] | None],
    seq: np.ndarray,
):
    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, 5, kwargs)
    _populate_kwarg("randrange", rand(2), None, kwargs)

    x = np.arange(20).reshape(4, 5)
    values = bootstrap.bootstrap((lambda a: np.mean(a, axis=1)), x, dim=1, out_dim=1, **kwargs)
    assert values.shape == (4, n_iter)
    m = np.mean(seq[:n_samples])
    np.testing.assert_allclose(values[:, 0], [m, m + 5, m + 10, m + 15])
