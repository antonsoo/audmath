import math

import numpy as np
import pytest

import audmath


@pytest.mark.parametrize(
    "x, axis, keepdims, expected",
    [
        ([], None, False, 0.0),
        ([], 0, False, 0.0),
        ([], None, True, np.array([0.0])),
        ([], 0, True, np.array([0.0])),
        (np.array([]), None, False, 0.0),
        (np.array([]), 0, False, 0.0),
        (np.array([]), None, True, np.array([0.0])),
        (np.array([]), 0, True, np.array([0.0])),
        (np.array([[]]), None, False, 0.0),
        (np.array([[]]), 0, False, 0.0),
        (np.array([[]]), 1, False, 0.0),
        (np.array([[]]), None, True, np.array([[0.0]])),
        (np.array([[]]), 0, True, np.array([[0.0]])),
        (np.array([[]]), 1, True, np.array([[0.0]])),
        (0, None, False, 0.0),
        (0.5, None, False, 0.5),
        (3, None, False, 3.0),
        ([3], None, False, 3.0),
        ([3], 0, False, 3.0),
        ([3], None, True, np.array([3.0])),
        ([3], 0, True, np.array([3.0])),
        (np.array([3]), None, False, 3.0),
        (np.array([3]), 0, False, 3.0),
        (np.array([3]), None, True, np.array([3.0])),
        (np.array([3]), 0, True, np.array([3.0])),
        (np.array([[3]]), None, False, 3.0),
        (np.array([[3]]), 0, False, 3.0),
        (np.array([[3]]), None, True, np.array([[3.0]])),
        (np.array([[3]]), 0, True, np.array([[3.0]])),
        ([0, 1, 2, 3], None, False, 1.8708286933869707),
        ([0, 1, 2, 3], 0, False, 1.8708286933869707),
        ([0, 1, 2, 3], None, True, np.array([1.8708286933869707])),
        ([0, 1, 2, 3], 0, True, np.array([1.8708286933869707])),
        (np.array([0, 1, 2, 3]), None, False, 1.8708286933869707),
        (np.array([0, 1, 2, 3]), 0, False, 1.8708286933869707),
        (np.array([0, 1, 2, 3]), None, True, np.array([1.8708286933869707])),
        (np.array([0, 1, 2, 3]), 0, True, np.array([1.8708286933869707])),
        (
            [[0, 1], [2, 3]],
            None,
            False,
            1.8708286933869707,
        ),
        (
            [[0, 1], [2, 3]],
            0,
            False,
            np.array([1.4142135623730951, 2.23606797749979]),
        ),
        (
            [[0, 1], [2, 3]],
            1,
            False,
            np.array([0.7071067811865476, 2.5495097567963922]),
        ),
        (
            [[0, 1], [2, 3]],
            None,
            True,
            np.array([[1.8708286933869707]]),
        ),
        (
            [[0, 1], [2, 3]],
            0,
            True,
            np.array([[1.4142135623730951], [2.23606797749979]]).T,
        ),
        (
            [[0, 1], [2, 3]],
            1,
            True,
            np.array([[0.7071067811865476], [2.5495097567963922]]),
        ),
        pytest.param(  # array with dim=0 has no axis
            3,
            0,
            False,
            3.0,
            marks=pytest.mark.xfail(raises=np.exceptions.AxisError),
        ),
        pytest.param(  # array with dim=0 has no axis
            3,
            0,
            True,
            3.0,
            marks=pytest.mark.xfail(raises=np.exceptions.AxisError),
        ),
        (2**32, None, False, float(2**32)),
        ([2**32, 2**32], None, False, float(2**32)),
    ],
)
def test_rms(x, axis, keepdims, expected):
    y = audmath.rms(x, axis=axis, keepdims=keepdims)
    np.testing.assert_array_equal(y, expected)
    if isinstance(y, np.ndarray):
        assert np.issubdtype(y.dtype, np.floating)
    else:
        assert np.issubdtype(type(y), np.floating)


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
@pytest.mark.parametrize("keepdims", [False, True])
def test_rms_integer(dtype, keepdims):
    limits = np.iinfo(dtype)
    x = np.array([[limits.max, 1], [limits.min, 0]], dtype=dtype)
    original = x.copy()
    # Python integers give exact squares even at the dtype limits.
    expected = np.array(
        [math.sqrt(sum(int(v) ** 2 for v in row) / len(row)) for row in x]
    )
    if keepdims:
        expected = expected[:, np.newaxis]
    np.testing.assert_allclose(
        audmath.rms(x, axis=1, keepdims=keepdims), expected, rtol=1e-15
    )
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rms_float_dtype(dtype):
    x = np.array([3, 4], dtype=dtype)
    y = audmath.rms(x)
    assert y.dtype == dtype
    np.testing.assert_allclose(y, math.sqrt(12.5))
