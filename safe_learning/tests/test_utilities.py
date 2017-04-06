"""Test the utilities."""

from __future__ import absolute_import, print_function, division

import numpy as np
from numpy.testing import assert_allclose

from safe_learning.utilities import dlqr


def test_dlqr():
    """Test the dlqr function."""

    true_k = np.array([[0.61803399]])
    true_p = np.array([[1.61803399]])

    k, p = dlqr(1, 1, 1, 1)
    assert_allclose(k, true_k)
    assert_allclose(p, true_p)

    k, p = dlqr([[1]], [[1]], [[1]], [[1]])
    assert_allclose(k, true_k)
    assert_allclose(p, true_p)
