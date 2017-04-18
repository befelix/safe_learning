"""Test the utilities."""

from __future__ import absolute_import, print_function, division

import pytest
import numpy as np
from numpy.testing import assert_allclose

from safe_learning.utilities import dlqr, get_storage, set_storage


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


class TestStorage(object):
    """Test the class storage."""

    @pytest.fixture
    def sample_class(self):
        """A sample class for testing."""
        class A(object):
            """Some class."""

            def __init__(self):
                """Initialize."""
                super(A, self).__init__()
                self.storage = {}

            def method(self, value):
                storage = get_storage(self.storage)
                set_storage(self.storage, [('value', value)])
                return storage

        return A()

    def test_storage(self, sample_class):
        """Test the storage."""
        storage = sample_class.method(5)
        assert storage is None
        storage = sample_class.method(4)
        assert storage['value'] == 5
        storage = sample_class.method(None)
        assert storage['value'] == 4
        storage = sample_class.method(None)
        assert storage['value'] is None
