from __future__ import division, print_function, absolute_import

from numpy.testing import *
import unittest
import numpy as np

from .triangulation import Triangulation


class TriangulationTest(TestCase):
    """Test the Triangulization method"""


    def test_values(self):
        points = np.array([[0, 0],
                           [1, 0],
                           [0, 1]])

        tri = Triangulation(points)

        test_points = np.vstack((points, np.array([[0.5, 0.5],
                                                   [0, 0.5],
                                                   [0.5, 0]])))

        H = tri.function_values_at(test_points).todense()

        true_H = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1],
                           [0, 0.5, 0.5],
                           [0.5, 0, 0.5],
                           [0.5, 0.5, 0]])

        assert_allclose(H, true_H)


if __name__ == '__main__':
    unittest.main()
