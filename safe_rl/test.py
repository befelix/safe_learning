from __future__ import division, print_function, absolute_import

from numpy.testing import *
import unittest
import numpy as np

from .triangulation import Triangulation, Delaunay


class DelaunayTest(TestCase):
    """Test the generalized Delaunay triangulation"""

    def test_find_simplex(self):
        limits = [[-1, 1], [-1, 2]]
        num_points = [2, 6]
        delaunay = Delaunay(limits, num_points)

        assert_equal(delaunay.nrectangles, 2 * 6)
        assert_equal(delaunay.ndim, 2)
        assert_equal(delaunay.nsimplex, 2 * 2 * 6)
        assert_equal(delaunay.offset, np.array([-1, -1]))
        assert_equal(delaunay.maxes, np.array([2, 3]) / np.array(num_points))

        lower = delaunay.triangulation.find_simplex(np.array([0, 0])).squeeze()
        upper = 1 - lower

        test_points = np.array([[-1, -1],
                                [-0.1, -0.55],
                                [0.1, -1],
                                [-0.1, -0.1]])

        true_result = np.array([lower, upper, 2 + lower, 4 + upper])


        result = delaunay.find_simplex(test_points)

        assert_allclose(result, true_result)

    def test_index_state_conversion(self):
        limits = [[-1.1, 1.5], [2.2, 2.4]]
        num_points = [7, 8]
        delaunay = Delaunay(limits, num_points)

        # Forward and backwards convert all indeces
        indeces = np.arange(delaunay.max_index)
        states = delaunay.index_to_state(indeces)
        indeces2 = delaunay.state_to_index(states)
        assert_equal(indeces, indeces2)

        # test 1D input
        delaunay.state_to_index([0, 2.3])
        delaunay.index_to_state(1)



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
