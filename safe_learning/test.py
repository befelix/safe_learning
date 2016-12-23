"""Unit tests for the safe_learning module."""

from __future__ import division, print_function, absolute_import

from numpy.testing import *
import unittest
import numpy as np

from .triangulation import (Delaunay, ScipyDelaunay, GridWorld,
                            PiecewiseConstant, Function, UncertainFunction)


class FunctionTest(TestCase):
    """Test the base class."""

    def test_errors(self):
        """Check notImplemented error."""
        f = Function()
        assert_raises(NotImplementedError, f.evaluate, None)
        assert_raises(NotImplementedError, f.gradient, None)


class UncertainFunctionTest(TestCase):
    """Test the base class."""

    def test_errors(self):
        """Check notImplemented error."""
        f = UncertainFunction()
        assert_raises(NotImplementedError, f.evaluate, None)
        assert_raises(NotImplementedError, f.gradient, None)


class ScipyDelaunayTest(TestCase):
    """Test the fake replacement for Scipy."""

    def test_init(self):
        """Test the initialization."""
        limits = [[-1, 1], [-1, 2]]
        num_points = [2, 6]
        sp_delaunay = ScipyDelaunay(limits, num_points)
        delaunay = Delaunay(limits, num_points)

        assert_equal(delaunay.nsimplex, sp_delaunay.nsimplex)
        assert_equal(delaunay.ndim, sp_delaunay.ndim)
        sp_delaunay.find_simplex(np.array([[0, 0]]))


class GridworldTest(TestCase):
    """Test the general GridWorld definitions."""

    def test_index_state_conversion(self):
        """Test all index conversions."""
        limits = [[-1.1, 1.5], [2.2, 2.4]]
        num_points = [7, 8]
        grid = GridWorld(limits, num_points)

        # Forward and backwards convert all indeces
        indeces = np.arange(grid.nindex)
        states = grid.index_to_state(indeces)
        indeces2 = grid.state_to_index(states)
        assert_equal(indeces, indeces2)

        # test 1D input
        grid.state_to_index([0, 2.3])
        grid.index_to_state(1)

        # Test rectangles
        rectangles = np.arange(grid.nrectangles)
        states = grid.rectangle_to_state(rectangles)
        rectangles2 = grid.state_to_rectangle(states + grid.unit_maxes / 2)
        assert_equal(rectangles, rectangles2)

        rectangle = grid.state_to_rectangle(100 * np.ones((1, 2)))
        assert_equal(rectangle, grid.nrectangles - 1)

        rectangle = grid.state_to_rectangle(-100 * np.ones((1, 2)))
        assert_equal(rectangle, 0)

        # Test rectangle corners
        corners = grid.rectangle_corner_index(rectangles)
        corner_states = grid.rectangle_to_state(rectangles)
        corners2 = grid.state_to_index(corner_states)
        assert_equal(corners, corners2)

        # Test point outside grid
        test_point = np.array([[-1.2, 2.]])
        index = grid.state_to_index(test_point)
        assert_equal(index, 0)

    def test_1d_numpoints(self):
        """Check 1-dimensional numpoints argument."""
        grid = GridWorld([[1, 2], [3, 4]], 2)
        assert_equal(grid.num_points, np.array([2, 2]))


class PiecewiseConstantTest(TestCase):
    """Test a piecewise constant function."""

    def test_init(self):
        """Test initialisation."""
        limits = [[-1, 1], [-1, 1]]
        npoints = 3
        pwc = PiecewiseConstant(limits, npoints, [1, 2, 3])
        assert_allclose(pwc.vertex_values, np.array([1, 2, 3]))

    def test_evaluation(self):
        """Simple tests."""
        limits = [[-1, 1], [-1, 1]]
        npoints = 3
        pwc = PiecewiseConstant(limits, npoints)

        vertex_points = pwc.index_to_state(np.arange(pwc.nindex))
        vertex_values = np.sum(vertex_points, axis=1)
        pwc.vertex_values = vertex_values

        test = pwc.evaluate(vertex_points)
        assert_allclose(test, vertex_values)

        outside_point = np.array([[-1.5, -1.5]])
        test1 = pwc.evaluate(outside_point)
        assert_allclose(test1, -2)

        # Test constraint evaluation
        test2 = pwc.evaluate_constraint(vertex_points)
        test2 = test2.toarray().dot(vertex_values)
        assert_allclose(test2, vertex_values)

    def test_gradient(self):
        """Test the gradient."""
        limits = [[-1, 1], [-1, 1]]
        npoints = 3
        pwc = PiecewiseConstant(limits, npoints)
        test_points = pwc.index_to_state(np.arange(pwc.nindex))
        gradient = pwc.gradient(test_points)
        assert_allclose(gradient, 0)


class DelaunayTest(TestCase):
    """Test the generalized Delaunay triangulation."""

    def test_find_simplex(self):
        """Test the implices on the grid."""
        limits = [[-1, 1], [-1, 2]]
        num_points = [2, 6]
        delaunay = Delaunay(limits, num_points)

        # Test the basic properties
        assert_equal(delaunay.nrectangles, 2 * 6)
        assert_equal(delaunay.ndim, 2)
        assert_equal(delaunay.nsimplex, 2 * 2 * 6)
        assert_equal(delaunay.offset, np.array([-1, -1]))
        assert_equal(delaunay.unit_maxes,
                     np.array([2, 3]) / np.array(num_points))
        assert_equal(delaunay.nrectangles, 2 * 6)

        # test the simplex indices
        lower = delaunay.triangulation.find_simplex(np.array([0, 0])).squeeze()
        upper = 1 - lower

        test_points = np.array([[0, 0],
                                [0.9, 0.45],
                                [1.1, 0],
                                [1.9, 2.9]])

        test_points += np.array(limits)[:, 0]

        true_result = np.array([lower, upper, 6 * 2 + lower, 11 * 2 + upper])
        result = delaunay.find_simplex(test_points)

        assert_allclose(result, true_result)

        # Test the ability to find simplices
        simplices = delaunay.simplices(result)
        true_simplices = np.array([[0, 1, 7],
                                   [1, 7, 8],
                                   [7, 8, 14],
                                   [13, 19, 20]])
        assert_equal(np.sort(simplices, axis=1), true_simplices)

        # Test point ouside domain (should map to bottom left and top right)
        assert_equal(lower, delaunay.find_simplex(np.array([[-100., -100.]])))
        assert_equal(delaunay.nsimplex - 1 - lower,
                     delaunay.find_simplex(np.array([[100., 100.]])))

    def test_values(self):
        """Test the function_value_at function."""
        eps = 1e-10

        delaunay = Delaunay([[0, 1], [0, 1]], [1, 1])

        test_points = np.array([[0, 0],
                                [1 - eps, 0],
                                [0, 1 - eps],
                                [0.5 - eps, 0.5 - eps],
                                [0, 0.5],
                                [0.5, 0]])
        nodes = delaunay.state_to_index(np.array([[0, 0],
                                                  [1, 0],
                                                  [0, 1]]))

        H = delaunay.evaluate_constraint(test_points).toarray()

        true_H = np.zeros((len(test_points), delaunay.nindex),
                          dtype=np.float)
        true_H[0, nodes[0]] = 1
        true_H[1, nodes[1]] = 1
        true_H[2, nodes[2]] = 1
        true_H[3, nodes[[1, 2]]] = 0.5
        true_H[4, nodes[[0, 2]]] = 0.5
        true_H[5, nodes[[0, 1]]] = 0.5

        assert_allclose(H, true_H, atol=1e-7)

        # Test value property
        values = np.random.rand(delaunay.nindex)
        delaunay.vertex_values = values
        v1 = H.dot(values)
        v2 = delaunay.evaluate(test_points)
        assert_allclose(v1, v2)

        # Test the projections
        test_point = np.array([-0.5, -0.5])
        delaunay.vertex_values = np.array([0, 1, 1])
        unprojected = delaunay.evaluate(test_point)
        delaunay.project = True
        projected = delaunay.evaluate(test_point)

        assert_allclose(projected, np.array([0, 0]))
        assert_allclose(unprojected, np.array([-1, -1]))

    def test_multiple_dimensions(self):
        """Test delaunay in three dimensions."""
        limits = [[0, 1]] * 3
        delaunay = Delaunay(limits, [1] * 3)
        assert_equal(delaunay.ndim, 3)
        assert_equal(delaunay.nrectangles, 1)
        assert_equal(delaunay.nsimplex, np.math.factorial(3))

        corner_points = np.array([[0, 0, 0],
                                  [1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1],
                                  [0, 1, 1],
                                  [1, 1, 0],
                                  [1, 0, 1],
                                  [1, 1, 1]], dtype=np.float)

        values = np.sum(delaunay.index_to_state(np.arange(8)), axis=1) / 3

        test_points = np.vstack((corner_points,
                                 np.array([[0, 0, 0.5],
                                           [0.5, 0, 0],
                                           [0, 0.5, 0],
                                           [0.5, 0.5, 0.5]])))
        corner_values = np.sum(corner_points, axis=1) / 3
        true_values = np.hstack((corner_values,
                                 np.array([1 / 6, 1 / 6, 1 / 6, 1 / 2])))

        delaunay.vertex_values = values
        result = delaunay.evaluate(test_points)
        assert_allclose(result, true_values, atol=1e-5)

    def test_gradient(self):
        """Test the gradient_at function."""
        delaunay = Delaunay([[0, 1], [0, 1]], [1, 1])

        points = np.array([[0, 0],
                           [1, 0],
                           [0, 1],
                           [1, 1]], dtype=np.int)
        nodes = delaunay.state_to_index(points)

        # Simplex with node values:
        # 3 - 1
        # | \ |
        # 1 - 2
        # --> x

        values = np.zeros(delaunay.nindex)
        values[nodes] = [1, 2, 3, 1]

        test_points = np.array([[0.01, 0.01],
                                [0.99, 0.99]])

        true_grad = np.array([[1, 2], [-2, -1]])

        # Construct true H (gradient as function of values)
        true_H = np.zeros((2 * delaunay.ndim, delaunay.nindex))

        true_H[0, nodes[[0, 1]]] = [-1, 1]
        true_H[1, nodes[[0, 2]]] = [-1, 1]
        true_H[2, nodes[[2, 3]]] = [-1, 1]
        true_H[3, nodes[[1, 3]]] = [-1, 1]

        # Evaluate gradient with and without values
        H = delaunay.gradient_constraint(test_points).toarray()
        delaunay.vertex_values = values
        grad = delaunay.gradient(test_points)

        # Compare
        # print(grad)
        assert_allclose(grad, true_grad)
        assert_allclose(H, true_H)
        # assert_allclose(true_grad, H.dot(values).reshape(-1, delaunay.ndim))


if __name__ == '__main__':
    unittest.main()
