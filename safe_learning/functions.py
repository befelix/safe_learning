"""An efficient implementation of Delaunay triangulation on regular grids."""

from __future__ import absolute_import, print_function, division

from collections import Sequence

import numpy as np
from scipy import spatial, sparse, interpolate, linalg
from sklearn.utils.extmath import cartesian

from .utilities import linearly_spaced_combinations


__all__ = ['DeterministicFunction', 'Triangulation', 'PiecewiseConstant',
           'GridWorld', 'UncertainFunction', 'FunctionStack',
           'QuadraticFunction', 'GPyGaussianProcess', 'sample_gp_function']


class Function(object):
    """A generic function class."""

    def __init__(self):
        super(Function, self).__init__()
        self.ndim = None

    def __call__(self, *points):
        """Equivalent to `self.evaluate()`.

        Parameters
        ----------
        points : ndarray

        Returns
        -------
        array : ndarray
            The values of the evaluated function.
        """
        return self.evaluate(*points)

    @classmethod
    def from_callable(cls, function, gradient=None):
        """Create a deterministic function from a callable.

        Parameters
        ----------
        function : callable
            A function that we want to evaluate.

        gradient : callable, optional
            A callable that returns the gradient

        Returns
        -------
        instance of DeterministicFunction
        """
        instance = cls()
        instance.evaluate = function
        if gradient is not None:
            instance.gradient = gradient
        return instance


class UncertainFunction(Function):
    """Base class for function approximators."""

    def __init__(self):
        """Initialization, see `UncertainFunction`."""
        super(UncertainFunction, self).__init__()

    def to_mean_function(self):
        """Turn the uncertain function into a deterministic 'mean' function."""
        def _only_first_output(function):
            """Remove all but the first output of a function.

            Parameters
            ----------
            function : callable

            Returns
            -------
            function : callable
                The modified function.
            """
            def new_function(*points):
                return function(*points)[0]
            return new_function

        new_evaluate = _only_first_output(self.evaluate)
        new_gradient = _only_first_output(self.gradient)

        return DeterministicFunction.from_callable(new_evaluate, new_gradient)

    def evaluate(self, *points):
        """Return the distribution over function values.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        mean : ndarray
            A 2D array with the expected function values at the points.
        error_bounds : ndarray
            Error bounds for each dimension of the estimate.
        """
        raise NotImplementedError()

    def gradient(self, *points):
        """Return the distribution over the gradient.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        mean : ndarray
            The expected function gradient at the points.
        var : ndarray
            Error bounds for each dimension of the estimate.
        """
        raise NotImplementedError()


class DeterministicFunction(Function):
    """Base class for function approximators."""

    def __init__(self):
        """Initialization, see `Function` for details."""
        super(DeterministicFunction, self).__init__()

    def evaluate(self, *points):
        """Return the function values.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        values : ndarray
            A 2D array with the function values at the points.
        """
        raise NotImplementedError()

    def gradient(self, *points):
        """Return the gradient.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.
        Returns
        -------
        gradient : ndarray
            The function gradient at the points.
        """
        raise NotImplementedError()


class FunctionStack(UncertainFunction):
    """A combination of multiple 1d (uncertain) functions for each dim.

    Parameters
    ----------
    functions : list
        The functions. There should be one for each dimension of the output.
    """

    def __init__(self, *functions):
        """Initialization, see `FunctionStack`."""
        super(FunctionStack, self).__init__()
        self.functions = functions
        self.deterministic = [isinstance(fun, DeterministicFunction)
                              for fun in functions]
        self.deterministic = np.array(self.deterministic)
        self.num_fun = len(self.functions)

    def evaluate(self, *points):
        """Evaluation, see `UncertainFunction.evaluate`."""
        mean = np.empty((len(points), self.num_fun), dtype=np.float)
        error = np.empty_like(mean)
        error[:, self.deterministic] = 0.

        for i, (fun, deterministic) in enumerate(
                zip(self.functions, self.deterministic)):
            prediction = fun.evaluate(*points)
            if deterministic:
                mean[:, i] = prediction.squeeze()
            else:
                mean[:, i] = prediction[0].squeeze()
                error[:, i] = prediction[1].squeeze()

        return mean, error

    def gradient(self, *points):
        """Gradient, see `UncertainFunction.gradient`."""
        super(FunctionStack, self).gradient(*points)


def concatenate_inputs(start):
    """Concatenate the numpy array inputs to the functions.

    Parameters
    ----------
    start : int
        The attribute number at which to start concatenating.
    """
    def wrap(function):
        def wrapped_function(*args, **kwargs):
            """A function that concatenates inputs."""
            to_concatenate = map(np.atleast_2d, args[start:])

            if len(to_concatenate) == 1:
                concatenated = to_concatenate
            else:
                concatenated = [np.hstack(to_concatenate)]

            args = list(args[:start]) + concatenated
            return function(*args, **kwargs)

        return wrapped_function

    return wrap


class GPyGaussianProcess(UncertainFunction):
    """An `UncertainFunction` for GPy Gaussian processes.

    Parameters
    ----------
    gaussian_process : instance of GPy.core.GP
        The Gaussian process model.
    beta : float
        The scaling factor for the standard deviation to create confidence
        intervals.

    Notes
    -----
    The evaluate and gradient functions can be called with multiple arguments,
    in which case they are concatenated before being passed to the GP.
    """

    def __init__(self, gaussian_process, beta=2.):
        """Initialization, see `FakeGP`."""
        super(GPyGaussianProcess, self).__init__()
        self.ndim = gaussian_process.input_dim
        self.gaussian_process = gaussian_process

        if callable(beta):
            self.beta = beta
        else:
            self.beta = lambda t: beta

    @concatenate_inputs(start=1)
    def evaluate(self, points):
        """Return the distribution over function values.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        mean : ndarray
            The expected function values at the points.
        error_bounds : ndarray
            Error bounds for each dimension of the estimate.
        """
        mean, var = self.gaussian_process.predict_noiseless(points)
        t = len(self.gaussian_process.X)
        return mean, self.beta(t) * np.sqrt(var)

    @concatenate_inputs(start=1)
    def gradient(self, points):
        """Return the distribution over the gradient.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        mean : ndarray
            The expected function gradient at the points.
        var : ndarray
            Error bounds for each dimension of the estimate.
        """
        mean, var = self.gaussian_process.predict_jacobian(points)

        t = len(self.gaussian_process.X)
        return mean, self.beta(t) * np.sqrt(var)

    def add_data_point(self, x, y):
        """Add data points to the GP model.

        Parameters
        ----------
        x : ndarray
            A 2d array with the new states to add to the GP model. Each new
            state is on a new row.
        y : ndarray
            A 2d array with the new measurements to add to the GP model. Each
            measurements is on a new row.
        """
        x_new = np.vstack((self.gaussian_process.X, x))
        y_new = np.vstack((self.gaussian_process.Y, y))
        self.gaussian_process.set_XY(x_new, y_new)


class ScipyDelaunay(spatial.Delaunay):
    """
    A dummy triangulation on a regular grid, very inefficient.

    Warning: The internal indexing is different from the one used in our
    implementation!

    Parameters
    ----------
    limits: array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d array-like
        The number of points with which to grid each dimension.
    """

    def __init__(self, limits, num_points):
        self.numpoints = num_points
        self.limits = np.asarray(limits, dtype=np.float)
        params = [np.linspace(limit[0], limit[1], n) for limit, n in
                  zip(limits, num_points)]
        output = np.meshgrid(*params)
        points = np.array([par.ravel() for par in output]).T
        super(ScipyDelaunay, self).__init__(points)


class DimensionError(Exception):
    pass


class GridWorld(object):
    """Base class for function approximators on a regular grid.

    Parameters
    ----------
    limits: 2d array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d array-like
        The number of points with which to grid each dimension.
    """

    def __init__(self, limits, num_points):
        """Initialization, see `GridWorld`."""
        super(GridWorld, self).__init__()

        self.limits = np.asarray(limits, dtype=np.float)

        if not (isinstance(num_points, Sequence) or
                isinstance(num_points, np.ndarray)):
            num_points = [num_points] * len(limits)
        self.num_points = np.asarray(num_points, dtype=np.int)

        if np.any(self.num_points < 2):
            raise DimensionError('There must be at least 2 points in each '
                                 'dimension.')

        # Compute offset and unit hyperrectangle
        self.offset = self.limits[:, 0]
        self.unit_maxes = ((self.limits[:, 1] - self.offset)
                           / (self.num_points - 1))
        self.offset_limits = np.stack((np.zeros_like(self.limits[:, 0]),
                                       self.limits[:, 1] - self.offset),
                                      axis=1)

        # Statistics about the grid
        self.nrectangles = np.prod(self.num_points - 1)
        self.nindex = np.prod(self.num_points)
        self.ndim = len(limits)

    @property
    def all_points(self):
        """Return all the discrete points of the discretization.

        Returns
        -------
        points : ndarray
            An array with all the discrete points with size
            (self.nindex, self.ndim).
        """
        return self.index_to_state(np.arange(self.nindex))

    def __len__(self):
        """Return the number of points in the discretization."""
        return self.nindex

    def _check_dimensions(self, states):
        """Raise an error if the states have the wrong dimension.

        Parameters
        ----------
        states : ndarray
        """
        if not states.shape[1] == self.ndim:
            raise DimensionError('the input argument has the wrong '
                                 'dimensions.')

    def _center_states(self, states, clip=True):
        """Center the states to the interval [0, x].

        Parameters
        ----------
        states : np.array
        clip : bool, optinal
            If False the data is not clipped to lie within the limits.

        Returns
        -------
        offset_states : ndarray
        """
        states = np.atleast_2d(states) - self.offset[None, :]
        eps = np.finfo(states.dtype).eps
        if clip:
            np.clip(states,
                    self.offset_limits[:, 0] + 2 * eps,
                    self.offset_limits[:, 1] - 2 * eps,
                    out=states)
        return states

    def index_to_state(self, indices):
        """Convert indices to physical states.

        Parameters
        ----------
        indices : ndarray (int)
            The indices of points on the discretization.

        Returns
        -------
        states : ndarray
            The states with physical units that correspond to the indices.
        """
        indices = np.atleast_1d(indices)
        ijk_index = np.vstack(np.unravel_index(indices, self.num_points)).T
        return ijk_index * self.unit_maxes + self.offset

    def state_to_index(self, states):
        """Convert physical states to indices.

        Parameters
        ----------
        states: ndarray
            Physical states on the discretization.

        Returns
        -------
        indices: ndarray (int)
            The indices that correspond to the physical states.
        """
        states = np.atleast_2d(states)
        self._check_dimensions(states)
        states = np.clip(states, self.limits[:, 0], self.limits[:, 1])
        states = (states - self.offset) / self.unit_maxes
        ijk_index = np.rint(states).astype(np.int)
        return np.ravel_multi_index(ijk_index.T, self.num_points)

    def state_to_rectangle(self, states, offset=True):
        """Convert physical states to its closest rectangle index.

        Parameters
        ----------
        states : ndarray
            Physical states on the discretization.
        offset : bool, optional
            If False the data is assumed to be already centered and clipped.

        Returns
        -------
        rectangles : ndarray (int)
            The indices that correspond to rectangles of the physical states.
        """
        states = np.atleast_2d(states)
        # clip to domain (find closest rectangle)
        if offset:
            self._check_dimensions(states)
            states = self._center_states(states, clip=True)

        ijk_index = np.floor_divide(states, self.unit_maxes).astype(np.int)
        return np.ravel_multi_index(ijk_index.T, self.num_points - 1)

    def rectangle_to_state(self, rectangles):
        """
        Convert rectangle indices to the states of the bottem-left corners.

        Parameters
        ----------
        rectangles : ndarray (int)
            The indices of the rectangles

        Returns
        -------
        states : ndarray
            The states that correspond to the bottom-left corners of the
            corresponding rectangles.
        """
        rectangles = np.atleast_1d(rectangles)
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        return (ijk_index.T * self.unit_maxes) + self.offset

    def rectangle_corner_index(self, rectangles):
        """Return the index of the bottom-left corner of the rectangle.

        Parameters
        ----------
        rectangles: ndarray
            The indices of the rectangles.

        Returns
        -------
        corners : ndarray (int)
            The indices of the bottom-left corners of the rectangles.
        """
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        return np.ravel_multi_index(np.atleast_2d(ijk_index),
                                    self.num_points)


class PiecewiseConstant(GridWorld, DeterministicFunction):
    """A piecewise constant function approximator.

    Parameters
    ----------
    limits: 2d array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d arraylike
        The number of points with which to grid each dimension.
    vertex_values: arraylike, optional
        A 2D array with the values at the vertices of the grid on each row.
    """

    def __init__(self, limits, num_points, vertex_values=None):
        """Initialization, see `PiecewiseConstant`."""
        super(PiecewiseConstant, self).__init__(limits, num_points)

        self._vertex_values = None
        self.vertex_values = vertex_values

    @property
    def vertex_values(self):
        """Return the vertex values."""
        return self._vertex_values

    @vertex_values.setter
    def vertex_values(self, values):
        """Set the vertex values."""
        if values is None:
            self._vertex_values = values
        else:
            values = np.asarray(values).reshape(self.nindex, -1)
            self._vertex_values = values

    def evaluate(self, points):
        """Return the function values.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        values : ndarray
            The function values at the points.
        """
        nodes = self.state_to_index(points)
        return self.vertex_values[nodes]

    def evaluate_constraint(self, points):
        """
        Obtain function values at points from triangulation.

        This function returns a sparse matrix that, when multiplied
        with the vector with all the function values on the vertices,
        returns the function values at points.

        Parameters
        ----------
        points : ndarray
            A 2d array where each row represents one point.

        Returns
        -------
        values
            A sparse matrix B so that evaluate(points) = B.dot(vertex_values).
        """
        npoints = len(points)
        weights = np.ones(npoints, dtype=np.int)
        rows = np.arange(npoints)
        cols = self.state_to_index(points)
        return sparse.coo_matrix((weights, (rows, cols)),
                                 shape=(npoints, self.nindex))

    def gradient(self, points):
        """Return the gradient.

        The gradient is always zero for piecewise constant functions!

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.
        Returns
        -------
        gradient : ndarray
            The function gradient at the points.
        """
        return np.broadcast_to(0, (len(points), self.ndim))


class _Delaunay1D(object):
    """A simple class that behaves like scipy.Delaunay for 1D inputs.

    Parameters
    ----------
    points : ndarray
        Coordinates of points to triangulate, shape (2, 1).
    """

    def __init__(self, points):
        """Initialization, see `_Delaunay1D`."""
        if points.shape[1] > 1:
            raise AttributeError('This only works for 1D inputs.')
        if points.shape[0] > 2:
            raise AttributeError('This only works for two points')

        self.points = points
        self.nsimplex = len(points) - 1

        self._min = np.min(points)
        self._max = np.max(points)

        self.simplices = np.array([[0, 1]])

    def find_simplex(self, points):
        """Find the simplices containing the given points.

        Parameters
        ----------
        points : ndarray
            2D array of coordinates of points for which to find simplices.

        Returns
        -------
        indices : ndarray
            Indices of simplices containing each point. Points outside the
            triangulation get the value -1.
        """
        points = points.squeeze()
        out_of_bounds = points > self._max
        out_of_bounds |= points < self._min
        return np.where(out_of_bounds, -1, 0)


class Triangulation(GridWorld, DeterministicFunction):
    """
    Efficient Delaunay triangulation on regular grids.

    This class is a wrapper around scipy.spatial.Delaunay for regular grids. It
    splits the space into regular hyperrectangles and then computes a Delaunay
    triangulation for only one of them. This single triangulation is then
    generalized to other hyperrectangles, without ever maintaining the full
    triangulation for all individual hyperrectangles.

    Parameters
    ----------
    limits: arraylike
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)].
    num_points: arraylike
        1D array with the number of points with which to grid each dimension.
    vertex_values: arraylike, optional
        A 2D array with the values at the vertices of the grid on each row.
    project: bool, optional
        Whether to project points onto the limits.
    """

    def __init__(self, limits, num_points, vertex_values=None, project=False):
        """Initialization."""
        super(Triangulation, self).__init__(limits, num_points)

        self._vertex_values = None
        self.vertex_values = vertex_values

        # Get triangulation
        if len(limits) == 1:
            corners = np.array([[0], self.unit_maxes])
            self.triangulation = _Delaunay1D(corners)
        else:
            hyperrectangle_corners = cartesian(np.diag(self.unit_maxes))
            self.triangulation = spatial.Delaunay(hyperrectangle_corners)
        self.unit_simplices = self._triangulation_simplex_indices()

        # Some statistics about the triangulation
        self.nsimplex = self.triangulation.nsimplex * self.nrectangles

        # Parameters for the hyperplanes of the triangulation
        self.hyperplanes = None
        self._update_hyperplanes()

        self.project = project

    @property
    def vertex_values(self):
        """Return the vertex values."""
        return self._vertex_values

    @vertex_values.setter
    def vertex_values(self, values):
        """Set the vertex values."""
        if values is None:
            self._vertex_values = values
        else:
            values = np.asarray(values).reshape(self.nindex, -1)
            self._vertex_values = values

    def _triangulation_simplex_indices(self):
        """Return the simplex indices in our coordinates.

        Returns
        -------
        simplices: ndarray (int)
            The simplices array in our extended coordinate system.

        Notes
        -----
        This is only used once in the initialization.
        """
        simplices = self.triangulation.simplices
        new_simplices = np.empty_like(simplices)

        # Convert the points to out indices
        index_mapping = self.state_to_index(self.triangulation.points +
                                            self.offset)

        # Replace each index with out new_index in index_mapping
        for i, new_index in enumerate(index_mapping):
            new_simplices[simplices == i] = new_index
        return new_simplices

    def _update_hyperplanes(self):
        """Compute the simplex hyperplane parameters on the triangulation."""
        self.hyperplanes = np.empty((self.triangulation.nsimplex,
                                     self.ndim, self.ndim),
                                    dtype=np.float)

        # Use that the bottom-left rectangle has the index zero, so that the
        # index numbers of scipy correspond to ours.
        for i, simplex in enumerate(self.unit_simplices):
            simplex_points = self.index_to_state(simplex)
            self.hyperplanes[i] = np.linalg.inv(simplex_points[1:] -
                                                simplex_points[:1])

    def find_simplex(self, points):
        """Find the simplices corresponding to points.

        Parameters
        ----------
        points : 2darray

        Returns
        -------
        simplices : np.array (int)
            The indices of the simplices
        """
        points = self._center_states(points, clip=True)

        # Convert to basic hyperrectangle coordinates and find simplex
        unit_coordinates = points % self.unit_maxes
        simplex_ids = self.triangulation.find_simplex(unit_coordinates)
        simplex_ids = np.atleast_1d(simplex_ids)

        # Adjust for the hyperrectangle index
        rectangles = self.state_to_rectangle(points, offset=False)
        simplex_ids += rectangles * self.triangulation.nsimplex

        return simplex_ids

    def simplices(self, indices):
        """Return the simplices corresponding to the simplex index.

        Parameters
        ----------
        indices : ndarray
            The indices of the simpleces

        Returns
        -------
        simplices : ndarray
            Each row consists of the indices of the simplex corners.
        """
        # Get the indices inside the unit rectangle
        unit_indices = np.remainder(indices, self.triangulation.nsimplex)
        simplices = self.unit_simplices[unit_indices].copy()

        # Shift indices to corresponding rectangle
        rectangles = np.floor_divide(indices, self.triangulation.nsimplex)
        corner_index = self.rectangle_corner_index(rectangles)

        if simplices.ndim > 1:
            corner_index = corner_index[:, None]

        simplices += corner_index
        return simplices

    def _get_weights(self, points):
        """Return the linear weights associated with points.

        Parameters
        ----------
        points : 2d array
            Each row represents one point

        Returns
        -------
        weights : ndarray
            An array that contains the linear weights for each point.
        simplices : ndarray
            The indeces of the simplices associated with each points
        """
        simplex_ids = self.find_simplex(points)

        if self.project:
            points = np.clip(points, self.limits[:, 0], self.limits[:, 1])

        simplices = self.simplices(simplex_ids)
        origins = self.index_to_state(simplices[:, 0])

        # Get hyperplane equations
        simplex_ids %= self.triangulation.nsimplex
        hyperplanes = self.hyperplanes[simplex_ids]

        # Some numbers for convenience
        nsimp = self.ndim + 1
        npoints = len(points)

        weights = np.empty((npoints, nsimp), dtype=np.float)

        # Pre-multiply each hyperplane by (point - origin)
        np.einsum('ij,ijk->ik', points - origins, hyperplanes,
                  out=weights[:, 1:])
        # The weights have to add up to one
        weights[:, 0] = 1 - np.sum(weights[:, 1:], axis=1)

        return weights, simplices

    def evaluate(self, points):
        """Return the function values.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        values : ndarray
            The function values at the points.
        """
        weights, simplices = self._get_weights(points)

        # Return function values if desired
        result = np.einsum('ij,ijk->ik',
                           weights,
                           self.vertex_values[simplices])
        return result

    def evaluate_constraint(self, points):
        """
        Obtain function values at points from triangulation.

        This function returns a sparse matrix that, when multiplied
        with the vector with all the function values on the vertices,
        returns the function values at points.

        Parameters
        ----------
        points : 2d array
            Each row represents one point.

        Returns
        -------
        values
            A sparse matrix B so that evaluate(points) = B.dot(vertex_values).
        """
        weights, simplices = self._get_weights(points)
        # Construct sparse matrix for optimization

        nsimp = self.ndim + 1
        npoints = len(simplices)
        # Indices of constraints (nsimp points per simplex, so we have nsimp
        #  values in each row; one for each simplex)
        rows = np.repeat(np.arange(len(points)), nsimp)
        cols = simplices.ravel()

        return sparse.coo_matrix((weights.ravel(), (rows, cols)),
                                 shape=(npoints, self.nindex))

    def _get_weights_gradient(self, points=None, indices=None):
        """Return the linear gradient weights associated with points.

        Parameters
        ----------
        points : ndarray
            Each row represents one point.
        indices : ndarray
            Each row represents one index. Ignored if points

        Returns
        -------
        weights : ndarray
            An array that contains the linear weights for each point.
        simplices : ndarray
            The indeces of the simplices associated with each points
        """
        if points is None:
            simplex_ids = indices
        else:
            simplex_ids = self.find_simplex(points)
        simplices = self.simplices(simplex_ids)

        # Get hyperplane equations
        simplex_ids %= self.triangulation.nsimplex

        # Some numbers for convenience
        nsimp = self.ndim + 1
        npoints = len(simplex_ids)

        # weights
        weights = np.empty((npoints, self.ndim, nsimp), dtype=np.float)

        weights[:, :, 1:] = self.hyperplanes[simplex_ids]
        weights[:, :, 0] = -np.sum(weights[:, :, 1:], axis=2)
        return weights, simplices

    def gradient(self, points):
        """Return the gradient.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.
        Returns
        -------
        gradient : ndarray
            The function gradient at the points. A 3D array with the gradient
            at the ith data points for the jth output with regard to the kth
            dimension stored at (i, j, k). The jth dimension is squeezed out
            for 1D functions.
        """
        weights, simplices = self._get_weights_gradient(points)
        # Return function values if desired
        res = np.einsum('ijk,ikl->ilj', weights, self.vertex_values[simplices])
        if res.shape[1] == 1:
            res = res.squeeze(axis=1)
        return res

    def gradient_constraint(self, points=None, indices=None):
        """
        Return the gradients at the respective points.

        This function returns a sparse matrix that, when multiplied
        with the vector of all the function values on the vertices,
        returns the gradients. Note that after the product you have to call
        ```np.reshape(grad, (ndim, -1))``` in order to obtain a proper
        gradient matrix.

        Parameters
        ----------
        points : ndarray
            Each row contains one state at which to evaluate the gradient.
        indices : ndarray
            The simplex indices. Ignored if points are provided.

        Returns
        -------
        gradient : scipy.sparse.coo_matrix
            A sparse matrix so that
            `grad(points) = B.dot(V(vertices)).reshape(ndim, -1)` corresponds
            to the true gradients
        """
        weights, simplices = self._get_weights_gradient(points=points,
                                                        indices=indices)

        # Some numbers for convenience
        nsimp = self.ndim + 1
        npoints = len(simplices)

        # Construct sparse matrix for optimization

        # Indices of constraints (ndim gradients for each point, which each
        # depend on the nsimp vertices of the simplex.
        rows = np.repeat(np.arange(npoints * self.ndim), nsimp)
        cols = np.tile(simplices, (1, self.ndim)).ravel()

        return sparse.coo_matrix((weights.ravel(), (rows, cols)),
                                 shape=(self.ndim * npoints, self.nindex))


class QuadraticFunction(DeterministicFunction):
    """A quadratic Lyapunov function.

    V(x) = x.T P x
    dV(x)/dx = 2 x.T P

    Parameters
    ----------
    matrix : np.array
        2d cost matrix for lyapunov function.
    """

    def __init__(self, matrix):
        """Initialization, see `QuadraticLyapunovFunction`."""
        super(QuadraticFunction, self).__init__()
        self.matrix = matrix

    @concatenate_inputs(start=1)
    def evaluate(self, points):
        """See `DeterministicFunction.evaluate`."""
        points = np.asarray(points)
        return np.sum(points.dot(self.matrix) * points, axis=1, keepdims=True)

    @concatenate_inputs(start=1)
    def gradient(self, points):
        """See `DeterministicFunction.gradient`."""
        points = np.asarray(points)
        return 2 * points.dot(self.matrix)


def sample_gp_function(kernel, bounds, num_samples, noise_var,
                       interpolation='linear', mean_function=None):
    """
    Sample a function from a gp with corresponding kernel within its bounds.

    Parameters
    ----------
    kernel : instance of GPy.kern.*
    bounds : list of tuples
        [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_samples : int or list
        If integer draws the corresponding number of samples in all
        dimensions and test all possible input combinations. If a list then
        the list entries correspond to the number of linearly spaced samples of
        the corresponding input
    noise_var : float
        Variance of the observation noise of the GP function
    interpolation : string
        If 'linear' interpolate linearly between samples, if 'kernel' use the
        corresponding mean RKHS-function of the GP.
    mean_function : callable
        Mean of the sample function

    Returns
    -------
    function : object
        function(x, noise=True)
        A function that takes as inputs new locations x to be evaluated and
        returns the corresponding noisy function values. If noise=False is
        set the true function values are returned (useful for plotting).
    """
    inputs = linearly_spaced_combinations(bounds, num_samples)
    cov = kernel.K(inputs) + np.eye(inputs.shape[0]) * 1e-6
    output = np.random.multivariate_normal(np.zeros(inputs.shape[0]),
                                           cov)

    if interpolation == 'linear':

        @concatenate_inputs(start=0)
        def evaluate_gp_function_linear(x, noise=True):
            """
            Evaluate the GP sample function with linear interpolation.

            Parameters
            ----------
            x : np.array
                2D array with inputs
            noise : bool
                Whether to include prediction noise
            """
            x = np.atleast_2d(x)
            y = interpolate.griddata(inputs, output, x, method='linear')
            y = np.atleast_2d(y)
            if mean_function is not None:
                y += mean_function(x)
            if noise:
                y += np.sqrt(noise_var) * np.random.randn(x.shape[0], 1)
            return y
        return evaluate_gp_function_linear
    elif interpolation == 'kernel':
        cho_factor = linalg.cho_factor(cov)
        alpha = linalg.cho_solve(cho_factor, output)

        @concatenate_inputs(start=0)
        def evaluate_gp_function_kernel(x, noise=True):
            """
            Evaluate the GP sample function with kernel interpolation.

            Parameters
            ----------
            x : np.array
                2D array with inputs
            noise : bool
                Whether to include prediction noise.
            """
            x = np.atleast_2d(x)
            y = kernel.K(x, inputs).dot(alpha)
            y = y[:, None]
            if mean_function is not None:
                y += mean_function(x)
            if noise:
                y += np.sqrt(noise_var) * np.random.randn(x.shape[0], 1)
            return y
        return evaluate_gp_function_kernel
