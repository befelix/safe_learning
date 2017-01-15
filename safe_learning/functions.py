"""An efficient implementation of Delaunay triangulation on regular grids."""

from __future__ import absolute_import, print_function, division

from collections import Sequence

import numpy as np
from scipy import spatial, sparse
from sklearn.utils.extmath import cartesian


__all__ = ['DeterministicFunction', 'Triangulation', 'PiecewiseConstant',
           'GridWorld', 'UncertainFunction', 'FunctionStack',
           'QuadraticFunction', 'GPyGaussianProcess', 'as_function']


def as_function(function):
    """Convert a callable to a function."""
    if hasattr(function, 'evaluate'):
        return function
    else:
        return DeterministicFunction.from_callable(function)


class Function(object):
    """A generic function class."""

    def __init__(self):
        super(Function, self).__init__()
        self.ndim = None


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
            A 2D array with the expected function values at the points.
        error_bounds : ndarray
            Error bounds for each dimension of the estimate.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()


class DeterministicFunction(Function):
    """Base class for function approximators."""

    def __init__(self):
        """Initialization, see `Function` for details."""
        super(DeterministicFunction, self).__init__()

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
        instance = DeterministicFunction()
        function.__doc__ = DeterministicFunction.evaluate.__doc__
        instance.evaluate = function
        if gradient is not None:
            gradient.__doc__ = DeterministicFunction.gradient.__doc__
            instance.gradient = gradient
        return instance

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
            A 2D array with the function values at the points.
        """
        raise NotImplementedError()

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

    def evaluate(self, points):
        """Evaluation, see `UncertainFunction.evaluate`."""
        mean = np.empty((len(points), self.num_fun), dtype=np.float)
        error = np.empty_like(mean)
        error[:, self.deterministic] = 0.

        for i, (fun, deterministic) in enumerate(
                zip(self.functions, self.deterministic)):
            prediction = fun.evaluate(points)
            if deterministic:
                mean[:, i] = prediction.squeeze()
            else:
                mean[:, i] = prediction[0].squeeze()
                error[:, i] = prediction[1].squeeze()

        return mean, error

    def gradient(self, points):
        """Gradient, see `UncertainFunction.gradient`."""
        super(FunctionStack, self).gradient(points)


def concatenate_inputs(function):
    """Concatenate the numpy array inputs to the functions."""
    def new_function(self, *args):
        """A function that concatenates inputs."""
        if len(args) == 1:
            return function(self, np.atleast_2d(args[0]))
        else:
            args_2d = map(np.atleast_2d, args)
            return function(self, np.hstack(args_2d))

    return new_function


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

    @concatenate_inputs
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

    @concatenate_inputs
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
    limits: 2d array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d array-like
        The number of points with which to grid each dimension.
    """

    def __init__(self, limits, num_points):
        self.numpoints = num_points
        self.limits = np.asarray(limits, dtype=np.float)
        params = [np.linspace(limit[0], limit[1], n + 1) for limit, n in
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

    def __init__(self, limits, num_points, vertex_values=None):
        """Initialization, see `GridWorld`."""
        super(GridWorld, self).__init__()

        self.limits = np.asarray(limits, dtype=np.float)

        if not (isinstance(num_points, Sequence) or
                isinstance(num_points, np.ndarray)):
            num_points = [num_points] * len(limits)
        self.num_points = np.asarray(num_points, dtype=np.int)

        # Compute offset and unit hyperrectangle
        self.offset = self.limits[:, 0]
        self.unit_maxes = (self.limits[:, 1] - self.offset) / self.num_points
        self.offset_limits = np.stack((np.zeros_like(self.limits[:, 0]),
                                       self.limits[:, 1] - self.offset),
                                      axis=1)

        # Statistics about the grid
        self.nrectangles = np.prod(self.num_points)
        self.nindex = np.prod(self.num_points + 1)
        self.ndim = len(limits)

        self.vertex_values = None
        if vertex_values is not None:
            self.vertex_values = np.asarray(vertex_values)

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
        ijk_index = np.vstack(np.unravel_index(indices, self.num_points + 1)).T
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
        return np.ravel_multi_index(ijk_index.T, self.num_points + 1)

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
        return np.ravel_multi_index(ijk_index.T,
                                    self.num_points)

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
        ijk_index = np.vstack(np.unravel_index(rectangles, self.num_points)).T
        return (ijk_index * self.unit_maxes) + self.offset

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
        ijk_index = np.vstack(np.unravel_index(rectangles, self.num_points)).T
        return np.ravel_multi_index(np.atleast_2d(ijk_index).T,
                                    self.num_points + 1)


class PiecewiseConstant(GridWorld, DeterministicFunction):
    """A piecewise constant function approximator.

    Parameters
    ----------
    limits: 2d array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d array-like
        The number of points with which to grid each dimension.
    vertex_values: 1d array_like, optional
        The values at the vertices of the grid.
    """

    def __init__(self, limits, num_points, vertex_values=None):
        """Initialization, see `PiecewiseConstant`."""
        super(PiecewiseConstant, self).__init__(limits, num_points,
                                                vertex_values=vertex_values)

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
        return self.vertex_values[nodes][:, None]

    def evaluate_constraint(self, points):
        """
        Obtain function values at points from triangulation.

        This function returns a sparse matrix that, when multiplied
        with the vector with all the function values on the vertices,
        returns the function values at points.

        Parameters
        ----------
        points : 2d array
            Each row represents one point

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
        return np.broadcast_to(0, (len(points), self.nindex))


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
    limits: 2d array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d array-like
        The number of points with which to grid each dimension.
    vertex_values: 1d array_like, optional
        The values at the vertices of the grid.
    project: bool, optional
        Whether to project points onto the limits.
    """

    def __init__(self, limits, num_points, vertex_values=None, project=False):
        """Initialization."""
        super(Triangulation, self).__init__(limits, num_points,
                                            vertex_values=vertex_values)

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
        result = np.einsum('ij,ij->i', weights, self.vertex_values[simplices])
        return result[:, None]

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

    def _get_weights_gradient(self, points, index=False):
        """Return the linear gradient weights asscoiated with points.

        Parameters
        ----------
        points : 2d array
            Each row represents one point
        index : bool
            Whether the indices of the triangles are provided instead of the
            points.

        Returns
        -------
        weights : ndarray
            An array that contains the linear weights for each point.
        simplices : ndarray
            The indeces of the simplices associated with each points
        """
        if index:
            simplex_ids = points
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
            The function gradient at the points.
        """
        weights, simplices = self._get_weights_gradient(points)
        # Return function values if desired
        return np.einsum('ijk,ik->ij', weights, self.vertex_values[simplices])

    def gradient_constraint(self, points, index=False):
        """
        Return the gradients at the respective points.

        This function returns a sparse matrix that, when multiplied
        with the vector of all the function values on the vertices,
        returns the gradients. Note that after the product you have to call
        ```np.reshape(grad, (ndim, -1))``` in order to obtain a proper
        gradient matrix.

        Parameters
        ----------
        points : 2d array
            Each row contains one state at which to evaluate the gradient.
        index : bool
            Whether the simplex indeces are provided instead of points.

        Returns
        -------
        gradient : scipy.sparse.coo_matrix
            A sparse matrix so that
            `grad(points) = B.dot(V(vertices)).reshape(ndim, -1)` corresponds
            to the true gradients
        """
        weights, simplices = self._get_weights_gradient(points, index=index)

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

    def evaluate(self, points):
        """See `DeterministicFunction.evaluate`."""
        points = np.asarray(points)
        return np.sum(points.dot(self.matrix) * points, axis=1)[:, None]

    def gradient(self, points):
        """See `DeterministicFunction.gradient`."""
        points = np.asarray(points)
        return 2 * points.dot(self.matrix)
