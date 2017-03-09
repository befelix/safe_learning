"""An efficient implementation of Delaunay triangulation on regular grids."""

from __future__ import absolute_import, print_function, division

import numpy as np

__all__ = ['DeterministicFunction', 'Triangulation', 'PiecewiseConstant',
           'GridWorld', 'UncertainFunction', 'FunctionStack',
           'QuadraticFunction', 'GaussianProcess', 'GPyGaussianProcess',
           'GPR_cached', 'GPflowGaussianProcess', 'sample_gp_function']

try:
    import GPflow
    import tensorflow as tf
    from GPflow.param import AutoFlow
    from GPflow.tf_wraps import eye
except ImportError:
    GPflow = None
    tf = None

from scipy import spatial, sparse, interpolate, linalg
from sklearn.utils.extmath import cartesian

from .utilities import linearly_spaced_combinations

_EPS = np.finfo(np.float).eps


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

        self.parameters = None

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

    def parameter_derivative(self, *points):
        """Return the derivative with respect to the parameter vector.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        gradient : ndarray
            The function gradient with respect to the parameters at the points.
        """
        if self.parameters is None:
            return None

        raise NotImplementedError('The derivatives towards the parameters is'
                                  'not implemented.')

    def gradient_parameter_derivative(self, *points):
        """Return the derivative of the gradient with respect to parameters.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        gradient : ndarray
            The function gradient with respect to the parameters at the points.
        """
        if self.parameters is None:
            return None

        raise NotImplementedError('The derivatives towards the parameters is'
                                  'not implemented.')


def concatenate_inputs(start=0):
    """Concatenate the numpy array inputs to the functions.

    Parameters
    ----------
    start : int, optional
        The attribute number at which to start concatenating.
    """
    def wrap(function):
        def wrapped_function(*args, **kwargs):
            """A function that concatenates inputs."""
            to_concatenate = list(map(np.atleast_2d, args[start:]))

            if len(to_concatenate) == 1:
                concatenated = to_concatenate
            else:
                concatenated = [np.hstack(to_concatenate)]

            args = list(args[:start]) + concatenated
            return function(*args, **kwargs)

        return wrapped_function

    return wrap


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

    @concatenate_inputs(start=1)
    def evaluate(self, points):
        """Evaluation, see `UncertainFunction.evaluate`."""
        mean = np.empty((len(points), self.num_fun), dtype=np.float)
        if np.all(self.deterministic):
            error = np.broadcast_to(0, (len(points), self.num_fun))
        else:
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

    def gradient(self, *points):
        """Gradient, see `UncertainFunction.gradient`."""
        for fun in self.functions:
            yield fun.gradient(*points)


def concatenate_inputs(start=0):
    """Concatenate the numpy array inputs to the functions.

    Parameters
    ----------
    start : int, optional
        The attribute number at which to start concatenating.
    """
    def wrap(function):
        def wrapped_function(*args, **kwargs):
            """A function that concatenates inputs."""
            to_concatenate = list(map(np.atleast_2d, args[start:]))

            if len(to_concatenate) == 1:
                concatenated = to_concatenate
            else:
                concatenated = [np.hstack(to_concatenate)]

            args = list(args[:start]) + concatenated
            return function(*args, **kwargs)

        return wrapped_function

    return wrap


class GaussianProcess(UncertainFunction):
    """An 'UncertainFunction' for GPy or GPflow gaussian process.

    Parameters
    ----------
    gaussian_process : instance of GPy.core.GP or GPflow.gpr.GPR
        The Gaussian process model.
    beta : float
        The scaling factor for the standard deviation to create confidence
        intervals.
    """

    def __init__(self, gaussian_process, beta=2):
        """Initialize GuassianProcess with either GPflow or GPy gp."""
        super(GaussianProcess, self).__init__()
        self.n_dim = gaussian_process.X.shape[-1]
        self.gaussian_process = gaussian_process

        if callable(beta):
            self.beta = beta
        else:
            self.beta = lambda t: beta

    @property
    def X(self):
        """Input location of observed data. One observation per row."""
        raise NotImplementedError

    @property
    def Y(self):
        """Observed output. One observation per row."""
        raise NotImplementedError

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
        raise NotImplementedError


class GPyGaussianProcess(GaussianProcess):
    """A `GaussianProcess` for GPy Gaussian processes.

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

    def __init__(self, gaussian_process, beta=2):
        """Initialization, see `FakeGP`."""
        super(GPyGaussianProcess, self).__init__(gaussian_process, beta)

    @property
    def X(self):
        """Input location of observed data. One observation per row."""
        return self.gaussian_process.X

    @property
    def Y(self):
        """Observed output. One observation per row."""
        return self.gaussian_process.Y

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
        _, var = self.gaussian_process.predict_noiseless(points)
        mean_dx, var_dx = self.gaussian_process.predictive_gradients(points)
        mean_dx = mean_dx.squeeze(-1)

        var[var <= 1e-10] = 1e-10
        std_dx = (0.5 / np.sqrt(var)) * var_dx
        t = len(self.gaussian_process.X)
        return mean_dx, self.beta(t) * std_dx

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
        x_new = np.vstack((self.X, x))
        y_new = np.vstack((self.Y, y))
        self.gaussian_process.set_XY(x_new, y_new)


class GPR_cached(GPflow.gpr.GPR):
    """GPflow.gpr.GPR class that stores cholesky decomposition.

    Parameters
    ----------
    x : ndarray
        A 2d array with states to initialize the GP model. Each state is on
        a row.
    y : ndarray
        A 2d array with measurements to initialize the GP model. Each
        measurement is on a row.

    """

    def __init__(self, x, y, kern):
        """Initialize GP and cholesky decomposition."""
        if GPflow is None:
            raise ImportError('This function requires the GPflow module.')
        # super(GPR_cached, self).__init__(self, x, y, kern)
        GPflow.gpr.GPR.__init__(self, x, y, kern)
        self.L, self.V = self.update_cholesky()

    @AutoFlow()
    def update_cholesky(self):
        """Return the cholesky decomposition for the observed points."""
        kernel = (self.kern.K(self.X)
                  + eye(tf.shape(self.X)[0]) * self.likelihood.variance)
        cholesky = tf.cholesky(kernel)

        target = self.Y - self.mean_function(self.X)
        alpha = tf.matrix_triangular_solve(cholesky, target)
        return cholesky, alpha

    def build_predict(self, Xnew, full_cov=False):
        """Predict mean and variance of the GP at locations in Xnew.

        Parameters
        ----------
        Xnew : ndarray
            The points at which to evaluate the function. One row for each
            data points.
        full_cov : bool
            if False retutns only the diagonal of the covariance matrix

        Returns
        -------
        mean : ndarray
            The expected function values at the points.
        error_bounds : ndarray
            Diagonal of the covariance matrix (or full matrix).

        """
        Kx = self.kern.K(self.X, Xnew)
        A = tf.matrix_triangular_solve(self.L, Kx, lower=True)
        fmean = (tf.matmul(tf.transpose(A), self.V)
                 + self.mean_function(Xnew))
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(tf.transpose(A), A)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)),
                           [1, tf.shape(self.Y)[1]])
        return fmean, fvar

    @AutoFlow((tf.float64, [None, None]))
    def GP_flow_predictive_gradients(self, Xnew):
        """Compute the gradient of mean and variance of the posterior in tf.

        TO DO: Use graph from build predict and add gradients on top
        """
        Kx = self.kern.K(self.X, Xnew)
        A = tf.matrix_triangular_solve(self.L, Kx, lower=True)
        fmean = (tf.matmul(tf.transpose(A), self.V)
                 + self.mean_function(Xnew))

        fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
        fvar = tf.reshape(fvar, (-1, 1))
        fvar = tf.tile(fvar, [1, tf.shape(self.Y)[1]])
        return tf.gradients(fmean, Xnew), tf.gradients(fvar, Xnew)

    def predictive_gradients(self, Xnew):
        """Wrap for same output format as GPy_GaussianProcess.

        TO DO: make this function a decorator for previous function
        """
        m_x, v_x = self.GP_flow_predictive_gradients(Xnew)
        m_x = np.expand_dims(np.array(m_x).squeeze(), axis=-1)
        v_x = np.array(v_x).squeeze()
        return m_x, v_x


class GPflowGaussianProcess(GaussianProcess):
    """A `GaussianProcess` for GPflow Gaussian processes.

    Parameters
    ----------
    gaussian_process : instance of GPy.core.GP
        The Gaussian process model.
    beta : float
        The scaling factor for the standard deviation to create
        confidence intervals.

    Notes
    -----
    The evaluate and gradient functions can be called with multiple
    arguments, in which case they are concatenated before being
    passed to the GP.
    """

    def __init__(self, gaussian_process, beta=2.):
        """Initialization."""
        if GPflow is None:
            raise ImportError('This function requires the GPflow module.')
        super(GPflowGaussianProcess, self).__init__(gaussian_process, beta)

    @property
    def X(self):
        """Input location of observed data. One observation per row."""
        return self.gaussian_process.X.value

    @property
    def Y(self):
        """Observed output. One observation per row."""
        return self.gaussian_process.Y.value

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
        mean, var = self.gaussian_process.predict_f(points)

        t = self.gaussian_process.X.shape[0]
        return mean, self.beta(t) * np.sqrt(var)

    @concatenate_inputs(start=1)
    def gradient(self, points):
        """Return the distribution over the gradient.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row
            for each data point.

        Returns
        -------
        mean : ndarray
            The expected function gradient at the points.
        var : ndarray
            Error bounds for each dimension of the estimate.
        """
        _, var = self.gaussian_process.predict_f(points)
        mean_dx, var_dx = self.gaussian_process.predictive_gradients(
            points)
        mean_dx = np.array(mean_dx).squeeze(-1)
        var = np.array(var)

        var[var <= 1e-10] = 1e-10
        std_dx = (0.5 / np.sqrt(var)) * var_dx
        t = self.gaussian_process.X.shape[0]
        return mean_dx, self.beta(t) * std_dx

    def add_data_point(self, x, y):
        """Add data points to the GP model and update cholesky.

        Parameters
        ----------
        x : ndarray
            A 2d array with the new states to add to the GP model. Each new
            state is on a new row.
        y : ndarray
            A 2d array with the new measurements to add to the GP model.
            Each measurements is on a new row.
        """
        self.gaussian_process.X = np.vstack((self.X, np.atleast_2d(x)))
        self.gaussian_process.Y = np.vstack((self.Y, np.atleast_2d(y)))
        (self.gaussian_process.L,
         self.gaussian_process.V) = self.gaussian_process.update_cholesky()


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

        self.limits = np.atleast_2d(limits)
        num_points = np.broadcast_to(num_points, len(self.limits))
        self.num_points = num_points.astype(np.int, copy=False)

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
        self.ndim = len(self.limits)

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
        states = np.atleast_2d(states).astype(np.float) - self.offset[None, :]
        if clip:
            np.clip(states,
                    self.offset_limits[:, 0] + 2 * _EPS,
                    self.offset_limits[:, 1] - 2 * _EPS,
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

        self._parameters = None
        self.parameters = vertex_values

    @property
    def parameters(self):
        """Return the vertex values."""
        return self._parameters

    @parameters.setter
    def parameters(self, values):
        """Set the vertex values."""
        if values is None:
            self._parameters = values
        else:
            self._parameters = np.asarray(values).reshape(self.nindex, -1)

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
        return self.parameters[nodes]

    def parameter_derivative(self, points):
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
            A sparse matrix B so that evaluate(points) = B.dot(parameters).
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

        self._parameters = None
        self.parameters = vertex_values

        # Get triangulation
        if len(self.limits) == 1:
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
    def parameters(self):
        """Return the vertex values."""
        return self._parameters

    @parameters.setter
    def parameters(self, values):
        """Set the vertex values."""
        if values is None:
            self._parameters = values
        else:
            values = np.asarray(values).reshape(self.nindex, -1)
            self._parameters = values

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
        points = np.atleast_2d(points)
        weights, simplices = self._get_weights(points)

        # Return function values if desired
        result = np.einsum('ij,ijk->ik',
                           weights,
                           self.parameters[simplices])
        return result

    def parameter_derivative(self, points):
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
            A sparse matrix B so that evaluate(points) = B.dot(parameters).
        """
        points = np.atleast_2d(points)
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
            simplex_ids = np.atleast_1d(indices)
        elif indices is None:
            simplex_ids = self.find_simplex(points)
        else:
            raise TypeError('Need to provide at least one input argument.')
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
        points = np.atleast_2d(points)
        weights, simplices = self._get_weights_gradient(points)
        # Return function values if desired
        res = np.einsum('ijk,ikl->ilj', weights, self.parameters[simplices, :])
        if res.shape[1] == 1:
            res = res.squeeze(axis=1)
        return res

    def gradient_parameter_derivative(self, points=None, indices=None):
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
                       interpolation='kernel', mean_function=None):
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
        Mean of the sample function. Note that if you are trying to pass a GPy
        mapping then you need to pass `mapping.f`.

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

        @concatenate_inputs()
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

        @concatenate_inputs()
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
