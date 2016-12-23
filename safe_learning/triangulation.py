"""An efficient implementation of Delaunay triangulation on regular grids."""

from __future__ import absolute_import, print_function, division

from collections import Sequence

import numpy as np
from scipy import spatial, sparse
from sklearn.utils.extmath import cartesian


__all__ = ['Delaunay', 'PiecewiseConstant', 'GridWorld', 'PiecewiseConstant']


class FunctionApproximator(object):
    """Base class for function approximators.

    Parameters
    ----------
    limits: 2d array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    """

    def __init__(self, limits):
        super(FunctionApproximator, self).__init__()
        self.limits = np.asarray(limits, dtype=np.float)
        self.ndim = None

    def values_at(self, points, vertex_values=None, project=False):
        """Return the function values."""
        raise NotImplementedError()

    def gradient_at(self, simplex_ids, vertex_values=None):
        """Return the gradient."""
        raise NotImplementedError()


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
        params = [np.linspace(limit[0], limit[1], n + 1) for limit, n in
                  zip(limits, num_points)]
        output = np.meshgrid(*params)
        points = np.array([par.ravel() for par in output]).T
        super(ScipyDelaunay, self).__init__(points)


class GridWorld(FunctionApproximator):
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
        super(GridWorld, self).__init__(limits)

        if not isinstance(num_points, Sequence):
            num_points = [num_points] * len(limits)
        self.num_points = np.atleast_1d(np.asarray(num_points, dtype=np.int))

        # Compute offset and unit hyperrectangle
        self.offset = self.limits[:, 0]
        self.unit_maxes = (self.limits[:, 1] - self.offset) / self.num_points
        self.offset_limits = np.stack((np.zeros_like(self.limits[:, 0]),
                                       self.limits[:, 1] - self.offset),
                                      axis=1)

        # Statistics about the grid
        self.nrectangles = np.prod(self.num_points)
        self.ndim = len(limits)
        self.nindex = np.prod(self.num_points + 1)

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
        # clip to domain (find closest rectangle)
        if offset:
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


class PiecewiseConstant(GridWorld):
    """A piecewise constant function approximator.

    Parameters
    ----------
    limits: 2d array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d array-like
        The number of points with which to grid each dimension.
    """

    def __init__(self, limits, num_points):
        """Initialization, see `PiecewiseConstant`."""
        super(PiecewiseConstant, self).__init__(limits, num_points)

    def values_at(self, points, vertex_values=None, project=False):
        """
        Obtain function values at points from triangulation.

        If the values on the vertices are provided, the function returns the
        values at the given points.
        Otherwise, this function returns a sparse matrix that, when multiplied
        with the vector with all the function values on the vertices,
        returns the function values at points.

        Parameters
        ----------
        points : 2d array
            Each row represents one point
        vertex_values : 1d array, optional
            The values for all the corners of the simplex
        project : bool, optional
            This parameter has no effect, since all inputs are projected.

        Returns
        -------
        values
            Either a vector of function values or a sparse matrix so that
            V(points) = B.dot(V(vertices))
        """
        nodes = self.state_to_index(points)

        if vertex_values is not None:
            return vertex_values[nodes]

        npoints = len(points)
        weights = np.ones(npoints, dtype=np.int)
        rows = np.arange(npoints)
        cols = nodes
        return sparse.coo_matrix((weights, (rows, cols)),
                                 shape=(npoints, self.nindex))

    def gradient_at(self, simplex_ids, vertex_values=None):
        """Return gradient (always zero)."""
        return np.zeros((len(simplex_ids), self.nindex))


class Delaunay(GridWorld):
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
    """

    def __init__(self, limits, num_points):
        """Initialization."""
        super(Delaunay, self).__init__(limits, num_points)

        # Get triangulation
        hyperrectangle_corners = cartesian(np.diag(self.unit_maxes))
        self.triangulation = spatial.Delaunay(hyperrectangle_corners)
        self.unit_simplices = self._triangulation_simplex_indices()

        # Some statistics about the triangulation
        self.nsimplex = self.triangulation.nsimplex * self.nrectangles

        # Parameters for the hyperplanes of the triangulation
        self.hyperplanes = None
        self._update_hyperplanes()

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

    def values_at(self, points, vertex_values=None, project=False):
        """
        Obtain function values at points from triangulation.

        If the values on the vertices are provided, the function returns the
        values at the given points.
        Otherwise, this function returns a sparse matrix that, when multiplied
        with the vector with all the function values on the vertices,
        returns the function values at points.

        Parameters
        ----------
        points : 2d array
            Each row represents one point
        vertex_values : 1d array, optional
            The values for all the corners of the simplex
        project : bool, optional
            Wether to project data points back onto the triangulation if they
            are defined outside the limits. This can increase robustness for
            iterative algorithms.

        Returns
        -------
        values
            Either a vector of function values or a sparse matrix so that
            V(points) = B.dot(V(vertices))
        """
        simplex_ids = self.find_simplex(points)

        if project:
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

        # Return function values if desired
        if vertex_values is not None:
            return np.einsum('ij,ij->i', weights, vertex_values[simplices])

        # Construct sparse matrix for optimization

        # Indices of constraints (nsimp points per simplex, so we have nsimp
        #  values in each row; one for each simplex)
        rows = np.repeat(np.arange(len(points)), nsimp)
        cols = simplices.ravel()

        return sparse.coo_matrix((weights.ravel(), (rows, cols)),
                                 shape=(npoints, self.nindex))

    def gradient_at(self, simplex_ids, vertex_values=None):
        """
        Return the gradients at the respective points.

        If the values on the vertices are provided, the function returns the
        gradients at the given points.
        Otherwise, this function returns a sparse matrix that, when multiplied
        with the vector of all the function values on the vertices,
        returns the gradients. Note that after the product you have to call
        ```np.reshape(grad, (ndim, -1))``` in order to obtain a proper
        gradient matrix.

        Parameters
        ----------
        simplex_ids : 1d array
            Each value represents the id of a simplex
        vertex_values : 1d array, optional
            The values for all the corners of the simplex

        Returns
        -------
        gradients
            Either a vector of gradient values or a sparse matrix so that
            grad(points) = B.dot(V(vertices)).reshape(ndim, -1) corresponds
            to the true gradients
        """
        simplex_ids = np.asarray(simplex_ids, dtype=np.int)
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

        # Return function values if desired
        if vertex_values is not None:
            return np.einsum('ijk,ik->ij', weights, vertex_values[simplices])

        # Construct sparse matrix for optimization

        # Indices of constraints (ndim gradients for each point, which each
        # depend on the nsimp vertices of the simplex.
        rows = np.repeat(np.arange(npoints * self.ndim), nsimp)
        cols = np.tile(simplices, (1, self.ndim)).ravel()

        return sparse.coo_matrix((weights.ravel(), (rows, cols)),
                                 shape=(self.ndim * npoints, self.nindex))
