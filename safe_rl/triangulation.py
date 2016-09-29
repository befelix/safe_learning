from __future__ import absolute_import, print_function, division


import numpy as np
from scipy import spatial, sparse
from sklearn.utils.extmath import cartesian


__all__ = ['Triangulation', 'Delaunay']


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
        self.limits = limits
        self.numpoints = num_points
        params = [np.linspace(limit[0], limit[1], n + 1) for limit, n in
                  zip(limits, num_points)]
        output = np.meshgrid(*params)
        points = np.array([par.ravel() for par in output]).T
        super(ScipyDelaunay, self).__init__(points)


class Delaunay(object):
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
        super(Delaunay, self).__init__()

        self.limits = np.asarray(limits)
        self.num_points = np.asarray(num_points, dtype=np.int)

        # Compute offset and unit hyperrectangle
        self.offset = self.limits[:, 0]
        self.unit_maxes = (self.limits[:, 1] - self.offset) / self.num_points

        # Get triangulation
        hyperrectangle_corners = cartesian(np.diag(self.unit_maxes))
        self.triangulation = spatial.Delaunay(hyperrectangle_corners)
        self.unit_simplices = self._triangulation_simplex_indices()

        # Some statistics about the triangulation
        self.nrectangles = np.prod(self.num_points)
        self.ndim = self.triangulation.ndim
        self.nsimplex = self.triangulation.nsimplex * self.nrectangles
        self.nindex = np.prod(self.num_points + 1)

        # Parameters for the hyperplanes of the triangulation
        self.hyperplanes = None
        self._update_hyperplanes()

    def _triangulation_simplex_indices(self):
        """Return the simplex indices in our coordinates.

        Returns
        -------
        simplices: ndarray (int)
            The simplices array in our extended coordinate system.
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

        for i, simplex in enumerate(self.unit_simplices):
            simplex_points = self.index_to_state(simplex)
            self.hyperplanes[i] = np.linalg.inv(simplex_points[1:] -
                                                simplex_points[:1])

    def index_to_state(self, indices):
        """Convert indices to physical states.

        Parameters
        ----------
        indices: ndarray (int)
            The indices of points on the discretization.

        Returns
        -------
        states: ndarray
            The states with physical units that correspond to the indices.
        """
        indices = np.atleast_1d(indices)
        ijk_index = np.vstack(np.unravel_index(indices, self.num_points + 1)).T
        return (ijk_index * self.unit_maxes) + self.offset

    def state_to_index(self, states):
        """Convert physical states to indices

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
        states = (states - self.offset) / self.unit_maxes
        ijk_index = np.rint(states).astype(np.int)
        return np.ravel_multi_index(ijk_index.T, self.num_points + 1)

    def state_to_rectangle(self, states):
        """Convert physical states to its closest rectangle index.

        Parameters
        ----------
        states: ndarray
            Physical states on the discretization.

        Returns
        -------
        rectangles: ndarray (int)
            The indices that correspond to rectangles of the physical states.
        """
        states = np.atleast_2d(states)
        eps = np.finfo(states.dtype).eps

        ijk_index = np.floor_divide(states - self.offset + 2 * eps,
                                    self.unit_maxes).astype(np.int)
        # print(ijk_index)
        return np.ravel_multi_index(np.atleast_2d(ijk_index).T,
                                    self.num_points)

    def rectangle_to_state(self, rectangles):
        """
        Convert rectangle indices to the states of the bottem-left corners.

        Parameters
        ----------
        rectangles: ndarray (int)
            The indices of the rectangles

        Returns
        -------
        states: ndarray
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
        corners: ndarray (int)
            The indices of the bottom-left corners of the rectangles.
        """
        ijk_index = np.vstack(np.unravel_index(rectangles, self.num_points)).T
        return np.ravel_multi_index(np.atleast_2d(ijk_index).T,
                                    self.num_points + 1)

    def find_simplex(self, points):
        """Find the simplices corresponding to points

        Parameters
        ----------
        points: 2darray

        Returns
        -------
        simplices: np.array (int)
            The indices of the simplices
        """
        points = np.atleast_2d(points)

        # Convert to basic hyperrectangle coordinates and find simplex
        unit_coordinates = (points - self.offset) % self.unit_maxes
        simplex_ids = self.triangulation.find_simplex(unit_coordinates)

        # Adjust for the hyperrectangle index
        rectangles = self.state_to_rectangle(points)
        simplex_ids += rectangles * self.triangulation.nsimplex

        return simplex_ids

    def simplices(self, indices):
        """Return the simplices corresponding to the simplex index.

        Parameters
        ----------
        indices: ndarray
            The indices of the simpleces

        Returns
        -------
        simplices: ndarray
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


class Triangulation(object):
    """
    Generalization of Delaunay triangulization with additional properties.

    A normal Delaunay triangulation, but provides additional methods to obtain
    the hyperplanes and gradients.

    Parameters
    ----------
    see scipy.spatial.Delaunay
    """

    def __init__(self, limits, num_points):
        super(Triangulation, self).__init__()
        self.delaunay = Delaunay(limits, num_points)

    def function_values_at(self, points, values=None):
        """
        Obtain function values at points from triangulation.

        Get a matrix that, when multiplied with the vector of function values
        on the vertices of the simplex, returns the function values at points.

        Parameters
        ----------
        points: 2d array
            Each row represents one point
        values: 1d array, optional
            The values for all the corners of the simplex

        Returns
        -------
        B: scipy.sparse.coo_matrix
            A sparse matrix so that V(points) = B * V(vertices)
        """
        simplex_ids = self.delaunay.find_simplex(points)
        simplices = self.delaunay.simplices(simplex_ids)
        origins = self.delaunay.index_to_state(simplices[:, 0])

        # Get hyperplane equations
        simplex_ids %= self.delaunay.triangulation.nsimplex
        hyperplanes = self.delaunay.hyperplanes[simplex_ids]

        hyp_weights = np.einsum('ij,ijk->ik', points - origins, hyperplanes)

        nsimp = self.delaunay.ndim + 1
        nindex = self.delaunay.nindex
        npoints = len(points)

        # The weights have to add up to one
        weights = np.empty((npoints, nsimp), dtype=np.float)
        weights[:, 0] = 1 - np.sum(hyp_weights, axis=1)
        weights[:, 1:] = hyp_weights

        if values is not None:
            return np.sum(weights * values[simplices], axis=1)

        # Indices of constraints (nsimp points per simplex, so we have nsimp
        #  values in each row; one for each simplex)
        rows = np.repeat(np.arange(len(points)), nsimp)
        cols = simplices.ravel()

        return sparse.coo_matrix((weights.ravel(), (rows, cols)),
                                 shape=(npoints, nindex))

    def gradient_at(self, points):
        """
        Compute the gradients at the respective points

        Parameters
        ----------
        points: 2d array
            Each row represents one point

        Returns
        -------
        B: scipy.sparse
            A sparse matrix so that gradient(points) = B * V(vertices)
        """
        raise NotImplementedError('Work in progress')
        simplex_ids = self.delaunay.find_simplex(points)

        num_constraints = len(points) * 3
        X = np.empty(3 * num_constraints, dtype=np.float)
        I = np.empty(3 * num_constraints, dtype=np.int32)
        J = np.empty(3 * num_constraints, dtype=np.int32)

        for i, simplex_id in enumerate(simplex_ids):
            # TODO: Add check for when point it is outside the triangulization

            # Ids for the corner points
            simplex = self.simplices(simplex_id)
            # Id of the origin points
            origin = self.points[simplex[0]]

            # pre-multiply tmp with the distance
            tmp = self.parameters[simplex_id].reshape(self.ndim, self.ndim)

            index = slice(3 * i, 3 * (i + 1))
            X[index] = [1 - np.sum(tmp), tmp[0], tmp[1]]
            I[index] = i
            J[index] = simplex

        # TODO: How do we handle that we get multiple derivatives here? 
        return sparse.coo_matrix((X, (I, J)),
                                 shape=(len(points), self.npoints)).tocsr()