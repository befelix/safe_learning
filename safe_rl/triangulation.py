from __future__ import absolute_import, print_function, division


import numpy as np
from scipy import spatial, sparse


__all__ = ['Triangulation']


class Triangulation(spatial.Delaunay):
    """
    Generalization of Delaunay triangulization with additional properties.

    A normal Delaunay triangulation, but provides additional methods to obtain
    the hyperplanes and gradients.

    Parameters
    ----------
    see scipy.spatial.Delaunay
    """

    def __init__(self, points):
        super(Triangulation, self).__init__(points)
        self.parameters = None
        self._update_equations()

    def _update_equations(self):
        """Compute the simplex equations for a given triangulation"""
        # TODO: It's enough to do this for one hypercube, rest are repetitions
        self.parameters = np.empty((self.nsimplex, self.ndim ** 2),
                                   dtype=np.float)

        for i, simplex in enumerate(self.simplices):
            simplex_points = self.points[simplex]
            self.parameters[i, :] = np.linalg.inv(simplex_points[1:] -
                                                  simplex_points[:1]).ravel()

    def function_values_at(self, points):
        """
        Obtain function values at points from triangulation.

        Get a matrix that, when multiplied with the vector of function values
        on the vertices of the simplex, returns the function values at points.

        Parameters
        ----------
        points: 2d array
            Each row represents one point

        Returns
        -------
        B: scipy.sparse
            A sparse matrix so that V(points) = B * V(vertices)
        """
        simplex_ids = self.find_simplex(points)

        num_constraints = len(points) * 3
        X = np.empty(num_constraints, dtype=np.float)
        I = np.empty(num_constraints, dtype=np.int32)
        J = np.empty(num_constraints, dtype=np.int32)

        for i, (point, simplex_id) in enumerate(zip(points, simplex_ids)):
            # TODO: Add check for when point it is outside the triangulization

            # Ids for the corner points
            simplex = self.simplices[simplex_id]
            # Id of the origin points
            origin = self.points[simplex[0]]

            # pre-multiply tmp with the distance
            tmp = self.parameters[simplex_id]
            tmp = tmp.reshape(self.ndim, self.ndim).T.dot(
                point - origin)

            index = slice(3 * i, 3 * (i + 1))
            X[index] = [1 - np.sum(tmp), tmp[0], tmp[1]]
            I[index] = i
            J[index] = simplex

        return sparse.coo_matrix((X, (I, J)),
                                 shape=(len(points), self.npoints)).tocsr()

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
        simplex_ids = self.find_simplex(points)

        num_constraints = len(points) * 3
        X = np.empty(3 * num_constraints, dtype=np.float)
        I = np.empty(3 * num_constraints, dtype=np.int32)
        J = np.empty(3 * num_constraints, dtype=np.int32)

        for i, simplex_id in enumerate(simplex_ids):
            # TODO: Add check for when point it is outside the triangulization

            # Ids for the corner points
            simplex = self.simplices[simplex_id]
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