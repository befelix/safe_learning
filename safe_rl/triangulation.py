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

    def function_values_at(self, points):
        """
        Obtain function values at points from triangulation.

        Get a matrix that, when multiplied with the vector of function values
        on the vertices of the simplex, returns the function values at points.

        Parameters
        ----------
        points: 2d array
            Each row represents one points

        Returns
        -------
        B: scipy.sparse
            A sparse matrix so that V(points) = B * V(vertices)
        """
        simplex_ids = self.find_simplex(points)
        simplices = self.simplices[simplex_ids]

        num_constraints = len(points) * 3
        X = np.empty(num_constraints, dtype=np.float)
        I = np.empty(num_constraints, dtype=np.int32)
        J = np.empty(num_constraints, dtype=np.int32)

        for i, (point, simplex) in enumerate(zip(points, simplices)):
            simplex_points = points[simplex]
            # TODO: Add check for when it is outside the triangulization

            tmp = np.linalg.solve((simplex_points[1:] - simplex_points[:1]).T,
                                  point - simplex_points[0])

            index = slice(3 * i, 3 * (i + 1))
            X[index] = [1 - np.sum(tmp), tmp[0], tmp[1]]
            I[index] = i
            J[index] = simplex

        return sparse.coo_matrix((X, (I, J)),
                                 shape=(len(points), self.npoints)).tocsr()