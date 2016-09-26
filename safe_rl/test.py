from __future__ import division, print_function, absolute_import

from numpy.testing import *
import numpy as np

from .triangulation import Triangulation


class TriangulationTest(TestCase):
    """Test the Triangulization method"""

    def test_initilization(self):

        points = np.random.randn((10, 2))
        Triangulation(points)
