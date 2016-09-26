from __future__ import absolute_import

from .triangulization import *

# Import test after __all__ (no documentation)
from numpy.testing import Tester
test = Tester().test

# Add everything to __all__
__all__ = [s for s in dir() if not s.startswith('_')]
