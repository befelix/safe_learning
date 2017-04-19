"""Import relevant packages and define testing environment."""

from __future__ import absolute_import

# Add the configuration settings
from .configuration import Configuration
config = Configuration()
del Configuration

from .functions import *
from .lyapunov import *
from .reinforcement_learning import *
from . import utilities

try:
    from pytest import main as run_tests
except ImportError:
    def run_tests():
        """Run the test package."""
        raise ImportError('Testing requires the pytest package.')
