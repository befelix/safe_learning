"""Unit tests for treinforcement learning."""

from __future__ import division, print_function, absolute_import

from numpy.testing import *
import unittest
import sys
import numpy as np
from safe_learning.utilities import dlqr

from safe_learning import PolicyIteration, Triangulation

if sys.version_info.major <= 2:
    import mock
else:
    from unittest import mock


class PolicyIterationTest(unittest.TestCase):
    """Test the policy iteration."""

    def integration_test(self):
        """Test the values."""
        a = np.array([[1.2]])
        b = np.array([[0.9]])
        q = np.array([[1]])
        r = np.array([[0.1]])

        def dynamics(x, u):
            return x.dot(a.T) + u.dot(b.T)

        k, p = dlqr(a, b, q, r)

        value_function = Triangulation([[-1, 1]], 50, project=True)

        u_max = np.sum(np.abs(k)) * 1.1

        def reward_function(x, a, _):
            return -np.sum(x ** 2 * np.diag(q), axis=1) - np.sum(
                a ** 2 * np.diag(r), axis=1)

        rl = PolicyIteration(value_function.all_points,
                             np.linspace(-u_max, u_max, 10)[:, None],
                             dynamics,
                             reward_function,
                             value_function)

        for i in range(5):
            rl.update_value_function()
            rl.update_policy()

        optimal_policy = -rl.state_space.dot(k.T)

        max_error = np.max(np.abs(rl.policy - optimal_policy))
        disc_error = np.max(np.diff(rl.action_space[:, 0]))

        assert (max_error < disc_error)


if __name__ == '__main__':
    unittest.main()
