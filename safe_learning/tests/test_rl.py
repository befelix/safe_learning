"""Unit tests for treinforcement learning."""

from __future__ import division, print_function, absolute_import

from numpy.testing import *
import unittest
import sys
import numpy as np
from safe_learning.utilities import dlqr

from safe_learning import PolicyIteration, Triangulation, DeterministicFunction

if sys.version_info.major <= 2:
    import mock
else:
    from unittest import mock

try:
    import cvxpy
except ImportError:
    cvxpy = None


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
        value_function = Triangulation([[-1, 1]], 19, project=True)
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

        assert(max_error < disc_error)
        assert_allclose(rl.values, value_function.vertex_values[:, 0])

    @unittest.skipIf(cvxpy is None, 'Skipping cvxpy tests.')
    def test_optimization(self):
        """Test the value function optimization."""
        pass

    # @mock.patch('safe_learning.reinforcement_learning.PolicyIteration')
    def test_future_values(self):
        """Test future values."""
        dynamics = mock.Mock()
        dynamics.return_value = 'states'

        rewards = mock.Mock()
        rewards.return_value = np.arange(4, dtype=np.float)

        value_function = mock.Mock()
        value_function.return_value = np.arange(4, dtype=np.float)

        states = np.arange(4)[:, None]
        actions = np.arange(2)[:, None]
        rl = PolicyIteration(states,
                             actions,
                             dynamics,
                             rewards,
                             value_function)

        future_values = rl.get_future_values(rl.policy)
        true_values = np.arange(4, dtype=np.float) * (1 + rl.gamma)

        dynamics.assert_called_with(rl.state_space, rl.policy)
        rewards.assert_called_with(rl.state_space, rl.policy, 'states')

        assert_allclose(future_values, true_values)

        # rl.terminal_states = np.array([0, 0, 0, 1], dtype=np.bool)
        # future_values = rl.get_future_values(rl.policy)
        # true_values[rl.terminal_states] = rewards()[rl.terminal_states]
        #
        # assert_allclose(future_values, true_values)


if __name__ == '__main__':
    unittest.main()
