"""Unit tests for treinforcement learning."""

from __future__ import division, print_function, absolute_import

from numpy.testing import assert_allclose
import unittest
import sys
import numpy as np
from safe_learning.utilities import dlqr

from safe_learning import PolicyIteration, _Triangulation, DeterministicFunction

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
        value_function = _Triangulation([[-1, 1]], 19, project=True)
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
        assert_allclose(rl.values, value_function.parameters[:, 0])

    @unittest.skipIf(cvxpy is None, 'Skipping cvxpy tests.')
    def test_optimization(self):
        """Test the value function optimization."""
        dynamics = mock.Mock()
        dynamics.return_value = 'states'

        rewards = mock.Mock()
        rewards.return_value = np.arange(4, dtype=np.float)

        # transition probabilities
        trans_probs = np.array([[0, .5, .5, 0],
                                [.2, .1, .3, .5],
                                [.3, .2, .4, .1],
                                [0, 0, 0, 1]],
                               dtype=np.float)

        value_function = mock.create_autospec(DeterministicFunction)
        value_function.parameter_derivative.return_value = trans_probs
        value_function.nindex = 4
        value_function.parameters = np.zeros((4, 1))

        states = np.arange(4)[:, None]
        actions = np.arange(2)[:, None]
        rl = PolicyIteration(states,
                             actions,
                             dynamics,
                             rewards,
                             value_function)

        true_values = np.linalg.solve(np.eye(4) - rl.gamma * trans_probs,
                                      rewards.return_value)

        rl.optimize_value_function()

        dynamics.assert_called_with(rl.state_space, rl.policy)
        rewards.assert_called_with(rl.state_space, rl.policy, 'states')

        assert_allclose(rl.values, true_values)

        rl.terminal_states = np.array([0, 0, 0, 1], dtype=np.bool)
        rl.optimize_value_function()

        trans_probs2 = np.array([[0, .5, .5, 0, 0],
                                 [.2, .1, .3, .5, 0],
                                 [.3, .2, .4, .1, 0],
                                 [0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 1]],
                                dtype=np.float)
        rewards2 = np.zeros(5)
        rewards2[:4] = rewards()
        true_values = np.linalg.solve(np.eye(5) - rl.gamma * trans_probs2,
                                      rewards2)

        assert_allclose(rl.values, true_values[:4])

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

        assert_allclose(true_values, future_values)

        rl.terminal_states = np.array([0, 0, 0, 1], dtype=np.bool)
        future_values = rl.get_future_values(rl.policy)
        true_values[rl.terminal_states] = rewards()[rl.terminal_states]

        assert_allclose(future_values, true_values)

    @mock.patch('safe_learning.reinforcement_learning.'
                'PolicyIteration.get_future_values')
    def test_policy_update(self, future_value_mock):
        """Test the policy update."""
        dynamics = mock.Mock()
        rewards = mock.Mock()
        value_function = mock.Mock()

        states = np.arange(4)[:, None]
        actions = np.arange(2)[:, None]
        rl = PolicyIteration(states,
                             actions,
                             dynamics,
                             rewards,
                             value_function)

        future_value_mock.return_value = np.arange(4, dtype=np.float)
        rl.update_policy()

        assert_allclose(rl.policy, 0)

        def constraint(action):
            if np.all(action.base == 0):
                return np.array([0.1, 0.5, -0.1, -0.2], dtype=np.float)
            else:
                return np.ones(4, dtype=np.float)

        rl.update_policy(constraint=constraint)
        assert_allclose(rl.policy, np.array([[0, 0, 1, 1]], dtype=np.float).T)


if __name__ == '__main__':
    unittest.main()
