"""Classes for reinforcement learning."""

from __future__ import absolute_import, division, print_function

from types import ModuleType

import tensorflow as tf
import numpy as np
try:
    import cvxpy
except ImportError as exception:
    cvxpy = exception

from .utilities import make_tf_fun

__all__ = ['PolicyIteration']


class OptimizationError(Exception):
    pass


class PolicyIteration(object):
    """A class for policy iteration.

    Parameters
    ----------
    state_space : ndarray
        An 2d array of physical states with one state vector on each row.
    policy : callable
        The policy that maps states to actions.
    dynamics : callable
        A function that can be called with states and actions as inputs and
        returns future states.
    reward_function : callable
        A function that takes the state, action, and next state as input and
        returns the reward corresponding to this transition.
    function_approximator : instance of `DeterministicFunction`
        The function approximator for the value function. It is used to
        evaluate the value function at states.
    gamma : float
        The discount factor for reinforcement learning.
    """

    def __init__(self, state_space, policy, dynamics, reward_function,
                 function_approximator, gamma=0.98):
        """Initialization.

        See `PolicyIteration` for details.
        """
        super(PolicyIteration, self).__init__()

        self.state_space = np.asarray(state_space)
        self.dynamics = dynamics
        self.reward_function = reward_function
        self.value_function = function_approximator
        self.gamma = gamma

        self.policy = policy
        self.values = tf.Variable(
            np.zeros((len(state_space), 1), dtype=np.float))

    def future_values(self, states, policy=None):
        """Return the value at the current states.

        Parameters
        ----------
        actions : ndarray
            The actions taken in the corresponding states.

        Returns
        -------
        The expected long term reward corresponding to the states and actions.
        """
        if policy is None:
            policy = self.policy

        actions = policy(states)

        next_states = self.dynamics(states, actions)
        rewards = self.reward_function(states, actions)

        # Only use the mean dynamics
        if isinstance(next_states, tuple):
            next_states, _ = next_states

        expected_values = self.value_function(next_states)

        # Perform value update
        updated_values = rewards + self.gamma * expected_values
        return updated_values

    def bellmann_error(self, states):
        """Compute the squared bellmann erlrror.

        Parameters
        ----------
        states : array

        Returns
        -------
        error : float
        """
        # Make sure we do not compute the gradient with respect to the
        # training target.
        target = tf.stop_gradient(self.future_values(states))
        # Squared bellmann error
        return tf.reduce_sum(tf.square(target - self.value_function(states)),
                             name='bellmann_error')

    def value_iteration(self):
        """Perform one step of value iteration."""
        future_values = self.future_values(self.state_space)
        return tf.assign(self.value_function.parameters, future_values,
                         name='value_iteration_update')

    @make_tf_fun(tf.float64)
    def _run_cvx_optimization(self, next_states, rewards):
        """A tensorflow wrapper around a cvxpy optimization for the value function.

        Parameters
        ----------
        next_states : ndarray
        rewards : ndarray

        Returns
        -------
        values : ndarray
            The optimal values at the states.
        """
        # Define random variables
        values = cvxpy.Variable(self.value_function.nindex)

        value_matrix = self.value_function.tri.parameter_derivative(
            next_states)
        # Make cvxpy work with sparse matrices
        value_matrix = cvxpy.Constant(value_matrix)

        objective = cvxpy.Maximize(cvxpy.sum_entries(values))
        constraints = [values <= rewards + self.gamma * value_matrix * values]
        prob = cvxpy.Problem(objective, constraints)

        # Solve optimization problem
        prob.solve()

        # Some error checking
        if not prob.status == cvxpy.OPTIMAL:
            raise OptimizationError('Optimization problem is {}'
                                    .format(prob.status))

        return np.array(values.value)

    def optimize_value_function(self):
        """Optimize the value function using cvx."""
        if not isinstance(cvxpy, ModuleType):
            raise cvxpy

        actions = self.policy(self.state_space)
        next_states = self.dynamics(self.state_space, actions)

        # Only use the mean dynamics
        if isinstance(next_states, tuple):
            next_states, var = next_states

        rewards = self.reward_function(self.state_space,
                                       actions)

        values = self._run_cvx_optimization(next_states, rewards)

        return tf.assign(self.value_function.parameters, values)
