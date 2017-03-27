"""Classes for reinforcement learning."""

from __future__ import absolute_import, division, print_function

from types import ModuleType

import tensorflow as tf
import numpy as np
try:
    import cvxpy
except ImportError as exception:
    cvxpy = exception

from .utilities import make_tf_fun, with_scope

__all__ = ['PolicyIteration']


class OptimizationError(Exception):
    pass


class PolicyIteration(object):
    """A class for policy iteration.

    Parameters
    ----------
    policy : callable
        The policy that maps states to actions.
    dynamics : callable
        A function that can be called with states and actions as inputs and
        returns future states.
    reward_function : callable
        A function that takes the state, action, and next state as input and
        returns the reward corresponding to this transition.
    value_function : instance of `DeterministicFunction`
        The function approximator for the value function. It is used to
        evaluate the value function at states.
    gamma : float
        The discount factor for reinforcement learning.
    """

    def __init__(self, policy, dynamics, reward_function, value_function,
                 gamma=0.98):
        """Initialization.

        See `PolicyIteration` for details.
        """
        super(PolicyIteration, self).__init__()
        self.dynamics = dynamics
        self.reward_function = reward_function
        self.value_function = value_function
        self.gamma = gamma

        state_space = self.value_function.discretization.all_points
        self.state_space = tf.stack(state_space, name='state_space')

        self.policy = policy

    @with_scope('future_values')
    def future_values(self, states, policy=None, actions=None):
        """Return the value at the current states.

        Parameters
        ----------
        states : ndarray
            The states at which to compute future values.
        policy : callable, optional
            The policy for which to evaluate. Defaults to `self.policy`. This
            argument is ignored if actions is not None.
        actions : array or tensor, optional
            The actions to be taken for the states.

        Returns
        -------
        The expected long term reward when taking an action according to the
        policy and then taking the value of self.value_function.
        """
        if actions is None:
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

    @with_scope('bellmann_error')
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

    @with_scope('value_iteration')
    def value_iteration(self):
        """Perform one step of value iteration."""
        future_values = self.future_values(self.state_space)
        return tf.assign(self.value_function.parameters, future_values,
                         name='value_iteration_update')

    @make_tf_fun(tf.float64)
    def _run_cvx_optimization(self, next_states, rewards):
        """A tensorflow wrapper around a cvxpy value function optimization.

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

    @with_scope('optimize_value_function')
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

    def discrete_policy_optimization(self, action_space, constraint=None):
        """Optimize the policy for a given value function.

        Parameters
        ----------
        action_space : ndarray
            The parameter value to evaluate (for each parameter). This is
            geared towards piecewise linear functions.
        constraint : callable
            A function that can be called with a policy. Returns the slack of
            the safety constraint for each state. A policy is safe if the slack
            is >=0 for all constraints.
        """
        n = self.state_space.shape[0]
        n_par, m = action_space.shape

        # Initialize
        values = np.empty((n, n_par), dtype=np.float)
        action_array = np.broadcast_to(np.zeros(m), (n, m))

        # Create future values object
        actions = tf.placeholder(tf.float64, shape=action_array.shape)
        future_values = self.future_values(self.state_space, actions=actions)
        feed_dict = {actions: action_array}

        # Compute values for each action
        for i, action in enumerate(action_space):
            # Update feed dict
            action_array.base[:] = action
            # Compute values
            values[:, i] = future_values.eval(feed_dict=feed_dict)[:, 0]

            if constraint is not None:
                # TODO: optimize safety if unsafe
                unsafe = constraint(action_array) < 0
                values[unsafe, i] = -np.inf

        # Select best action for policy
        assign = tf.assign(self.policy.parameters,
                           action_space[np.argmax(values, axis=1)])
        assign.eval()
