"""Classes for reinforcement learning."""

from __future__ import absolute_import, division, print_function

import numpy as np
try:
    import cvxpy
except ImportError:
    cvxpy = None

__all__ = ['PolicyIteration']


class OptimizationError(Exception):
    pass


class PolicyIteration(object):
    """A class for policy iteration.

    Parameters
    ----------
    state_space : ndarray
        An 2d array of physical states with one state vector on each row.
    action_space : ndarray
        An 2d array of available actions with one action vector on each row.
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
    terminal_states : ndarray (bool)
        A boolean vector which indicates terminal states. Defaults to False for
        all states. Terminal states get a terminal reward and are not updated.
    """

    def __init__(self, state_space, action_space, dynamics, reward_function,
                 function_approximator, gamma=0.98, terminal_states=None):
        """Initialization.

        See `PolicyIteration` for details.
        """
        super(PolicyIteration, self).__init__()

        self.state_space = np.asarray(state_space)
        self.action_space = np.asarray(action_space)
        self.dynamics = dynamics
        self.reward_function = reward_function
        self.value_function = function_approximator
        self.gamma = gamma
        self.terminal_states = terminal_states

        # Random initial policy
        action_index = np.random.randint(low=0,
                                         high=len(action_space),
                                         size=len(state_space))
        self.policy = self.action_space[action_index]
        self.values = np.zeros(len(state_space), dtype=np.float)

    @property
    def values(self):
        """Return the vertex values."""
        return self.value_function.parameters[:, 0]

    @values.setter
    def values(self, values):
        """Set the vertex values."""
        values = np.asarray(values)
        self.value_function.parameters = values.reshape(-1, 1)

    def get_future_values(self, actions):
        """Return the value at the current states.

        Parameters
        ----------
        actions : ndarray
            The actions taken in the corresponding states.

        Returns
        -------
        The expected long term reward corresponding to the states and actions.
        """
        states = self.state_space
        next_states = self.dynamics(states, actions)
        rewards = self.reward_function(states, actions, next_states)
        rewards = rewards.squeeze()

        expected_values = self.value_function(next_states).squeeze()

        # Perform value update
        updated_values = rewards + self.gamma * expected_values

        # Adapt values of terminal states
        if self.terminal_states is not None:
            terminal = self.terminal_states
            updated_values[terminal] = rewards[terminal]

        return updated_values

    def update_value_function(self):
        """Perform one round of value updates."""
        self.value_function.parameters = self.get_future_values(self.policy)

    def optimize_value_function(self):
        """Solve a linear program to optimize the value function."""
        if cvxpy is None:
            raise ImportError('This function requires the cvxpy module.')

        next_states = self.dynamics(self.state_space, self.policy)
        rewards = self.reward_function(self.state_space,
                                       self.policy,
                                       next_states)

        # Define random variables
        values = cvxpy.Variable(self.value_function.nindex)
        objective = cvxpy.Maximize(cvxpy.sum_entries(values))

        value_matrix = self.value_function.parameter_derivative(next_states)
        # Make cvxpy work with sparse matrices
        value_matrix = cvxpy.Constant(value_matrix)

        future_values = rewards + self.gamma * value_matrix * values

        if self.terminal_states is None:
            constraints = [values <= future_values]
        else:
            terminal = self.terminal_states
            not_terminal = ~self.terminal_states
            constraints = [values[not_terminal] <= future_values[not_terminal],
                           values[terminal] == rewards[terminal]]

        prob = cvxpy.Problem(objective, constraints)
        prob.solve()

        if not prob.status == cvxpy.OPTIMAL:
            raise OptimizationError('Optimization problem is {}'
                                    .format(prob.status))

        self.value_function.parameters[:] = values.value

    def update_policy(self, constraint=None):
        """Optimize the policy for a given value function.

        Parameters
        ----------
        constraint : callable
            A function that can be called with a policy. Returns the slack of
            the safety constraint for each state. A policy is safe if the slack
            is >=0 for all constraints.
        """
        # Initialize
        values = np.empty((len(self.state_space), len(self.action_space)),
                          dtype=np.float)

        action_size = (len(self.state_space), self.action_space.shape[1])
        action_array = np.broadcast_to(np.zeros(action_size[1]), action_size)

        # Compute values for each action
        for i, action in enumerate(self.action_space):
            action_array.base[:] = action
            values[:, i] = self.get_future_values(action_array)

            if constraint is not None:
                # TODO: optimize safety if unsafe
                unsafe = constraint(action_array) < 0
                values[unsafe, i] = -np.inf

        # Select best action for policy
        self.policy = self.action_space[np.argmax(values, axis=1)]
