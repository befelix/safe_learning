"""Classes for reinforcement learning."""

from __future__ import absolute_import, division, print_function

import numpy as np


__all__ = ['PolicyIteration']


class PolicyIteration(object):
    """A class for policy iteration.

    Parameters
    ----------
    state_space : ndarray
        An array of physical states with one state vector on each row.
    action_space : ndarray
        An array of available actions with one action vector on each row.
    dynamics : callable
        A function that can be called with states and actions as inputs and
        returns future states.
    reward_function : callable
        A function that takes the state, action, and next state as input and
        returns the reward corresponding to this transition.
    function_approximator : callable
        The function approximator for the value function. It takes the states
        as inputs together with the vertex_values and returns the corresponding
        function values on the continuous domain.
    gamma : float
        The discount factor for reinforcement learning.
    terminal_states : ndarray (bool)
        A boolean vector which indicates terminal states. Defaults to False for
        all states. Terminal states get a terminal reward and are not updated.
    terminal_reward : float
        The reward associated with terminal states.
    """

    def __init__(self, state_space, action_space, dynamics, reward_function,
                 function_approximator, gamma=0.98, terminal_states=None,
                 terminal_reward=None):
        """Initialization.

        See `PolicyIteration` for details.
        """
        super(PolicyIteration, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.dynamics = dynamics
        self.reward_function = reward_function
        self.function_approximator = function_approximator
        self.gamma = gamma

        # Random initial policy
        self.policy = np.random.choice(action_space, size=len(state_space))
        self.values = np.zeros(len(state_space), dtype=np.float)

        self.terminal_reward = terminal_reward
        self.terminal_states = None
        if terminal_states is not None:
            self.terminal_states = terminal_states
            self.values[self.terminal_states] = self.terminal_reward

    def get_future_values(self, states, actions, out=None):
        """Return the value at the current states.

        Parameters
        ----------
        states : ndarray
            The states at which to evaluate.
        actions : ndarray
            The actions taken in the corresponding states.
        out : ndarray, optional
            The array to which to write the results.

        Returns
        -------
        The expected long term reward corresponding to the states and actions.
        """
        next_states = self.dynamics(states, actions)
        rewards = self.reward_function(states, actions, next_states)

        expected_values = self.function_approximator.function_values_at(
            next_states,
            vertex_values=self.values)

        if out is None:
            out = np.empty(len(states), dtype=np.float)

        # Perform value update
        out[:] = rewards + self.gamma * expected_values

        # Adapt values of terminal states
        if self.terminal_states is not None:
            out[self.terminal_states] = self.terminal_reward

        return out

    def update_value_function(self):
        """Perform one round of value updates."""
        self.get_future_values(self.state_space, self.policy, out=self.values)

    def update_policy(self):
        """Optimize the policy for a given value function."""
        # Initialize
        values = np.empty((len(self.state_space), len(self.action_space)),
                          dtype=np.float)
        action_size = (len(self.state_space), 1)

        # Compute values for each action
        for i, action in enumerate(self.action_space):
            self.get_future_values(self.state_space,
                                   np.broadcast_to(action, action_size),
                                   out=values[:, i])

        # Select best action for policy
        self.policy[:] = self.action_space[np.argmax(values, axis=1)]
