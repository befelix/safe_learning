from __future__ import absolute_import, division, print_function

import numpy as np


__all__ = ['PolicyIteration']


class PolicyIteration(object):

    def __init__(self, state_space, action_space, dynamics, reward_function,
                 function_approximator, gamma=0.98, terminal_states=None,
                 terminal_reward=None):
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
        """Get the value at the current states"""
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
        out[self.terminal_states] = self.terminal_reward

        return out

    def update_value_function(self):
        """Perform one round of value updates.

        Parameters
        ----------
        states: ndarray
        actions: ndarray
        vertex_values: ndarray

        Returns
        -------
        values: ndarray
            The updated values
        """
        self.get_future_values(self.state_space, self.policy, out=self.values)

    def update_policy(self):
        """Optimize the policy for a given value function.

        Parameters
        ----------
        states: ndarray
        vertex_values: ndarray

        Returns
        -------
        policy: ndarray
            The optimal policy for the given value function.
        """
        # Initialize
        values = np.empty((len(self.state_space), len(self.action_space)),
                          dtype=np.float)
        actions = np.empty((len(self.state_space), 1), dtype=np.float)

        # Compute values for each action
        for i, action in enumerate(self.action_space):
            actions[:] = action
            self.get_future_values(self.state_space, actions, out=values[:, i])

        # Select best action for policy
        self.policy[:] = self.action_space[np.argmax(values, axis=1)]
