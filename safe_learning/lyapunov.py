"""Implements the Lyapunov functions and learning."""

from __future__ import absolute_import, division, print_function

from collections import Sequence
from heapq import heappush, heappop
import itertools
from future.builtins import zip, range

import numpy as np
import tensorflow as tf

from .utilities import (batchify, get_storage, set_storage, with_scope,
                        get_feed_dict, unique_rows)
from safe_learning import config

__all__ = ['Lyapunov', 'smallest_boundary_value', 'get_lyapunov_region',
           'get_safe_sample']


def smallest_boundary_value(fun, discretization):
    """Determine the smallest value of a function on its boundary.

    Parameters
    ----------
    fun : callable
        A tensorflow function that we want to evaluate.
    discretization : instance of `GridWorld`
        The discretization. If None, then the function is assumed to be
        defined on a discretization already.

    Returns
    -------
    min_value : float
        The smallest value on the boundary.
    """
    min_value = np.inf
    feed_dict = get_feed_dict(tf.get_default_graph())

    # Check boundaries for each axis
    for i in range(discretization.ndim):
        # Use boundary values only for the ith element
        tmp = list(discretization.discrete_points)
        tmp[i] = discretization.discrete_points[i][[0, -1]]

        # Generate all points
        columns = (x.ravel() for x in np.meshgrid(*tmp, indexing='ij'))
        all_points = np.column_stack(columns)

        # Update the minimum value
        smallest = tf.reduce_min(fun(all_points))
        min_value = min(min_value, smallest.eval(feed_dict=feed_dict))

    return min_value


def get_lyapunov_region(lyapunov, discretization, init_node):
    """Get the region within which a function is a Lyapunov function.

    Parameters
    ----------
    lyapunov : callable
        A tensorflow function.
    discretization : instance of `GridWorld`
        The discretization on which to check the increasing property.
    init_node : tuple
        The node at which to start the verification.

    Returns
    -------
    region : ndarray
        A boolean array that contains all the states for which lyapunov is a
        Lyapunov function that can be used for stability verification.
    """
    # Turn values into a multi-dim array
    feed_dict = lyapunov.feed_dict

    values = lyapunov(discretization.all_points).eval(feed_dict=feed_dict)
    lyapunov_values = values.reshape(discretization.num_points)

    # Starting point for the verification
    init_value = lyapunov_values[init_node]

    ndim = discretization.ndim
    num_points = discretization.num_points

    # Indeces for generating neighbors
    index_generator = itertools.product(*[(0, -1, 1) for _ in range(ndim)])
    neighbor_indeces = np.array(tuple(index_generator)[1:])

    # Array keeping track of visited nodes
    visited = np.zeros(discretization.num_points, dtype=np.bool)
    visited[init_node] = True

    # Create priority queue
    tiebreaker = itertools.count()
    last_value = init_value
    priority_queue = [(init_value, tiebreaker.next(), init_node)]

    while priority_queue:
        value, _, next_node = heappop(priority_queue)

        # Check if we reached the boundary of the discretization
        if np.any(0 == next_node) or np.any(next_node == num_points - 1):
            visited[tuple(next_node)] = False
            break

        # Make sure we are in the positive definite part of the function.
        if value < last_value:
            break

        last_value = value

        # Get all neighbors
        neighbors = next_node + neighbor_indeces

        # Remove neighbors that are already part of the visited set
        is_new = ~visited[np.split(neighbors.T, ndim)]
        neighbors = neighbors[is_new[0]]

        if neighbors.size:
            indices = np.split(neighbors.T, ndim)
            # add to visited set
            visited[indices] = True
            # get values
            values = lyapunov_values[indices][0]

            # add to priority queue
            for value, neighbor in zip(values, neighbors):
                heappush(priority_queue, (value, next(tiebreaker), neighbor))

    # Prune nodes that were neighbors, but haven't been visited
    for _, _, node in priority_queue:
        visited[tuple(node)] = False

    return visited


class Lyapunov(object):
    """A class for general Lyapunov functions.

    Parameters
    ----------
    discretization : ndarray
        A discrete grid on which to evaluate the Lyapunov function.
    lyapunov_function : callable or instance of `DeterministicFunction`
        The lyapunov function. Can be called with states and returns the
        corresponding values of the Lyapunov function.
    dynamics : a callable or an instance of `Function`
        The dynamics model. Can be either a deterministic function or something
        uncertain that includes error bounds.
    lipschitz_dynamics : ndarray or float
        The Lipschitz constant of the dynamics. Either globally, or locally
        for each point in the discretization (within a radius given by the
        discretization constant. This is the closed-loop Lipschitz constant
        including the policy!
    lipschitz_lyapunov : ndarray or float
        The Lipschitz constant of the lyapunov function. Either globally, or
        locally for each point in the discretization (within a radius given by
        the discretization constant.
    epsilon : float
        The discretization constant.
    policy : ndarray, optional
        The control policy used at each state (Same number of rows as the
        discretization).
    initial_set : ndarray, optional
        A boolean array of states that are known to be safe a priori.
    """

    def __init__(self, discretization, lyapunov_function, dynamics,
                 lipschitz_dynamics, lipschitz_lyapunov,
                 epsilon, policy, initial_set=None):
        """Initialization, see `Lyapunov` for details."""
        super(Lyapunov, self).__init__()

        self.discretization = discretization
        self.policy = policy

        # Keep track of the safe sets
        self.safe_set = np.zeros(np.prod(discretization.num_points),
                                 dtype=np.bool)

        self.initial_safe_set = initial_set
        if initial_set is not None:
            self.safe_set[initial_set] = True

        # Discretization constant
        self.epsilon = epsilon

        # Make sure dynamics are of standard framework
        self.dynamics = dynamics

        # Make sure Lyapunov fits into standard framework
        self.lyapunov_function = lyapunov_function

        # Storage for graph
        self._storage = dict()
        self.feed_dict = get_feed_dict(tf.get_default_graph())

        # Lyapunov values
        self.values = None

        self.c_max = tf.placeholder(config.dtype, shape=())
        self.feed_dict[self.c_max] = 0.

        self.update_values()

        self._lipschitz_dynamics = lipschitz_dynamics
        self._lipschitz_lyapunov = lipschitz_lyapunov

    def lipschitz_dynamics(self, states):
        """Return the Lipschitz constant for given states and actions.

        Parameters
        ----------
        states : ndarray or Tensor

        Returns
        -------
        lipschitz : float, ndarray or Tensor
            If lipschitz_dynamics is a callable then returns local Lipschitz
            constants. Otherwise returns the Lipschitz constant as a scalar.
        """
        if hasattr(self._lipschitz_dynamics, '__call__'):
            return self._lipschitz_dynamics(states)
        else:
            return self._lipschitz_dynamics

    def lipschitz_lyapunov(self, states):
        """Return the local Lipschitz constant at a given state.

        Parameters
        ----------
        states : ndarray or Tensor

        Returns
        -------
        lipschitz : float, ndarray or Tensor
            If lipschitz_lyapunov is a callable then returns local Lipschitz
            constants. Otherwise returns the Lipschitz constant as a scalar.
        """
        if hasattr(self._lipschitz_lyapunov, '__call__'):
            return self._lipschitz_lyapunov(states)
        else:
            return self._lipschitz_lyapunov

    def threshold(self, states):
        """Return the safety threshold for the Lyapunov condition.

        Parameters
        ----------
        states : ndarray or Tensor

        Returns
        -------
        lipschitz : float, ndarray or Tensor
            Either the scalar threshold or local thresholds, depending on
            whether lipschitz_lyapunov and lipschitz_dynamics are local or not.
        """
        lv = self.lipschitz_lyapunov(states)
        lf = self.lipschitz_dynamics(states)
        return -lv * (1. + lf) * self.epsilon

    def is_safe(self, state):
        """Return a boolean array that indicates whether the state is safe.

        Parameters
        ----------
        state : ndarray

        Returns
        -------
        safe : boolean
            Is true if the corresponding state is inside the safe set.
        """
        return self.safe_set[self.discretization.state_to_index(state)]

    def update_values(self):
        """Update the discretized values when the Lyapunov function changes."""
        points = self.discretization.all_points
        self.values = self.lyapunov_function(points).eval().squeeze()

    def v_decrease_confidence(self, states, next_states):
        """
        Compute confidence intervals for the decrease along Lyapunov function.

        Parameters
        ----------
        states : np.array
            The states at which to start (could be equal to discretization).
        next_states : np.array
            The dynamics evaluated at each point on the discretization. If
            the dynamics are uncertain then next_states is a tuple with mean
            and error bounds.

        Returns
        -------
        mean : np.array
            The expected decrease in values at each grid point.
        error_bounds : np.array
            The error bounds for the decrease at each grid point
        """
        if isinstance(next_states, Sequence):
            next_states, error_bounds = next_states
            lv = self.lipschitz_lyapunov(next_states)
            bound = lv * tf.reduce_sum(error_bounds, axis=1, keep_dims=True)
        else:
            bound = tf.constant(0., dtype=config.dtype)

        v_decrease = (self.lyapunov_function(next_states)
                      - self.lyapunov_function(states))

        return v_decrease, bound

    def v_decrease_bound(self, states, next_states):
        """
        Compute confidence intervals for the decrease along Lyapunov function.

        Parameters
        ----------
        states : np.array
            The states at which to start (could be equal to discretization).
        next_states : np.array or tuple
            The dynamics evaluated at each point on the discretization. If
            the dynamics are uncertain then next_states is a tuple with mean
            and error bounds.

        Returns
        -------
        upper_bound : np.array
            The upper bound on the change in values at each grid point.
        """
        v_dot, v_dot_error = self.v_decrease_confidence(states, next_states)

        return v_dot + v_dot_error

    def safety_constraint(self, policy, include_initial=True):
        """Return the safe set for a given policy.

        Parameters
        ----------
        policy : ndarray
            The policy used at each discretization point.
        include_initial : bool, optional
            Whether to include the initial safe set.

        Returns
        -------
        constraint : ndarray
            A boolean array indicating where the safety constraint is
            fulfilled.
        """
        prediction = self.dynamics(self.discretization, policy)
        v_dot_bound = self.v_decrease_bound(self.discretization, prediction)

        # Update the safe set
        v_dot_negative = v_dot_bound < self.threshold

        # Make sure initial safe set is included
        if include_initial and self.initial_safe_set is not None:
            v_dot_negative[self.initial_safe_set] = True

        return v_dot_negative

    @with_scope('update_safe_set')
    def update_safe_set(self):
        """Compute and update the safe set."""
        storage = get_storage(self._storage)

        if storage is None:
            # Placeholder for states to evaluate for safety
            tf_states = tf.placeholder(config.dtype,
                                       shape=[None, self.discretization.ndim],
                                       name='verification_states')
            tf_actions = self.policy(tf_states)
            next_states = self.dynamics(tf_states, tf_actions)

            decrease = self.v_decrease_bound(tf_states, next_states)
            threshold = self.threshold(tf_states)
            tf_negative = tf.squeeze(tf.less(decrease, threshold), axis=1)

            storage = [('tf_states', tf_states), ('negative', tf_negative)]
            set_storage(self._storage, storage)
        else:
            tf_states, tf_negative = storage.values()

        # Get relevant properties
        feed_dict = self.feed_dict
        batch_size = config.gp_batch_size

        # reset the safe set
        safe_set = np.zeros_like(self.safe_set)
        value_order = np.argsort(self.values)

        if self.initial_safe_set is not None:
            safe_set[self.initial_safe_set] = True

            # Permute the initial safe set too
            safe_set = safe_set[value_order]

        # Verify safety in batches
        batch_generator = batchify((value_order, safe_set), batch_size)
        index_to_state = self.discretization.index_to_state

        for i, (indices, safe_batch) in batch_generator:

            feed_dict[tf_states] = index_to_state(indices)

            # Update the safety with the safe_batch result
            safe_batch |= tf_negative.eval(feed_dict=feed_dict)
            # TODO: Make the discretization adaptive?

            # Boolean array: argmin returns first element that is False
            # If all are safe then it returns 0
            bound = np.argmin(safe_batch)

            # Check if there are unsafe elements in the batch
            if bound > 0 or not safe_batch[0]:
                # Make sure all following points are labeled as unsafe
                safe_batch[bound:] = False
                break

        # The largest index of a safe value
        max_index = i + bound - 1
        # Set placeholder for c_max to the corresponding value
        feed_dict[self.c_max] = self.values[value_order[max_index]]

        # Restore the order of the safe set
        safe_nodes = value_order[safe_set]
        self.safe_set[:] = False

        self.safe_set[safe_nodes] = True

        # Ensure the initial safe set is kept
        if self.initial_safe_set is not None:
            self.safe_set[self.initial_safe_set] = True


def perturb_actions(states, actions, perturbations, limits=None):
    """Create state-action pairs by perturbing the actions.

    Parameters
    ----------
    states : ndarray
        An (N x n) array of states at which we want to generate state-action
        pairs.
    actions : ndarray
        An (N x m) array of baseline actions at these states. These
        corresponds to the actions taken by the current policy.
    perturbations : ndarray
        An (X x m) array of policy perturbations that are to be applied to
        each state-action pair.
    limits : list
        List of action-limit tuples.

    Returns
    -------
    state-actions : ndarray
        An (N*X x n+m) array of state-actions pairs, where for each state
        the corresponding action is perturbed by the perturbations.
    """
    num_states, state_dim = states.shape

    # repeat states
    states_new = np.repeat(states, len(perturbations), axis=0)

    # generate perturbations from perturbations around baseline policy
    actions_new = (np.repeat(actions, len(perturbations), axis=0)
                   + np.tile(perturbations, (num_states, 1)))

    state_actions = np.column_stack((states_new, actions_new))

    if limits is not None:
        # Clip the actions
        perturbations = state_actions[:, state_dim:]
        np.clip(perturbations, limits[:, 0], limits[:, 1], out=perturbations)
        # Remove rows that are not unique
        state_actions = unique_rows(state_actions)

    return state_actions


_STORAGE = {}


@with_scope('get_safe_sample')
def get_safe_sample(lyapunov, perturbations=None, limits=None, positive=False,
                    num_samples=None, actions=None):
    """Compute a safe state-action pair for sampling.

    This function returns the most uncertain state-action pair close to the
    current policy (as a result of the perturbations) that is safe (maps
    back into the region of attraction).

    Parameters
    ----------
    lyapunov : instance of `Lyapunov'
        A Lyapunov instance with an up-to-date safe set.
    perturbations : ndarray
        An array that, on each row, has a perturbation that is added to the
        baseline policy in `lyapunov.policy`.
    limits : ndarray, optional
        The actuator limits. Of the form [(u_1_min, u_1_max), (u_2_min,..)...].
        If provided, state-action pairs are clipped to ensure the limits.
    positive : bool
        Whether the Lyapunov function is positive-definite (radially
        increasing). If not, additional checks are carried out to ensure
        safety of samples.
    num_samples : int, optional
        Number of samples to select (uniformly at random) from the safe
        states within lyapunov.discretization as testing points.
    actions : ndarray
        A list of actions to evaluate for each state. Ignored if perturbations
        is not None.

    Returns
    -------
    state-action : ndarray
        A row-vector that contains a safe state-action pair that is
        promising for obtaining future observations.
    var : float
        The uncertainty remaining at this state.
    """
    state_disc = lyapunov.discretization

    state_dim = state_disc.ndim
    if perturbations is None:
        action_dim = actions.shape[1]
    else:
        action_dim = perturbations.shape[1]
    action_limits = limits

    storage = get_storage(_STORAGE, index=lyapunov)

    if storage is None:
        # Placeholder for safe states
        tf_safe_states = tf.placeholder(config.dtype, shape=[None, state_dim])
        tf_actions = lyapunov.policy(tf_safe_states)

        # Placeholder for state-actions to evaluate
        tf_state_actions = tf.placeholder(config.dtype,
                                          shape=[None, state_dim + action_dim])

        mean, var = lyapunov.dynamics(tf_state_actions)
        bound = tf.reduce_sum(var, axis=1, keep_dims=True)
        # Account for deviations of the next value due to uncertainty
        lf = lyapunov.lipschitz_dynamics(tf_safe_states)
        lv = lyapunov.lipschitz_lyapunov(mean)
        error = lv * lf * bound
        values = lyapunov.lyapunov_function(mean) + error

        # Check whether the value is below c_max
        maps_inside = tf.less(values, lyapunov.c_max,
                              name='maps_inside_levelset')

        # Put everything into storage
        storage = [('tf_safe_states', tf_safe_states),
                   ('tf_actions', tf_actions),
                   ('tf_state_actions', tf_state_actions),
                   ('mean', mean),
                   ('bound', bound),
                   ('maps_inside', maps_inside)]
        set_storage(_STORAGE, storage, index=lyapunov)
    else:
        (tf_safe_states, tf_actions, tf_state_actions,
         mean, bound, maps_inside) = storage.values()

    # All the safe states within the discretization
    safe_states = state_disc.index_to_state(np.where(lyapunov.safe_set))

    # Subsample safe states
    if num_samples is not None and len(safe_states) > num_samples:
        idx = np.random.choice(len(safe_states), num_samples, replace=False)
        safe_states = safe_states[idx]

    # Update the feed_dict accordingly
    feed_dict = lyapunov.feed_dict
    feed_dict[tf_safe_states] = safe_states

    if perturbations is None:
        # Generate all state-action pairs
        arrays = [arr.ravel() for arr in np.meshgrid(safe_states,
                                                     actions,
                                                     indexing='ij')]
        state_actions = np.column_stack(arrays)
    else:
        # Generate state-action pairs around the current policy
        safe_actions = tf_actions.eval(feed_dict=feed_dict)
        state_actions = perturb_actions(safe_states,
                                        safe_actions,
                                        perturbations=perturbations,
                                        limits=action_limits)

    # Update feed value
    lyapunov.feed_dict[tf_state_actions] = state_actions

    # Evaluate the safety of the proposed state-action pairs
    session = tf.get_default_session()
    maps_inside, mean, var = session.run([maps_inside, mean, bound],
                                         feed_dict=lyapunov.feed_dict)
    maps_inside = maps_inside.squeeze(axis=1)

    # Check whether states map back to the safe set in expectation
    if not positive:
        next_state_index = lyapunov.discretization.state_to_index(mean)
        safe_in_expectation = lyapunov.safe_set[next_state_index]
        maps_inside &= safe_in_expectation

    # Return only state-actions pairs that are safe
    var_safe = var[maps_inside]
    max_id = np.argmax(var_safe)
    return state_actions[maps_inside, :][[max_id]], var_safe[max_id].squeeze()
