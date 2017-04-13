"""Implements the Lyapunov functions and learning."""

from __future__ import absolute_import, division, print_function

from collections import Sequence
from heapq import heappush, heappop
import itertools
from future.builtins import zip, range

import numpy as np
import tensorflow as tf

from .utilities import batchify, get_storage, set_storage, with_scope
from safe_learning import config

__all__ = ['Lyapunov', 'smallest_boundary_value', 'get_lyapunov_region']


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

    if hasattr(fun, 'feed_dict'):
        feed_dict = fun.feed_dict
    else:
        feed_dict = {}

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
    if hasattr(lyapunov, 'feed_dict'):
        feed_dict = lyapunov.feed_dict
    else:
        feed_dict = {}

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
        discretization constant.
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

        # Lyapunov values
        self.values = None
        self.update_values()

        # Origin node
        origin = np.argmin(np.abs(discretization.discrete_points), axis=1)
        self.origin = origin

        self.lipschitz_dynamics = lipschitz_dynamics
        self.lipschitz_lyapunov = lipschitz_lyapunov

    @property
    def threshold(self):
        """Return the safety threshold for the Lyapunov condition."""
        lv, lf = self.lipschitz_lyapunov, self.lipschitz_dynamics
        return -lv * (1. + lf) * self.epsilon

    @property
    def lipschitz(self):
        """Return the lipschitz constant."""
        lv, lf = self.lipschitz_lyapunov, self.lipschitz_dynamics
        return lv * (1. + lf)

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
            bound = self.lipschitz_lyapunov * tf.reduce_sum(error_bounds,
                                                            axis=1)
        else:
            bound = tf.constant(0., dtype=config.dtype)

        v_decrease = (self.lyapunov_function(next_states)
                      - self.lyapunov_function(states))

        return tf.squeeze(v_decrease, axis=1), bound

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
        order = np.argsort(self.values)
        state_order = self.discretization.index_to_state(order)

        storage = get_storage(self)

        if storage is None:
            # Set up the tensorflow pipeline
            states = tf.placeholder(config.dtype,
                                    shape=[None, self.discretization.ndim],
                                    name='verification_states')
            next_states = self.dynamics(states, self.policy(states))
            decrease = self.v_decrease_bound(states, next_states)

            storage = [('states', states), ('decrease', decrease)]
            set_storage(self, storage)
        else:
            states, decrease = storage.values()

        # Get relevant properties
        feed_dict = self.dynamics.feed_dict.copy()
        batch_size = config.gp_batch_size

        # reset the safe set
        safe_set = np.zeros_like(self.safe_set)

        if self.initial_safe_set is not None:
            safe_set[self.initial_safe_set] = True

            # Permute the initial safe set too
            safe_set = safe_set[order]

        # Verify safety in batches
        batch_generator = batchify((state_order, safe_set), batch_size)

        for state_batch, safe_batch in batch_generator:

            feed_dict[states] = state_batch
            result = decrease.eval(feed_dict=feed_dict)

            # TODO: Make the discretization adaptive depending on result
            negative = result <= self.threshold
            safe_batch |= negative

            # Boolean array: argmin returns first element that is False
            # If all are safe then it returns 0
            bound = np.argmin(safe_batch)

            # Check if there are unsafe elements in the batch
            if bound > 0 or not safe_batch[0]:
                # Make sure all following points are labeled as unsafe
                safe_batch[bound:] = False
                break

        # Restore the order of the safe set
        safe_nodes = order[safe_set]
        self.safe_set[:] = False
        self.safe_set[safe_nodes] = True
