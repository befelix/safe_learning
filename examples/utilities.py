from __future__ import division, print_function

import sys
import os
import importlib
import numpy as np
import scipy
import tensorflow as tf
from scipy import signal

from safe_learning import DeterministicFunction
from safe_learning import UncertainFunction
from safe_learning import config
from safe_learning.utilities import (concatenate_inputs, get_storage, set_storage)
from collections import OrderedDict
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
if sys.version_info.major == 2:
    import imp

__all__ = ['constrained_batch_sampler', 'compute_closedloop_response',
           'import_from_directory', 'InvertedPendulum', 'CartPole']

_STORAGE = {}
NP_DTYPE = config.np_dtype
TF_DTYPE = config.dtype
DPI = 300

HEAT_MAP = plt.get_cmap('inferno', lut=None)
HEAT_MAP.set_over('white')
HEAT_MAP.set_under('black')

LEVEL_MAP = plt.get_cmap('viridis', lut=18)
LEVEL_MAP.set_over('gold')
LEVEL_MAP.set_under('indigo')


def tf_function_compose(tf_input, tf_function, num_compositions, output_name='function_composition', **kwargs):
    '''Apply a function multiple times to the input.'''
    
    def body(intermediate, idx):
        intermediate = tf_function(intermediate, **kwargs)
        idx = idx + 1
        return intermediate, idx

    def condition(rollout, states, idx):
        return idx < num_compositions

    initial_idx = tf.constant(0, dtype=TF_DTYPE)
    initial_intermediate = tf_input
    shape_invariants = [initial_intermediate.get_shape(), initial_idx.get_shape()]
    tf_output, _ = tf.while_loop(condition, body, [initial_intermediate, initial_idx], shape_invariants, name=output_name)
    
    return tf_output


def gridify(norms, maxes=None, num_points=25):    
    norms = np.asarray(norms).ravel()
    if maxes is None:
        maxes = norms
    else:
        maxes = np.asarray(maxes).ravel()
    limits = np.column_stack((- maxes / norms, maxes / norms))
    
    if isinstance(num_points, int):
        num_points = [num_points, ] * len(norms)
    grid = safe_learning.GridWorld(limits, num_points)
    return grid


def compute_roa(grid, closed_loop_dynamics, horizon=100, tol=1e-3, equilibrium=None, no_traj=True):
    if isinstance(grid, np.ndarray):
        all_points = grid
        nindex = grid.shape[0]
        ndim = grid.shape[1]
    else: # grid is a GridWorld instance
        all_points = grid.all_points
        nindex = grid.nindex
        ndim = grid.ndim
    
    # Forward-simulate all trajectories from initial points in the discretization
    if no_traj:
        end_states = all_points
        for t in range(1, horizon):
            end_states = closed_loop_dynamics(end_states)
    else:
        trajectories = np.empty((nindex, ndim, horizon))
        trajectories[:, :, 0] = all_points
        for t in range(1, horizon):
            trajectories[:, :, t] = closed_loop_dynamics(trajectories[:, :, t - 1])
        end_states = trajectories[:, :, -1]
            
    if equilibrium is None:
        equilibrium = np.zeros((1, ndim))
    
    # Compute an approximate ROA as all states that end up "close" to 0
    dists = np.linalg.norm(end_states - equilibrium, ord=2, axis=1, keepdims=True).ravel()
    roa = (dists <= tol)
    if no_traj:
        return roa
    else:
        return roa, trajectories


def binary_cmap(color='red', alpha=1.):
    if color=='red':
        color_code = (1., 0., 0., alpha)
    elif color=='green':
        color_code = (0., 1., 0., alpha)
    elif color=='blue':
        color_code = (0., 0., 1., alpha)
    else:
        color_code = color
    transparent_code = (1., 1., 1., 0.)
    return ListedColormap([transparent_code, color_code])


def find_nearest(array, value, sorted_1d=True):
    if not sorted_1d:
        array = np.sort(array)
    idx = np.searchsorted(array, value, side='left')
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx - 1]) < np.abs(value - array[idx])):
        idx -= 1
    return idx, array[idx]

def balanced_confusion_weights(y, y_true, scale_by_total=True):
    y = y.astype(np.bool)
    y_true = y_true.astype(np.bool)
    
    # Assuming labels in {0, 1}, count entries from confusion matrix
    TP = ( y &  y_true).sum()
    TN = (~y & ~y_true).sum()
    FP = ( y & ~y_true).sum()
    FN = (~y &  y_true).sum()
    confusion_counts = np.array([[TN, FN], [FP, TP]])
    
    # Scale up each sample by inverse of confusion weight
    weights = np.ones_like(y, dtype=float)
    weights[ y &  y_true] /= TP
    weights[~y & ~y_true] /= TN
    weights[ y & ~y_true] /= FP
    weights[~y &  y_true] /= FN
    if scale_by_total:
        weights *= y.size
    
    return weights, confusion_counts


def balanced_class_weights(y_true, scale_by_total=True):
    y = y_true.astype(np.bool)
    nP = y.sum()
    nN = y.size - y.sum()
    class_counts = np.array([nN, nP])
    
    weights = np.ones_like(y, dtype=float)
    weights[ y] /= nP
    weights[~y] /= nN
    if scale_by_total:
        weights *= y.size
    
    return weights, class_counts


class LyapunovNetwork(DeterministicFunction):
    def __init__(self, input_dim, layer_dims, activations, eps=1e-6,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 name='lyapunov_network'):
        """TODO."""
        super(LyapunovNetwork, self).__init__(name=name)
        self.input_dim = input_dim
        self.num_layers = len(layer_dims)
        self.activations = activations
        self.eps = eps
        self.initializer = initializer

        if layer_dims[0] < input_dim:
            raise ValueError('The first layer dimension must be at least the input dimension!')

        if np.all(np.diff(layer_dims) >= 0):
            self.output_dims = layer_dims
        else:
            raise ValueError('Each layer must maintain or increase the dimension of its input!')

        self.hidden_dims = np.zeros(self.num_layers, dtype=int)
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]
            self.hidden_dims[i] = np.ceil((layer_input_dim + 1) / 2).astype(int)

        # For printing results nicely
        self.layer_partitions = np.zeros(self.num_layers, dtype=int)
        for i, dim_diff in enumerate(np.diff(np.concatenate([[self.input_dim], self.output_dims]))):
            if dim_diff > 0:
                self.layer_partitions[i] = 2
            else:
                self.layer_partitions[i] = 1

    def build_evaluation(self, points):
        """Build the evaluation graph."""
        net = points
        if isinstance(net, np.ndarray):
            net = tf.constant(net)
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]

            W = tf.get_variable('weights_posdef_{}'.format(i), [self.hidden_dims[i], layer_input_dim], TF_DTYPE, self.initializer)
            kernel = tf.matmul(W, W, transpose_a=True) + self.eps * tf.eye(layer_input_dim, dtype=TF_DTYPE)
            dim_diff = self.output_dims[i] - layer_input_dim
            if dim_diff > 0:
                W = tf.get_variable('weights_{}'.format(i), [dim_diff, layer_input_dim], TF_DTYPE, self.initializer)
                kernel = tf.concat([kernel, W], axis=0)

            layer_output = tf.matmul(net, kernel, transpose_b=True)
            net = self.activations[i](layer_output, name='layer_output_{}'.format(i))

        # Quadratic form of output
        W = tf.get_variable('sqrt_shape_matrix', [self.output_dims[-1], self.output_dims[-1]], TF_DTYPE, self.initializer)
        P = tf.matmul(W, W, transpose_a=True) + self.eps * tf.eye(self.output_dims[-1], dtype=TF_DTYPE)
        values = tf.reduce_sum(tf.matmul(net, P) * net, axis=1, keepdims=True, name='quadratic_form')
        # values = tf.reduce_sum(tf.square(net), axis=1, keepdims=True, name='quadratic_form')

        return values

    def print_params(self):
        offset = 0
        params = self.parameters.eval()
        for i, dim_diff in enumerate(np.diff(np.concatenate([[self.input_dim], self.output_dims]))):
            print('Layer weights {}:'.format(i))
            W0 = params[offset + i]
            print('W0:\n{}'.format(W0))
            if dim_diff > 0:
                W1 = params[offset + 1 + i]
                print('W1:\n{}'.format(W1))
            else:
                offset += 1
            kernel = W0.T.dot(W0) + mapping.eps * np.eye(W0.shape[1])
            eigvals, _ = np.linalg.eig(kernel)
            print('Eigenvalues of (W0.T*W0 + eps*I):', eigvals, '\n')


class RBFNetwork(DeterministicFunction):
    def __init__(self, limits, num_states, variances=None, initializer=tf.contrib.layers.xavier_initializer(), name='rbf_network'):
        super(RBFNetwork, self).__init__(name=name)
        self.discretization = safe_learning.GridWorld(limits, num_states)
        if variances is not None:
            self.variances = variances
        else:
            self.variances = np.min(self.discretization.unit_maxes) ** 2
        self._initializer = initializer
        self._betas = 1 / (2 * self.variances)
        self.centres = self.discretization.all_points
        self._centres_3D = np.reshape(self.centres.T, (1, state_dim, self._hidden_units))
    
    def build_evaluation(self, states):
        W = tf.get_variable('weights', dtype=OPTIONS.tf_dtype, shape=[self.discretization.nindex, 1], initializer=self._initializer)
        states_3D = tf.expand_dims(states, axis=2)
        phi_X = tf.exp(-self._betas * tf.reduce_sum(tf.square(states_3D - self._centres_3D), axis=1, keep_dims=False))
        output = tf.matmul(phi_X, W)
        return output


def tf_rollout(initial_states, tf_dynamics, tf_policy, tf_reward_function, horizon, gamma=1.0):    
    if isinstance(gamma, tf.Tensor):
        tf_gamma = tf.cast(gamma, OPTIONS.tf_dtype)
    else:
        tf_gamma = tf.constant(gamma, dtype=OPTIONS.tf_dtype)
    
    if isinstance(horizon, tf.Tensor):
        tf_horizon = tf.cast(horizon, OPTIONS.tf_dtype)
    else:
        tf_horizon = tf.constant(horizon, dtype=OPTIONS.tf_dtype)
    
    def body(rollout, states, idx):
        actions = tf_policy(states)
        rollout = rollout + tf.pow(tf_gamma, idx) * tf_reward_function(states, actions)
        states = tf_dynamics(states, actions)
        idx = idx + 1
        return rollout, states, idx

    def condition(rollout, states, idx):
        return idx < tf_horizon

    initial_idx = tf.constant(0, dtype=OPTIONS.tf_dtype)
    initial_rollout = tf.constant(0, shape=[1, 1], dtype=OPTIONS.tf_dtype) # broadcasting takes care of N states
    
    shape_invariants = [tf.TensorShape([None, 1]), initial_states.get_shape(), initial_idx.get_shape()]
    rollout, final_states, _ = tf.while_loop(condition, body, [initial_rollout, initial_states, initial_idx], 
                                             shape_invariants)
    return rollout, final_states

        
def get_unique_subfolder(data_dir, prefix):
    """Return a unique sub-folder name.

    The returned string consists of prefix appended by an underscore and
    then an integer. The string is unique from other similarly prefixed
    folder names.
    """
    subfolders = [f for f in os.listdir(data_dir) if f[:len(prefix)] == prefix]
    if len(subfolders) == 0:
        unique_name = prefix + '_0'
    else:
        indices = [int(sf[len(prefix)+1:]) for sf in subfolders]
        max_index = np.max(indices)
        unique_name = prefix + '_' + str(max_index + 1)
    return unique_name


def constrained_batch_sampler(dynamics, policy, state_dim, batch_size,
                              action_limit=None, zero_pad=0):
    """TODO."""
    batch = tf.random_uniform([int(batch_size), state_dim], -1, 1,
                              dtype=TF_DTYPE, name='batch_sample')
    actions = policy(batch)
    future_batch = dynamics(batch, actions)
    maps_inside = tf.reduce_all(tf.logical_and(future_batch >= -1, future_batch <= 1), axis=1)
    maps_inside_idx = tf.squeeze(tf.where(maps_inside))
    constrained_batch = tf.gather(batch, maps_inside_idx)
    if action_limit is not None:
        c = np.abs(action_limit)
        undersaturated = tf.reduce_all(tf.logical_and(actions >= -c, actions <= c), axis=1)
        undersaturated_idx = tf.squeeze(tf.where(undersaturated))
        constrained_batch = tf.gather(batch, undersaturated_idx)
    if constrained_batch.get_shape()[0] == 0:
        constrained_batch = tf.zeros([1, state_dim], dtype=TF_DTYPE)
    if zero_pad > 0:
        zero_padding = tf.constant([[0, int(zero_pad)], [0, 0]])
        constrained_batch = tf.pad(constrained_batch, zero_padding)
    return constrained_batch


def sample_ellipsoid(covariance, level, num_samples):
    # Sample from unit sphere
    dim = covariance.shape[0]
    N = int(num_samples)
    mean = np.zeros(dim)
    Y = np.random.multivariate_normal(mean, covariance, N)
    Y = Y / np.linalg.norm(Y, ord=2, axis=1, keepdims=True)

    # Scale to spheres of different radii
    r = np.random.uniform(0., level, N).reshape((-1, 1))
    Y = Y*np.sqrt(r)

    # Transform to ellipsoid samples
    U = np.linalg.cholesky(covariance).T   # P = L.dot(L.T), U = L.T
    X = scipy.linalg.solve_triangular(U, Y.T, lower=False).T

    return X


def approx_local_lipschitz(func, query_point, epsilon, num_samples):
    """TODO. Only works for a single query point."""
    # Sample deviations from an epsilon-ball centered at the query point
    d = query_point.shape[1]
    dX = sample_ellipsoid(np.identity(d), epsilon, num_samples)
    X = dX + query_point

    # Compute the function deviations for each sample
    dF = func(X) - func(query_point)

    # Compute the 1-norm change for each sample from the query point
    nF = tf.norm(dF, ord=1, axis=1)
    nX = tf.norm(dX, ord=1, axis=1)

    # Compute Lipschitz constant as the maximum of the ratios
    L = tf.reduce_max(nF / nX)

    return L


def apply_2dfunc_along_last(func2d, arr3d):
    M, N, P = arr3d.shape
    arr2d = np.reshape(arr3d, (M, N*P), order='F')

    def func1d(arr1d):
        # Assume func2d maps PxN arrays to PxH arrays
        return func2d(arr1d.reshape((P, N)))

    # Apply along rows; result is MxPxH
    return np.apply_along_axis(func1d, 1, arr2d)


def approx_local_lipschitz_grid(func, query_points, epsilon, num_samples):
    # Sample deviations from an epsilon-ball centred at the query point
    d = query_points.shape[1]
    dX = sample_ellipsoid(np.identity(d), epsilon, num_samples)
    X = dX.T[None, :, :] + query_points[:, :, None]

    # Compute the function deviations for each sample, for each query point
    F = apply_2dfunc_along_last(func, X)
    dF = F - func(query_points)[:, :, None]

    # Compute the 1-norm change for each sample from each query point
    nF = np.linalg.norm(dF, ord=1, axis=2)
    nX = np.linalg.norm(dX, ord=1, axis=1).reshape((1, -1))

    # Compute Lipschitz constant as the maximum of the ratios
    L = np.amax(nF / nX, axis=1, keepdims=True)

    return L


def sample_box(limits, num_samples):
    """Sample uniformly from a box in the state space.

    Parameters
    ----------
    limits : list
        The rectangular domain from which to collect samples, given as a list
        of 2-D lists
    num_samples : int
        The number of samples to collect.

    Returns
    -------
    samples : ndarray
        State space samples, where each row is a sample

    """
    dim = len(limits)
    N = int(num_samples)
    for i in range(dim):
        single_dim = np.random.uniform(limits[i][0], limits[i][1], N).reshape((-1, 1))
        if i == 0:
            samples = single_dim
        else:
            samples = np.concatenate((samples, single_dim), axis=1)
    return samples


def sample_box_boundary(limits, num_samples):
    # First sample from the entire box
    N = int(num_samples)
    samples = sample_box(limits, N)

    # Uniformly choose which dimension to fix, and at which end point
    dim = len(limits)
    fixed_dims = np.random.choice(dim, N)
    end_points = [limits[d][i] for (d, i) in zip(fixed_dims, np.random.choice(2, N))]
    samples[np.arange(N), fixed_dims] = end_points

    return samples


def get_parameter_change(old_params, new_params, ord='inf'):
    """Measure the change in parameters.

    Parameters
    ----------
    old_params : list
        The old parameters as a list of ndarrays, typically from
        session.run(var_list)
    new_params : list
        The old parameters as a list of ndarrays, typically from
        session.run(var_list)
    ord : {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional
        TODO.

    Returns
    -------
    change : float
        The parameter change measured as a norm of the vector difference.

    """
    if ord=='inf':
        ord = np.inf
    elif ord=='-inf':
        ord = -np.inf
    
    old_params = np.concatenate([param.ravel() for param in old_params])
    new_params = np.concatenate([param.ravel() for param in new_params])
    change = np.linalg.norm(new_params - old_params, ord=ord)
    
    return change


def compute_closedloop_response(dynamics, policy, state_dim, steps, dt,
                                reference='zero', const=1.0, ic=None):
    """TODO."""
    action_dim = policy.output_dim

    if reference == 'impulse':
        r = np.zeros((steps + 1, action_dim))
        r[0, :] = (1 / dt) * np.ones((1, action_dim))
    elif reference == 'step':
        r = const*np.ones((steps + 1, action_dim))
    elif reference == 'zero':
        r = np.zeros((steps + 1, action_dim))

    times = dt*np.arange(steps + 1, dtype=NP_DTYPE).reshape((-1, 1))
    states = np.zeros((steps + 1, state_dim), dtype=NP_DTYPE)
    actions = np.zeros((steps + 1, action_dim), dtype=NP_DTYPE)
    if ic is not None:
        states[0, :] = np.asarray(ic, dtype=NP_DTYPE).reshape((1, state_dim))

    session = tf.get_default_session()
    with tf.name_scope('compute_closedloop_response'):
        current_ref = tf.placeholder(TF_DTYPE, shape=[1, action_dim])
        current_state = tf.placeholder(TF_DTYPE, shape=[1, state_dim])
        current_action = policy(current_state)
        next_state = dynamics(current_state, current_action + current_ref)
        data = [current_action, next_state]

    for i in range(steps):
        feed_dict = {current_ref: r[[i], :], current_state: states[[i], :]}
        actions[i, :], states[i + 1, :] = session.run(data, feed_dict)

    # Get the last action for completeness
    feed_dict = {current_ref: r[[-1], :], current_state: states[[-1], :]}
    actions[-1, :], _ = session.run(data, feed_dict)

    return states, actions, times, r


def import_from_directory(library, path):
    """Import a library from a directory outside the path.

    Parameters
    ----------
    library: string
        The name of the library.
    path: string
        The path of the folder containing the library.

    """
    try:
        return importlib.import_module(library)
    except ImportError:
        module_path = os.path.abspath(path)
        version = sys.version_info

        if version.major == 2:
            f, filename, desc = imp.find_module(library, [module_path])
            return imp.load_module(library, f, filename, desc)
        else:
            sys.path.append(module_path)
            return importlib.import_module(library)


class ContinuousTimeDynamicSystem(DeterministicFunction):
    """TODO."""
    def __init__(self, dt=0.01, normalization=None, name='Dynamics'):
        """Initialization; see `ContinuousTimeDynamicSystem`."""
        super(ContinuousTimeDynamicSystem, self).__init__(name)
        self.dt = dt
        # self.state_dim = 4
        # self.action_dim = 1
        self.normalization = normalization
        if normalization is not None:
            self.normalization = [np.array(norm, dtype=config.np_dtype)
                                  for norm in normalization]
            self.inv_norm = [norm ** -1 for norm in self.normalization]
        

class VanDerPol(DeterministicFunction):
    """Van der Pol oscillator."""
    def __init__(self, damping=1, dt=0.01, normalization=None):
        """Initialization; see `VanDerPol`."""
        super(VanDerPol, self).__init__(name='VanDerPol')
        self.damping = damping
        self.dt = dt
        self.state_dim = 2
        self.action_dim = 0
        self.normalization = normalization
        if normalization is not None:
            self.normalization = np.array(normalization, dtype=config.np_dtype)
            self.inv_norm = self.normalization ** -1

    def normalize(self, state):
        """Normalize states and actions."""
        if self.normalization is None:
            return state
        Tx_inv = np.diag(self.inv_norm)
        state = tf.matmul(state, Tx_inv)
        return state

    def denormalize(self, state):
        """De-normalize states and actions."""
        if self.normalization is None:
            return state
        Tx = np.diag(self.normalization)
        state = tf.matmul(state, Tx)
        return state

    def linearize(self):
        """Return the discretized, scaled, linearized system.

        Returns
        -------
        Ad : ndarray
            The discrete-time state matrix.

        """
        A = np.array([[0, -1], [1, -1]], dtype=config.np_dtype)
        if self.normalization is not None:
            Tx = np.diag(self.normalization)
            Tx_inv = np.diag(self.inv_norm)
            A = np.linalg.multi_dot((Tx_inv, A, Tx))
        B = np.zeros([2, 1])
        Ad, _, _, _, _ = signal.cont2discrete((A, B, 0, 0), self.dt, method='zoh')
        return Ad

    @concatenate_inputs(start=1)
    def build_evaluation(self, state_action):
        """Evaluate the dynamics."""
        state, _ = tf.split(state_action, [2, 1], axis=1)
        state = self.denormalize(state)
        inner_euler_steps = 10
        dt = self.dt / inner_euler_steps
        for _ in range(inner_euler_steps):
            state_derivative = self.ode(state)
            state = state + dt * state_derivative
        return self.normalize(state)

    def ode(self, state):
        """Compute the state time-derivative.

        Parameters
        ----------
        state: ndarray or Tensor
            States.

        Returns
        -------
        state_derivative: Tensor
            The state derivative according to the dynamics.

        """
        x, y = tf.split(state, 2, axis=1)
        x_dot = - y
        y_dot = x + self.damping * (x ** 2 - 1) * y
        state_derivative = tf.concat((x_dot, y_dot), axis=1)
        return state_derivative


class CartPole(DeterministicFunction):
    """Cart with mounted inverted pendulum.

    Parameters
    ----------
    pendulum_mass : float
    cart_mass : float
    length : float
    dt : float, optional
        The sampling period used for discretization.
    normalization : tuple, optional
        A tuple (Tx, Tu) of 1-D arrays or lists used to normalize the state and
        action, such that x = diag(Tx) * x_norm and u = diag(Tu) * u_norm.

    """
    def __init__(self, pendulum_mass, cart_mass, length, rot_friction=0.0,
                 dt=0.01, normalization=None):
        """Initialization; see `CartPole`."""
        super(CartPole, self).__init__(name='CartPole')
        self.pendulum_mass = pendulum_mass
        self.cart_mass = cart_mass
        self.length = length
        self.rot_friction = rot_friction
        self.dt = dt
        self.gravity = 9.81
        self.state_dim = 4
        self.action_dim = 1
        self.normalization = normalization
        if normalization is not None:
            self.normalization = [np.array(norm, dtype=config.np_dtype)
                                  for norm in normalization]
            self.inv_norm = [norm ** -1 for norm in self.normalization]

    def normalize(self, state, action):
        """Normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx_inv, Tu_inv = map(np.diag, self.inv_norm)
        state = tf.matmul(state, Tx_inv)

        if action is not None:
            action = tf.matmul(action, Tu_inv)

        return state, action

    def denormalize(self, state, action):
        """De-normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx, Tu = map(np.diag, self.normalization)

        state = tf.matmul(state, Tx)
        if action is not None:
            action = tf.matmul(action, Tu)

        return state, action

    def linearize(self):
        """Return the discretized, scaled, linearized system.

        Returns
        -------
        Ad : ndarray
            The discrete-time state matrix.
        Bd : ndarray
            The discrete-time action matrix.

        """
        m = self.pendulum_mass
        M = self.cart_mass
        L = self.length
        b = self.rot_friction
        g = self.gravity

        A = np.array([[0, 0,                     1, 0                            ],
                      [0, 0,                     0, 1                            ],
                      [0, g * m / M,             0, -b / (M * L)                 ],
                      [0, g * (m + M) / (L * M), 0, -b * (m + M) / (m * M * L**2)]],
                     dtype=config.np_dtype)

        B = np.array([0, 0, 1 / M, 1 / (M * L)]).reshape((-1, self.action_dim))

        if self.normalization is not None:
            Tx, Tu = map(np.diag, self.normalization)
            Tx_inv, Tu_inv = map(np.diag, self.inv_norm)
            A = np.linalg.multi_dot((Tx_inv, A, Tx))
            B = np.linalg.multi_dot((Tx_inv, B, Tu))

        Ad, Bd, _, _, _ = signal.cont2discrete((A, B, 0, 0), self.dt,
                                               method='zoh')
        return Ad, Bd

    @concatenate_inputs(start=1)
    def build_evaluation(self, state_action):
        """Evaluate the dynamics."""
        # Denormalize
        state, action = tf.split(state_action, [4, 1], axis=1)
        state, action = self.denormalize(state, action)

        inner_euler_steps = 10
        dt = self.dt / inner_euler_steps
        for _ in range(inner_euler_steps):
            state_derivative = self.ode(state, action)
            state = state + dt * state_derivative

        return self.normalize(state, None)[0]

    def ode(self, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        state: ndarray or Tensor
            States.
        action: ndarray or Tensor
            Actions.

        Returns
        -------
        state_derivative: Tensor
            The state derivative according to the dynamics.

        """
        # Physical dynamics
        m = self.pendulum_mass
        M = self.cart_mass
        L = self.length
        b = self.rot_friction
        g = self.gravity

        x, theta, v, omega = tf.split(state, [1, 1, 1, 1], axis=1)

        x_dot = v
        theta_dot = omega

        det = L*(M + m*tf.square(tf.sin(theta)))
        v_dot = (action - m*L*tf.square(omega)*tf.sin(theta) - b*omega*tf.cos(theta) + 0.5*m*g*L*tf.sin(2*theta)) * L/det
        omega_dot = (action*tf.cos(theta) - 0.5*m*L*tf.square(omega)*tf.sin(2*theta) - b*(m + M)*omega/(m*L)
                     + (m + M)*g*tf.sin(theta)) / det

        state_derivative = tf.concat((x_dot, theta_dot, v_dot, omega_dot), axis=1)

        return state_derivative


class InvertedPendulum(DeterministicFunction):
    """Inverted Pendulum.

    Parameters
    ----------
    mass : float
    length : float
    friction : float, optional
    dt : float, optional
        The sampling time.
    normalization : tuple, optional
        A tuple (Tx, Tu) of arrays used to normalize the state and actions. It
        is so that diag(Tx) *x_norm = x and diag(Tu) * u_norm = u.

    """

    def __init__(self, mass, length, friction=0, dt=1 / 80,
                 normalization=None):
        """Initialization; see `InvertedPendulum`."""
        super(InvertedPendulum, self).__init__()
        self.mass = mass
        self.length = length
        self.gravity = 9.81
        self.friction = friction
        self.dt = dt

        self.normalization = normalization
        if normalization is not None:
            self.normalization = [np.array(norm, dtype=config.np_dtype)
                                  for norm in normalization]
            self.inv_norm = [norm ** -1 for norm in self.normalization]

    @property
    def inertia(self):
        """Return inertia of the pendulum."""
        return self.mass * self.length ** 2

    def normalize(self, state, action):
        """Normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx_inv, Tu_inv = map(np.diag, self.inv_norm)
        state = tf.matmul(state, Tx_inv)

        if action is not None:
            action = tf.matmul(action, Tu_inv)

        return state, action

    def denormalize(self, state, action):
        """De-normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx, Tu = map(np.diag, self.normalization)

        state = tf.matmul(state, Tx)
        if action is not None:
            action = tf.matmul(action, Tu)

        return state, action

    def linearize(self):
        """Return the linearized system.

        Returns
        -------
        a : ndarray
            The state matrix.
        b : ndarray
            The action matrix.

        """
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        A = np.array([[0, 1],
                      [gravity / length, -friction / inertia]],
                     dtype=config.np_dtype)

        B = np.array([[0],
                      [1 / inertia]],
                     dtype=config.np_dtype)

        if self.normalization is not None:
            Tx, Tu = map(np.diag, self.normalization)
            Tx_inv, Tu_inv = map(np.diag, self.inv_norm)

            A = np.linalg.multi_dot((Tx_inv, A, Tx))
            B = np.linalg.multi_dot((Tx_inv, B, Tu))

        sys = signal.StateSpace(A, B, np.eye(2), np.zeros((2, 1)))
        sysd = sys.to_discrete(self.dt)
        return sysd.A, sysd.B

    @concatenate_inputs(start=1)
    def build_evaluation(self, state_action):
        """Evaluate the dynamics."""
        # Denormalize
        state, action = tf.split(state_action, [2, 1], axis=1)
        state, action = self.denormalize(state, action)

        n_inner = 10
        dt = self.dt / n_inner
        for i in range(n_inner):
            state_derivative = self.ode(state, action)
            state = state + dt * state_derivative

        return self.normalize(state, None)[0]

    def ode(self, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        states: ndarray or Tensor
            Unnormalized states.
        actions: ndarray or Tensor
            Unnormalized actions.

        Returns
        -------
        x_dot: Tensor
            The normalized derivative of the dynamics

        """
        # Physical dynamics
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        angle, angular_velocity = tf.split(state, 2, axis=1)

        x_ddot = gravity / length * tf.sin(angle) + action / inertia

        if friction > 0:
            x_ddot -= friction / inertia * angular_velocity

        state_derivative = tf.concat((angular_velocity, x_ddot), axis=1)

        # Normalize
        return state_derivative



def debug(lyapunov, true_dynamics, state_norm, Nmax=None, do_print=False, newly_safe_only=True, plot=None, old_safe_set=None, fixed_state=(0., 0., 0., 0.)):

    storage = get_storage(_STORAGE, index=lyapunov)
    if storage is None:
        tf_states = tf.placeholder(TF_DTYPE, shape=[None, lyapunov.discretization.ndim], name='states')
        tf_actions = lyapunov.policy(tf_states)
        tf_values = lyapunov.lyapunov_function(tf_states)

        tf_future_states = lyapunov.dynamics(tf_states, tf_actions)
        tf_mean_decrease, tf_error = lyapunov.v_decrease_confidence(tf_states, tf_future_states)
        tf_threshold = lyapunov.threshold(tf_states, lyapunov.tau)

        tf_true_future_states = true_dynamics(tf_states, tf_actions)
        tf_true_decrease = lyapunov.lyapunov_function(tf_true_future_states) - tf_values

        storage = [('tf_states', tf_states),
                   ('tf_values', tf_values),
                   ('tf_threshold', tf_threshold),
                   ('tf_mean_decrease', tf_mean_decrease),
                   ('tf_error', tf_error),
                   ('tf_true_decrease', tf_true_decrease)]
        set_storage(_STORAGE, storage, index=lyapunov)
    else:
        tf_states, tf_values, tf_threshold, tf_mean_decrease, tf_error, tf_true_decrease = storage.values()

    session = tf.get_default_session()
    feed_dict = lyapunov.feed_dict

    if do_print:
        # print('Dynamics Lipschitz constant (L_f*(L_pi + 1)): {}'.format(lyapunov.lipschitz_dynamics(0).eval()))
        print('beta: {}'.format(lyapunov.dynamics.functions[0].beta))
        print('tau: {}'.format(lyapunov.tau))
        print('c_n: {}\n'.format(lyapunov.feed_dict[lyapunov.c_max]))

        # Only consider those states that became safe from updating the model
        if newly_safe_only:
            safe_set = np.logical_xor(lyapunov.safe_set, lyapunov.initial_safe_set)
        else:
            safe_set = lyapunov.safe_set
        safe_states = lyapunov.discretization.all_points[safe_set, :]

        if len(safe_states) == 0:
            print('\nNo new safe states! Try collecting more data to improve model.')
        else:
            feed_dict[tf_states] = safe_states
            values, mean_decrease, error, threshold = session.run([tf_values, tf_mean_decrease, tf_error, tf_threshold], feed_dict)

            # Use pandas frame for nice printing
            decrease = mean_decrease + error
            N = np.clip(np.ceil(threshold / decrease).astype(np.int), 0, None)
            tau = lyapunov.tau / N

            data = OrderedDict()
            order = np.argsort(values.ravel())

            data['v(x)'] = values[order].ravel()
            data['v(mu) - v(x)'] = mean_decrease[order].ravel()
            data['error'] = error[order].ravel()
            data['N'] = N[order].ravel()
            data['tau req.'] = tau[order].ravel()

            frame = DataFrame(data)
            print(frame, '\n')

    if plot=='pendulum':
        plt.rc('font', size=5)
        plt.rc('lines', markersize=3)

        feed_dict[tf_states] = lyapunov.discretization.all_points
        threshold, mean_decrease, error, true_decrease = session.run([tf_threshold, tf_mean_decrease, tf_error, tf_true_decrease], feed_dict)
        if not isinstance(error, np.ndarray):
            error = np.zeros_like(mean_decrease)

        if isinstance(threshold, np.ndarray) and threshold.shape[0] > 1:
            threshold = threshold.reshape(lyapunov.discretization.num_points)
        mean_decrease = mean_decrease.reshape(lyapunov.discretization.num_points)
        error = error.reshape(lyapunov.discretization.num_points)
        true_decrease = true_decrease.reshape(lyapunov.discretization.num_points)
        decrease = mean_decrease + error

        safe_set = lyapunov.safe_set
        # if old_safe_set is not None:
        #     safe_set = safe_set.astype(NP_DTYPE) + old_safe_set.astype(NP_DTYPE)
        safe_set = safe_set.reshape(lyapunov.discretization.num_points).astype(int)

        if state_norm is not None:
            theta_max, omega_max = state_norm
            scale = np.array([np.rad2deg(theta_max), np.rad2deg(omega_max)]).reshape((-1, 1))
            limits = scale * lyapunov.discretization.limits
        else:
            limits = lyapunov.discretization.limits

        # Figure
        fig, axes = plt.subplots(2, 3, figsize=(8, 4), dpi=DPI)
        fig.subplots_adjust(wspace=0.6, hspace=0.15)
        for ax in axes.ravel():
            ax.set_xlabel(r'$\theta$ [deg]')
            ax.set_ylabel(r'$\omega$ [deg/s]')

        cmap = colors.ListedColormap(['indigo', 'yellow', 'orange', 'red'])
        bounds = [0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        #
        z = (decrease < threshold).astype(int) + 2 * safe_set
        ax = axes[0, 0]
        im = ax.imshow(z.T, origin='lower', extent=limits.ravel(), aspect=limits[0, 0] / limits[1, 0], cmap=cmap, norm=norm)
        cbar = fig.colorbar(im, ax=ax, boundaries=bounds, ticks=[0.5, 1.5, 2.5, 3.5])
        cbar.ax.set_yticklabels(['unsafe\n$\Delta v \geq 0$', 'unsafe\n$\Delta v < 0$', 'safe\n$\Delta v \geq 0$', 'safe\n$\Delta v < 0$'])
        if isinstance(lyapunov.dynamics, UncertainFunction):
            if state_norm is not None:
                theta_max, omega_max = state_norm
                scale = np.array([np.rad2deg(theta_max), np.rad2deg(omega_max)]).ravel()
                X = scale * lyapunov.dynamics.functions[0].X[:, :2]
            else:
                X = lyapunov.dynamics.functions[0].X[:, :2]
            ax.plot(X[:, 0], X[:, 1], 'kx')

        #
        cmap = plt.get_cmap('viridis', lut=None)
        cmap.set_over('gold')
        cmap.set_under('indigo')
        z = lyapunov._n.reshape(lyapunov.discretization.num_points)
        ax = axes[1, 0]
        im = ax.imshow(z.T, origin='lower', extent=limits.ravel(), aspect=limits[0, 0] / limits[1, 0], cmap=cmap, vmin=1)
        fig.colorbar(im, ax=ax, label=r'$N$')

        #
        z = decrease
        ax = axes[0, 1]
        im = ax.imshow(z.T, origin='lower', extent=limits.ravel(), aspect=limits[0, 0] / limits[1, 0], cmap=HEAT_MAP, vmax=0.)
        fig.colorbar(im, ax=ax, label=r'$v(\mu_\pi(x)) - v(x) + error$')

        #
        z = true_decrease
        ax = axes[1, 1]
        im = ax.imshow(z.T, origin='lower', extent=limits.ravel(), aspect=limits[0, 0] / limits[1, 0], cmap=HEAT_MAP, vmax=0.)
        fig.colorbar(im, ax=ax, label=r'$v(f(x)) - v(x)$')

        #
        z = mean_decrease - threshold
        ax = axes[0, 2]
        im = ax.imshow(z.T, origin='lower', extent=limits.ravel(), aspect=limits[0, 0] / limits[1, 0], cmap=HEAT_MAP, vmax=0.)
        fig.colorbar(im, ax=ax, label=r'$v(\mu_\pi(x)) - v(x) + L_{\Delta v}\tau$')

        #
        z = true_decrease - threshold
        ax = axes[1, 2]
        im = ax.imshow(z.T, origin='lower', extent=limits.ravel(), aspect=limits[0, 0] / limits[1, 0], cmap=HEAT_MAP, vmax=0.)
        fig.colorbar(im, ax=ax, label=r'$v(f(x)) - v(x) + L_{\Delta v}\tau$')

        plt.show()

    elif plot=='cartpole':
        safe_set = lyapunov.safe_set
        fixed_state = np.asarray(fixed_state, dtype=NP_DTYPE)
        for i in range(4):
            dist = np.square(lyapunov.discretization.discrete_points[i] - fixed_state[i])
            idx = np.argmin(dist)
            fixed_state[i] = lyapunov.discretization.discrete_points[i][idx]
        x_fix, theta_fix, v_fix, omega_fix = fixed_state

        pos_set = np.logical_and(lyapunov.discretization.all_points[:, 1] == theta_fix, lyapunov.discretization.all_points[:, 3] == omega_fix)
        vel_set = np.logical_and(lyapunov.discretization.all_points[:, 0] == x_fix, lyapunov.discretization.all_points[:, 2] == v_fix)

        feed_dict[tf_states] = lyapunov.discretization.all_points[pos_set, :]
        threshold_pos, dec_pos, true_dec_pos = session.run([tf_threshold, tf_mean_decrease + tf_error, tf_true_decrease], feed_dict)
        if np.prod(threshold_pos.shape) > 1:
            threshold_pos = threshold_pos.reshape(lyapunov.discretization.num_points[(0, 2), ])
        dec_pos = dec_pos.reshape(lyapunov.discretization.num_points[(0, 2), ])
        true_dec_pos = true_dec_pos.reshape(lyapunov.discretization.num_points[(0, 2), ])
        safe_pos = safe_set[pos_set].reshape(lyapunov.discretization.num_points[(0, 2), ]).astype(NP_DTYPE)

        feed_dict[tf_states] = lyapunov.discretization.all_points[vel_set, :]
        threshold_vel, dec_vel, true_dec_vel = session.run([tf_threshold, tf_mean_decrease + tf_error, tf_true_decrease], feed_dict)
        if np.prod(threshold_vel.shape) > 1:
            threshold_vel = threshold_vel.reshape(lyapunov.discretization.num_points[(1, 3), ])
        dec_vel = dec_vel.reshape(lyapunov.discretization.num_points[(1, 3), ])
        true_dec_vel = true_dec_vel.reshape(lyapunov.discretization.num_points[(1, 3), ])
        safe_vel = safe_set[vel_set].reshape(lyapunov.discretization.num_points[(1, 3), ]).astype(NP_DTYPE)

        if state_norm is not None:
            x_max, theta_max, v_max, omega_max = state_norm
            scale = np.array([x_max, np.rad2deg(theta_max), v_max, np.rad2deg(omega_max)]).reshape((-1, 1))
            limits = scale * lyapunov.discretization.limits
            x_fix, theta_fix, v_fix, omega_fix = fixed_state * scale.ravel()
        else:
            limits = lyapunov.discretization.limits

        # Figure
        plt.rc('font', size=6)
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=DPI)
        fig.subplots_adjust(wspace=0.6, hspace=0.1)

        ###################################################################################################

        # Colormap
        cmap = colors.ListedColormap(['indigo', 'yellow', 'orange', 'red'])
        bounds = [0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        #
        z = (dec_pos < threshold_pos) + 2 * safe_pos
        ax = axes[0, 0]
        ax.set_title(r'$\theta = %.3g$ deg, $\omega = %.3g$ deg/s' % (theta_fix, omega_fix))
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$v$ [m/s]')
        im = ax.imshow(z.T, origin='lower', extent=limits[(0, 2), :].ravel(), aspect=limits[0, 0] / limits[2, 0], cmap=cmap, norm=norm)
        cbar = fig.colorbar(im, ax=ax, boundaries=bounds, ticks=[0.5, 1.5, 2.5, 3.5])
        cbar.ax.set_yticklabels(['unsafe\n$\Delta v \geq 0$', 'unsafe\n$\Delta v < 0$', 'safe\n$\Delta v \geq 0$', 'safe\n$\Delta v < 0$'])

        #
        z = (dec_vel < threshold_vel) + 2 * safe_vel
        ax = axes[1, 0]
        ax.set_title(r'$x = %.3g$ m, $v = %.3g$ m/s' % (x_fix, v_fix))
        ax.set_xlabel(r'$\theta$ [deg]')
        ax.set_ylabel(r'$\omega$ [deg/s]')
        im = ax.imshow(z.T, origin='lower', extent=limits[(1, 3), :].ravel(), aspect=limits[1, 0] / limits[3, 0], cmap=cmap, norm=norm)
        cbar = fig.colorbar(im, ax=ax, boundaries=bounds, ticks=[0.5, 1.5, 2.5, 3.5])
        cbar.ax.set_yticklabels(['unsafe\n$\Delta v \geq 0$', 'unsafe\n$\Delta v < 0$', 'safe\n$\Delta v \geq 0$', 'safe\n$\Delta v < 0$'])

        ###################################################################################################

        cmap = plt.get_cmap('viridis')
        cmap.set_under('indigo')
        cmap.set_over('gold')

        #
        z = dec_pos
        ax = axes[0, 1]
        ax.set_title(r'$\theta = %.3g$ deg, $\omega = %.3g$ deg/s' % (theta_fix, omega_fix))
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$v$ [m/s]')
        im = ax.imshow(z.T, origin='lower', extent=limits[(0, 2), :].ravel(), aspect=limits[0, 0] / limits[2, 0], cmap=HEAT_MAP, vmax=0.0)
        cbar = fig.colorbar(im, ax=ax, label=r'$v(\mu(x)) - v(x) + error$')

        #
        z = dec_vel
        ax = axes[1, 1]
        ax.set_title(r'$x = %.3g$ m, $v = %.3g$ m/s' % (x_fix, v_fix))
        ax.set_xlabel(r'$\theta$ [deg]')
        ax.set_ylabel(r'$\omega$ [deg/s]')
        im = ax.imshow(z.T, origin='lower', extent=limits[(1, 3), :].ravel(), aspect=limits[1, 0] / limits[3, 0], cmap=HEAT_MAP, vmax=0.0)
        cbar = fig.colorbar(im, ax=ax, label=r'$v(\mu(x)) - v(x) + error$')

        ###################################################################################################

        #
        # z = true_dec_pos
        z = true_dec_pos - threshold_vel
        ax = axes[0, 2]
        ax.set_title(r'$\theta = %.3g$ deg, $\omega = %.3g$ deg/s' % (theta_fix, omega_fix))
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$v$ [m/s]')
        im = ax.imshow(z.T, origin='lower', extent=limits[(0, 2), :].ravel(), aspect=limits[0, 0] / limits[2, 0], cmap=HEAT_MAP, vmax=0.0)
        # cbar = fig.colorbar(im, ax=ax, label=r'$v(f(x)) - v(x)$')
        cbar = fig.colorbar(im, ax=ax, label=r'$v(f(x)) - v(x) + L_{\Delta v}\tau$')

        #
        # z = true_dec_vel
        z = true_dec_vel - threshold_vel
        ax = axes[1, 2]
        ax.set_title(r'$x = %.3g$ m, $v = %.3g$ m/s' % (x_fix, v_fix))
        ax.set_xlabel(r'$\theta$ [deg]')
        ax.set_ylabel(r'$\omega$ [deg/s]')
        im = ax.imshow(z.T, origin='lower', extent=limits[(1, 3), :].ravel(), aspect=limits[1, 0] / limits[3, 0], cmap=HEAT_MAP, vmax=0.0)
        # cbar = fig.colorbar(im, ax=ax, label=r'$v(f(x)) - v(x)$')
        cbar = fig.colorbar(im, ax=ax, label=r'$v(f(x)) - v(x) + L_{\Delta v}\tau$')

        ###################################################################################################

        #
        if Nmax is None:
            z = lyapunov._n[pos_set].reshape(lyapunov.discretization.num_points[(0, 2), ])
        else:
            z = np.clip(np.ceil(np.divide(threshold_pos, dec_pos, out=np.ones_like(dec_pos), where=dec_pos!=0)), 0, None)
        ax = axes[0, 3]
        ax.set_title(r'$\theta = %.3g$ deg, $\omega = %.3g$ deg/s' % (theta_fix, omega_fix))
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$v$ [m/s]')
        im = ax.imshow(z.T, origin='lower', extent=limits[(0, 2), :].ravel(), aspect=limits[0, 0] / limits[2, 0], cmap=LEVEL_MAP, vmin=1, vmax=Nmax)
        cbar = fig.colorbar(im, ax=ax, label=r'$N$') #, ticks=np.arange(1, Nmax + 1))

        #
        if Nmax is None:
            z = lyapunov._n[vel_set].reshape(lyapunov.discretization.num_points[(1, 3), ])
        else:
            z = np.clip(np.ceil(np.divide(threshold_vel, dec_vel, out=np.ones_like(dec_vel), where=dec_vel!=0)), 0, None)
        ax = axes[1, 3]
        ax.set_title(r'$x = %.3g$ m, $v = %.3g$ m/s' % (x_fix, v_fix))
        ax.set_xlabel(r'$\theta$ [deg]')
        ax.set_ylabel(r'$\omega$ [deg/s]')
        im = ax.imshow(z.T, origin='lower', extent=limits[(1, 3), :].ravel(), aspect=limits[1, 0] / limits[3, 0], cmap=LEVEL_MAP, vmin=1, vmax=Nmax)
        cbar = fig.colorbar(im, ax=ax, label=r'$N$') #, ticks=np.arange(1, Nmax + 1))

        ###################################################################################################

        plt.show()