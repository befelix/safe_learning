from __future__ import division, print_function

import sys
import os
import importlib
import numpy as np
import tensorflow as tf
from scipy import signal
from safe_learning import DeterministicFunction
from safe_learning import config
from safe_learning.utilities import concatenate_inputs
if sys.version_info.major == 2:
    import imp


__all__ = ['uniform_state_sampler', 'compute_closedloop_response',
           'import_from_directory', 'InvertedPendulum']


np_dtype = config.np_dtype
tf_dtype = config.dtype


def sample_ellipsoid(covariance, level, num_samples):
    # Sample from unit sphere
    dim = covariance.shape[0]
    N = int(num_samples)
    mean = np.zeros(dim)
    Y = np.random.multivariate_normal(mean, covariance, N)
    Y = Y / np.linalg.norm(Y, ord=2, axis=1, keepdims=True)

    # Scale to spheres of different radii
    r = np.random.uniform(0., level, N).reshape((-1, 1))
    Y = Y*r

    # Transform to ellipsoid samples
    U = np.linalg.cholesky(covariance).T   # P = L.dot(L.T), U = L.T
    X = scipy.linalg.solve_triangular(U, Y.T, lower=False).T

    return X


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


def get_max_parameter_change(old_params, new_params):
    """Get the maximum absolute parameter change value.

    Parameters
    ----------
    old_params : list
        The old parameters as a list of ndarrays, typically from
        session.run(var_list)
    new_params : list
        The old parameters as a list of ndarrays, typically from
        session.run(var_list)

    Returns
    -------
    max_change : float
        The maximum absolute difference between all elements in old_params and
        new_params
    """
    max_change = 0.0
    for old, new in zip(old_params, new_params):
        candidate = np.max(np.abs(new - old))
        if candidate > max_change:
            max_change = candidate
    return max_change


def compute_closedloop_response(dynamics, policy, steps, dt, reference='zero', const=1.0, ic=None):
    """
    """
    state_dim = policy.input_dim
    action_dim = policy.output_dim
    
    state_dim = 4
    action_dim = 1
    
    if reference=='impulse':
        r = np.zeros((steps + 1, action_dim))
        r[0, :] = (1 / dt) * np.ones((1, action_dim))
    elif reference=='step':
        r = const*np.ones((steps + 1, action_dim))
    elif reference=='zero':
        r = np.zeros((steps + 1, action_dim))

    times = dt*np.arange(steps + 1, dtype=np_dtype).reshape((-1, 1))
    states = np.zeros((steps + 1, state_dim), dtype=np_dtype)
    actions = np.zeros((steps + 1, action_dim), dtype=np_dtype)
    if ic is not None:
        states[0, :] = np.asarray(ic, dtype=np_dtype).reshape((1, state_dim))

    session = tf.get_default_session()    
    with tf.name_scope('compute_closedloop_response'):
        current_ref = tf.placeholder(tf_dtype, shape=[1, action_dim])
        current_state = tf.placeholder(tf_dtype, shape=[1, state_dim])
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

    def __init__(self, pendulum_mass, cart_mass, length,
                 dt=0.01, normalization=None):
        super(CartPole, self).__init__(name='CartPole')
        self.pendulum_mass = pendulum_mass
        self.cart_mass = cart_mass
        self.length = length
        self.friction = 0.0  # TODO
        self.dt = dt
        self.gravity = 9.81
        self.state_dim = 4
        self.action_dim = 1
        self.norm = normalization
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
        g = self.gravity

        A = np.array([[0, 0,                     1, 0],
                      [0, 0,                     0, 1],
                      [0, g * m / M,             0, 0],
                      [0, g * (m + M) / (L * M), 0, 0]],
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
        # TODO
        raise NotImplementedError('TODO.')


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
        super(InvertedPendulum, self).__init__()
        self.mass = mass
        self.length = length
        self.gravity = 9.81
        self.friction = friction
        self.dt = dt

        self.normalization = normalization
        if normalization is not None:
            self.normalization = [arr.astype(config.np_dtype) for
                                  arr in normalization]
            self.inv_norm = [arr ** -1 for arr in self.normalization]

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
        """Denormalize states and actions."""
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
