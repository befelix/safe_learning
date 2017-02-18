"""Implements the Lyapunov functions and learning."""

from __future__ import absolute_import, division, print_function

import numpy as np

from .functions import UncertainFunction

__all__ = ['LyapunovContinuous', 'LyapunovDiscrete']


def line_search_bisection(f, bound, accuracy):
    """
    Maximize c so that constraint fulfilled.

    This algorithm assumes continuity of f; that is,
    there exists a fixed value c, such that f(x) is
    False for x < c and True otherwise. This holds true,
    for example, for the level sets that we consider.

    Parameters
    ----------
    f: callable
        A function that takes a scalar value and return True if
        the constraint is fulfilled, False otherwise.
    bound: iterable
        Interval within which to search
    accuracy: float
        The interval up to which the algorithm shall search

    Returns
    -------
    c: list
        The interval in which the optimum lies.
    """
    # Break if lower bound does not fulfill constraint
    if not f(bound[0]):
        return None

    # Break if bound was too small
    if f(bound[1]):
        bound[0] = bound[1]
        return bound

    # Improve bound until accuracy is achieved
    while bound[1] - bound[0] > accuracy:
        mean = (bound[0] + bound[1]) / 2

        if f(mean):
            bound[0] = mean
        else:
            bound[1] = mean

    return bound


class Lyapunov(object):
    """Base class for Lyapunov functions.

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
    epsilon : float
        The discretization constant.
    initial_set : ndarray, optional
        A boolean array of states that are known to be safe a priori.
    policy : ndarray, optional
        The control policy used at each state (Same number of rows as the
        discretization).
    """

    def __init__(self, discretization, lyapunov_function, dynamics,
                 epsilon, initial_set=None, policy=None):
        """Initialization, see `Lyapunov` for details."""
        super(Lyapunov, self).__init__()

        self.discretization = discretization
        if policy is None:
            self.policy = np.empty((len(self.discretization), 0),
                                   dtype=np.float)
        else:
            self.policy = policy

        # Keep track of the safe sets
        self.safe_set = np.zeros(len(discretization), dtype=np.bool)
        self.v_dot_negative = np.zeros(len(discretization), dtype=np.bool)
        self.initial_safe_set = initial_set
        if initial_set is not None:
            self.initial_safe_set = np.asarray(initial_set, dtype=np.bool)
            self.initial_safe_set = self.initial_safe_set.squeeze()
            self.safe_set[:] = self.initial_safe_set
        self.cmax = 0

        # Discretization constant
        self.epsilon = epsilon

        # Make sure dynamics are of standard framework
        self.dynamics = dynamics
        self.uncertain_dynamics = isinstance(dynamics, UncertainFunction)

        # Make sure Lyapunov fits into standard framework
        self.lyapunov_function = lyapunov_function

        # Lyapunov values
        self.V = self.lyapunov_function(self.discretization).squeeze()

    @property
    def is_discrete(self):
        """Whether the system is discrete-time."""
        return isinstance(self, LyapunovDiscrete)

    @property
    def is_continuous(self):
        """Whether the system is continuous-time."""
        return isinstance(self, LyapunovContinuous)

    def v_decrease_confidence(self, dynamics, error_bounds=None):
        """
        Compute confidence intervals for the decrease along Lyapunov function.

        Parameters
        ----------
        dynamics : np.array
            The dynamics evaluated at each point on the discretization.
        error_bounds : np.array, optional
            Point-wise error error_bounds for the dynamics. If None, the error
            is assumed to be zero.

        Returns
        -------
        mean : np.array
            The expected decrease in V at each grid point.
        error_bounds : np.array
            The error bounds for the decrease at each grid point. This is None
            if the error_bound is None.
        """
        raise NotImplementedError

    @property
    def threshold(self):
        """Return the safety threshold for the Lyapunov condition."""
        raise NotImplementedError

    def _levelset_is_safe(self, c):
        """
        Return true if V(c) is subset of S.

        Parameters
        ----------
        c : float
            The level set value

        Returns
        -------
        safe : boolean
        """
        # All points that have V<=c should be safe (have S=True)
        return np.all(self.v_dot_negative[self.V <= c])

    def max_safe_levelset(self, accuracy, interval=None):
        """Find maximum level set of V in S.

        Parameters
        ----------
        accuracy : float
            The accuracy up to which the level set is computed
        interval : list
            Interval within which the level set is search. Defaults
            to [0, max(V) + accuracy]

        Returns
        -------
        c : float
            The value of the maximum level set
        """
        if interval is None:
            interval = [0, np.max(self.V) + accuracy]

        bound = line_search_bisection(self._levelset_is_safe,
                                      interval,
                                      accuracy)

        if bound is None:
            return 0
        else:
            return bound[0]

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

        if self.uncertain_dynamics:
            v_dot, v_dot_error = self.v_decrease_confidence(
                self.discretization, *prediction)
            # Upper bound on V_dot
            v_dot_bound = v_dot + v_dot_error
        else:
            v_dot_bound, _ = self.v_decrease_confidence(self.discretization,
                                                        prediction)

        # Update the safe set
        v_dot_negative = v_dot_bound < self.threshold

        # Make sure initial safe set is included
        if include_initial and self.initial_safe_set is not None:
            v_dot_negative |= self.initial_safe_set

        return v_dot_negative

    def update_safe_set(self, accuracy, interval=None):
        """Compute the safe set.

        Parameters
        ----------
        accuracy : float
            The accuracy up to which the level set is computed
        interval : list
            Interval within which the level set is search. Defaults
            to [0, max(V) + accuracy]
        """
        self.v_dot_negative = self.safety_constraint(self.policy)
        self.cmax = self.max_safe_levelset(accuracy, interval)
        self.safe_set = self.V <= self.cmax


class LyapunovContinuous(Lyapunov):
    """A class for general Lyapunov functions.

    Parameters
    ----------
    discretization : ndarray
        A discrete grid on which to evaluate the Lyapunov function.
    lyapunov_function : instance of `DeterministicFunction`
        The lyapunov function. Can be called with states and returns the
        corresponding values of the Lyapunov function. For continuous-time
        systems, this function must include the derivative. One can also pass
        a tuple of callables (lyapunov_function, lyapunov_gradient).
    dynamics : a callable or an instance of `Function`
        The dynamics model. Can be either a deterministic function or something
        uncertain that includes error bounds.
    epsilon : float
        The discretization constant.
    initial_set : ndarray, optional
        A boolean array of states that are known to be safe a priori.
    """

    def __init__(self, discretization, lyapunov_function, dynamics, epsilon,
                 lipschitz, initial_set=None, policy=None):
        """Initialization, see `LyapunovFunction`."""
        super(LyapunovContinuous, self).__init__(discretization,
                                                 lyapunov_function,
                                                 dynamics,
                                                 epsilon,
                                                 initial_set=initial_set,
                                                 policy=policy)

        self.lipschitz = lipschitz
        self.dV = self.lyapunov_function.gradient(self.discretization)

    @property
    def threshold(self):
        """Return the safety threshold for the Lyapunov condition."""
        return -self.lipschitz * self.epsilon

    @staticmethod
    def lipschitz_constant(dynamics_bound, lipschitz_dynamics,
                           lipschitz_lyapunov, lipschitz_lyapunov_derivative):
        """Compute the Lipschitz constant of dot{V}.

        Note
        ----
        All of the following parameters can be either floats or ndarrays. If
        they are ndarrays, they should represent the local properties of the
        functions at the discretization points within an epsilon-ball.

        Parameters
        ----------
        dynamics_bound : float
            The largest absolute value that the dynamics can achieve.
        lipschitz_dynamics : float
            The Lipschitz constant of the dynamics.
        lipschitz_lyapunov : float
            The Lipschitz constant of the Lyapunov function.
        lipschitz_lyapunov_derivative : float
            The Lipschitz constant of the derivative of the Lyapunov function.
        """
        return (dynamics_bound * lipschitz_lyapunov_derivative
                + lipschitz_lyapunov * lipschitz_dynamics)

    def v_decrease_confidence(self, dynamics, error_bounds=None):
        """
        Compute confidence intervals for the decrease along Lyapunov function.

        Parameters
        ----------
        dynamics : np.array
            The dynamics evaluated at each point on the discretization.
        error_bounds : np.array
            Point-wise error error_bounds for the dynamics.

        Returns
        -------
        mean : np.array
            The expected decrease in V at each grid point.
        error_bounds : np.array
            The error bounds for the decrease at each grid point
        """
        # V_dot_mean = dV * mu
        # V_dot_var = sum_i(|dV_i| * var_i)
        mean = np.sum(self.dV * dynamics, axis=1)

        if error_bounds is None:
            error = 0
        else:
            error = np.sum(np.abs(self.dV) * error_bounds, axis=1)
        return mean, error


class LyapunovDiscrete(Lyapunov):
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
    initial_set : ndarray, optional
        A boolean array of states that are known to be safe a priori.
    policy : ndarray, optional
        The control policy used at each state (Same number of rows as the
        discretization).
    """

    def __init__(self, discretization, lyapunov_function, dynamics,
                 lipschitz_dynamics, lipschitz_lyapunov, epsilon,
                 initial_set=None, policy=None):
        """Initialization, see `LyapunovFunction`."""
        super(LyapunovDiscrete, self).__init__(discretization,
                                               lyapunov_function,
                                               dynamics,
                                               epsilon,
                                               initial_set=initial_set,
                                               policy=policy)

        self.lipschitz_dynamics = lipschitz_dynamics
        self.lipschitz_lyapunov = lipschitz_lyapunov

    @property
    def threshold(self):
        """Return the safety threshold for the Lyapunov condition."""
        lv, lf = self.lipschitz_lyapunov, self.lipschitz_dynamics
        return -lv * (1. + lf) * self.epsilon

    def v_decrease_confidence(self, states, next_states, error_bounds=None):
        """
        Compute confidence intervals for the decrease along Lyapunov function.

        Parameters
        ----------
        states : np.array
            The states at which to start (could be equal to discretization).
        next_states : np.array
            The dynamics evaluated at each point on the discretization.
        error_bounds : np.array
            Point-wise error error_bounds for the dynamics. Have to be strictly
            positive.

        Returns
        -------
        mean : np.array
            The expected decrease in V at each grid point.
        error_bounds : np.array
            The error bounds for the decrease at each grid point
        """
        next_states = (self.lyapunov_function(next_states)[:, 0]
                       - self.lyapunov_function(states)[:, 0])

        if error_bounds is None:
            bound = 0
        else:
            # Compute the error bound
            bound = self.lipschitz_lyapunov * np.sum(error_bounds, axis=1)

        return next_states, bound

