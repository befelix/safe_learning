"""Implements the Lyapunov functions and learning."""

from __future__ import absolute_import, division, print_function

from .functions import UncertainFunction, DeterministicFunction, Function

import numpy as np
from collections import Sequence

__all__ = ['LyapunovContinuous']


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
    """Baseclass for Lyapunov functions.

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
    """

    def __init__(self, discretization, lyapunov_function, dynamics,
                 epsilon, initial_set=None):
        """Initialization, see `Lyapunov` for details."""
        super(Lyapunov, self).__init__()

        self.discretization = discretization

        # Keep track of the safe sets
        self.initial_safe_set = np.asarray(initial_set, dtype=np.bool)
        self.safe_set = np.zeros(len(discretization), dtype=np.bool)
        self.v_dot_negative = np.zeros(len(discretization), dtype=np.bool)
        if initial_set is not None:
            self.safe_set[:] = self.initial_safe_set
        self.cmax = 0

        # Discretization constant
        self.epsilon = epsilon

        # Make sure dynamics are of standard framework
        if isinstance(dynamics, Function):
            self.dynamics = dynamics
        elif isinstance(Sequence, dynamics):
            self.dynamics = DeterministicFunction.from_callable(*dynamics)
        else:
            self.dynamics = DeterministicFunction.from_callable(dynamics)
        self.uncertain_dynamics = isinstance(dynamics, UncertainFunction)

        # Make sure Lyapunov fits into standard framework
        if isinstance(lyapunov_function, DeterministicFunction):
            self.lyapunov_function = lyapunov_function
        else:
            self.lyapunov_function = DeterministicFunction.from_callable(
                lyapunov_function)

        # Lyapunov values
        self.V = self.lyapunov_function.evaluate(self.discretization)

    def v_decrease_confidence(self, dynamics, error_bounds):
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
        raise NotImplementedError

    @property
    def threshold(self):
        """Return the safety threshold for the Lyapunov condition."""
        raise NotImplementedError

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
        def levelset_is_safe(c):
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

        if interval is None:
            interval = [0, np.max(self.V) + accuracy]

        return line_search_bisection(levelset_is_safe,
                                     interval,
                                     accuracy)[0]

    def update_safe_set(self, accuracy, interval=None):
        """Compute the safe set.

        Parameters
        ----------
        accuracy : float
            The accuracy up to which the level set is computed
        interval : list
            Interval within which the level set is search. Defaults
            to [0, max(V) + accuracy]

        Returns
        -------
        safe_set : ndarray
            The safe set.
        """
        prediction = self.dynamics.evaluate(self.discretization)

        if self.uncertain_dynamics:
            v_dot, v_dot_error = self.v_decrease_confidence(*prediction)
            # Upper bound on V_dot
            v_dot_bound = v_dot + v_dot_error
        else:
            v_dot_bound = prediction

        # Update the safe set
        self.v_dot_negative[:] = v_dot_bound < self.threshold

        # Make sure initial safe set is included
        if self.initial_safe_set is not None:
            self.v_dot_negative |= self.initial_safe_set

        self.cmax = self.max_safe_levelset(accuracy, interval)
        self.safe_set[:] = self.V <= self.cmax


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
                 lipschitz, initial_set=None):
        """Initialization, see `LyapunovFunction`."""
        super(LyapunovContinuous, self).__init__(discretization,
                                                 lyapunov_function,
                                                 dynamics,
                                                 epsilon,
                                                 initial_set=initial_set)

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

    def v_decrease_confidence(self, dynamics, error_bounds):
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
        return (np.sum(self.dV * dynamics, axis=1),
                np.sum(np.abs(self.dV) * error_bounds, axis=1))


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
    """

    def __init__(self, discretization, lyapunov_function, dynamics,
                 lipschitz_dynamics, lipschitz_lyapunov, epsilon,
                 initial_set=None):
        """Initialization, see `LyapunovFunction`."""
        super(LyapunovDiscrete, self).__init__(discretization,
                                               lyapunov_function,
                                               dynamics,
                                               epsilon,
                                               initial_set=initial_set)

        self.lipschitz_dynamics = lipschitz_dynamics
        self.lipschitz_lyapunov = lipschitz_lyapunov

        self.V, self.dV = lyapunov_function(discretization)

    @property
    def threshold(self):
        """Return the safety threshold for the Lyapunov condition."""
        lv, lf = self.lipschitz_lyapunov, self.lipschitz_dynamics
        return lv * (1. + lf) * self.epsilon

    def v_decrease_confidence(self, dynamics, error_bounds):
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
        dynamics = self.lyapunov_function(dynamics) - self.V

        # Condition checks if lipschitz constants are given per dimension
        if (self.lipschitz_dynamics.ndim == 2
                and self.lipschitz_dynamics.shape[1] > 1):
            bound = np.sum(self.lipschitz_dynamics * error_bounds, axis=1)
        else:
            bound = self.lipschitz_dynamics * np.sum(error_bounds, axis=1)

        return dynamics, bound
