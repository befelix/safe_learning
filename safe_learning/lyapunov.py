
from __future__ import absolute_import, division, print_function

import numpy as np
import GPy


__all__ = ['LyapunovFunction']


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


class FakeGP(object):
    """Fake GP for deterministic functions"""
    def __init__(self, function):
        self.function = function

    def predict_noiseless(self, X):
        mean = self.function(X)
        var = np.zeros_like(mean)
        return mean, var


class LyapunovFunction(object):

    def __init__(self, discretization, lyapunov_function,
                 dynamics_model, initial_set=None, beta=2):
        super(LyapunovFunction, self).__init__()

        self.discretization = discretization
        self.beta = beta

        if isinstance(dynamics_model, GPy.core.GP):
            self.dynamics_model = dynamics_model
        else:
            self.dynamics_model = FakeGP(dynamics_model)

        # Keep track of the safe sets
        self.initial_safe_set = np.asarray(initial_set, dtype=np.bool)
        self.safe_set = np.zeros(len(discretization), dtype=np.bool)
        self.v_dot_negative = np.zeros(len(discretization), dtype=np.bool)
        if initial_set is not None:
            self.safe_set[:] = self.initial_safe_set
        self.cmax = 0

        self.lyapunov_function = lyapunov_function
        self.V, self.dV = lyapunov_function(discretization)

    def v_dot_distribution(self, mean, variance):
        """
        Compute the distribution over V_dot, given the dynamics distribution.

        Parameters
        ----------
        dV: np.array
            The derivatives of the Lyapunov function at grid points
        mean: np.array
            gp mean of the dynamics (including prior dynamics as mean)
        variance: np.array
            gp var of the dynamics

        Returns
        -------
        mean - np.array
            The mean of V_dot at each grid point
        var - np.array
            The variance of V_dot at each grid point
        """
        # V_dot_mean = dV * mu
        # V_dot_var = sum_i(|dV_i| * var_i)
        # Should be dV.T var dV if we considered correlation
        # by considering correlations (predicting the sum term directly).
        return (np.sum(self.dV * mean, axis=1),
                np.sum(self.dV ** 2 * variance, axis=1))

    def max_safe_levelset(self, accuracy, interval=None):
        """
        Find maximum level set of V in S.


        Parameters
        ----------
        S: boolean array
            Elements are True if V_dot <= L tau
        V: np.array
            1d array with values of Lyapunov function.
        accuracy :float
            The accuracy up to which the level set is computed
        interval: list
            Interval within which the level set is search. Defaults
            to [0, max(V) + accuracy]

        Returns
        -------
        c - float
            The value of the maximum level set
        """

        def levelset_is_safe(c):
            """
            Return true if V(c) is subset of S

            Parameters
            ----------
            c: float
                The level set value

            Returns
            -------
            safe: boolean
            """
            # All points that have V<=c should be safe (have S=True)
            return np.all(self.v_dot_negative[self.V <= c])

        if interval is None:
            interval = [0, np.max(self.V) + accuracy]

        return line_search_bisection(levelset_is_safe,
                                     interval,
                                     accuracy)[0]

    def update_safe_set(self, threshold, accuracy, interval=None):
        """
        Compute the safe set

        Parameters
        ----------
        threshold: float
            The safety threshold, in the paper threshold = -L * tau

        Returns
        -------
        safe_set: ndarray
            The safe set.
        """
        # Get the distribution over the dynamics and V_dot
        mean, var = self.dynamics_model.predict_noiseless(self.discretization)
        V_dot_mean, V_dot_var = self.v_dot_distribution(mean, var)

        # Upper bound on V_dot
        V_dot_bound = V_dot_mean + self.beta * np.sqrt(V_dot_var)

        # Update the safe set
        self.v_dot_negative[:] = V_dot_bound < threshold

        # Make sure initial safe set is included
        if self.initial_safe_set is not None:
            self.v_dot_negative |= self.initial_safe_set

        self.cmax = self.max_safe_levelset(accuracy, interval)
        self.safe_set[:] = self.V <= self.cmax
