"""
This file defines the main functions needed to compute the region of
attraction of a general, nonlinear system that is modeled by a GP.

Author: Felix Berkenkamp, Learning & Adaptive Systems Group, ETH Zurich
        (GitHub: befelix)
"""


from __future__ import division, print_function

import numpy as np


__all__ = ['line_search_bisection', 'compute_v_dot_distribution',
           'compute_v_dot_upper_bound', 'get_safe_set', 'find_max_levelset']


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


def compute_v_dot_distribution(dV, mean, variance):
    """
    Compute the distribution over V_dot, given the gp dynamics model.

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
    return np.sum(dV * mean, axis=1), np.sum(dV**2 * variance, axis=1)


def compute_v_dot_upper_bound(dV, mean, variance=None, beta=2.):
    """
    Compute the safe set

    Parameters
    ----------
    dV: np.array
        The derivatives of the Lyapunov function at grid points
    mean: np.array
        mean of the dynamics (including prior dynamics as mean)
    variance: np.array
        variance of the dynamics
    beta: float
        The confidence interval for the GP-prediction

    Returns
    -------
    V_dot - np.array
        The beta-upper confidence bound on V_dot
    """
    if variance is None:
        variance = np.zeros_like(mean)
    # V_dot_mean = dV * mu
    # V_dot_var = sum_i(|dV_i| * var_i)
    # Should be dV.T var dV if we considered correlation
    # by considering correlations (predicting the sum term directly).
    mean, variance = compute_v_dot_distribution(dV, mean, variance)
    return mean + beta * np.sqrt(variance)


def get_safe_set(V_dot, threshold, S0=None):
    """
    Compute the safe set

    Parameters
    ----------
    V_dot: np.array
        V_dot at all grid points
    threshold: float
        The safety threshold, in the paper threshold = -L * tau
    S0: np.array
        The deterministic safe set
    """
    if S0 is None:
        return V_dot < threshold
    else:
        return np.logical_or(S0, V_dot < threshold)


def find_max_levelset(S, V, accuracy, interval=None):
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
        return np.all(S[V <= c])

    if interval is None:
        interval = [0, np.max(V) + accuracy]
    return line_search_bisection(levelset_is_safe,
                                 interval,
                                 accuracy)[0]
