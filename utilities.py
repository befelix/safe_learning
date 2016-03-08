from __future__ import division

from collections import Sequence

import numpy as np
import scipy as sp


__all__ = ['combinations', 'linearly_spaced_combinations',
           'line_search_bisection', 'compute_v_dot', 'get_safe_set',
           'find_max_levelset', 'lqr', 'quadratic_lyapunov_function',
           'sample_gp_function']


def combinations(arrays):
    """Return a single array with combinations of parameters.

    Parameters
    ----------
    arrays: list of np.array

    Returns
    -------
    array - np.array
        An array that contains all combinations of the input arrays
    """
    return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))


def linearly_spaced_combinations(bounds, num_samples):
    """
    Return 2-D array with all linearly spaced combinations with the bounds.

    Parameters
    ----------
    bounds: sequence of tuples
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_samples: integer or array_likem
        Number of samples to use for every dimension. Can be a constant if
        the same number should be used for all, or an array to fine-tune
        precision. Total number of data points is num_samples ** len(bounds).

    Returns
    -------
    combinations: 2-d array
        A 2-d arrray. If d = len(bounds) and l = prod(num_samples) then it
        is of size l x d, that is, every row contains one combination of
        inputs.
    """
    num_vars = len(bounds)
    if not isinstance(num_samples, Sequence):
        num_samples = [num_samples] * num_vars

    # Create linearly spaced test inputs
    inputs = [np.linspace(b[0], b[1], n) for b, n in zip(bounds,
                                                         num_samples)]

    # Convert to 2-D array
    return combinations(inputs)


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
    
    if f(bound[1]):
        return bound[1]
    
    while bound[1] - bound[0] > accuracy:
        mean = (bound[0] + bound[1]) / 2
        
        if f(mean):
            bound[0] = mean
        else:
            bound[1] = mean
    
    return bound
    
    
def compute_v_dot(dV, mean, var=None, beta=2.):
    """
    Compute the safe set
    
    Parameters
    ----------
    dV: np.array
        The derivatives of the Lyapunov function at grid points
    mean: np.array
        gp mean of the dynamics (including prior dynamics as mean)
    var: np.array
        gp var of the dynamics
    beta: float
        The confidence interval for the GP-prediction
        
    Returns
    -------
    V_dot - np.array
        The beta-upper confidence bound on V_dot 
    """    
    # V_dot_mean = dV * mu
    # V_dot_var = sum_i(|dV_i| * var_i)
    # Should be dV.T var dV if we considered correlation
    # by considering correlations (predicting the sum term directly).
    if var is None:
        return np.sum(dV * mean, axis=1)
    else:
        return (np.sum(dV * mean, axis=1) +
                beta * np.sqrt(np.sum(dV**2 * var, axis=1)) )
    
    
def get_safe_set(V_dot, threshold, S0=None):
    """
    Compute the safe set
    
    Parameters
    ----------
    V_dot: np.array
        V_dot at all grid points
    threshold: float
        The safety threshold, in the paper threshold = tau * L
    S0: np.array
        The deterministic safe set
    """    
    if S0 is None:
        return V_dot < -threshold
    else:
        return np.logical_or(S0, V_dot < -threshold)
        
        
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
                                 
                                 
def lqr(A, B, Q, R):
    """
    Compute the continuous time LQR-controller. 
    
    Parameters
    ----------
    A - np.array
    B - np.array
    Q - np.array
    R - np.array
     
    Returns
    -------
    K - np.array
        Controller matrix
    P - np.array
        Cost to go matrix
    """
    P = sp.linalg.solve_continuous_are(A, B, Q, R)
     
    # LQR gain
    K = np.linalg.solve(R, B.T.dot(P))

    return K, P
    
    
def quadratic_lyapunov_function(x, P):
    """
    Compute V(x) and dV(x)/dx for a quadratic Lyapunov function
    
    V(x) = x.T P x
    dV(x)/dx = 2 x.T P
    
    Equivalent, but slower implementation:
    np.array([ xi.dot(p.dot(xi.T)) for xi in x])
    
    Parameters
    ----------
    x - np.array
        2d array that has a state vector xi on each row
    P - np.array
        2d cost matrix for lyapunov function

    Returns
    -------
    V - np.array
        1d array with V(x)
    dV - np.array
        2d array with dV(x)/dx on each row
    """
    return np.sum(x.dot(P) * x, axis=1), 2 * x.dot(P)


def sample_gp_function(kernel, bounds, num_samples, noise_var,
                       interpolation='linear'):
    """
    Sample a function from a gp with corresponding kernel within its bounds.

    Parameters
    ----------
    kernel: instance of GPy.kern.*
    bounds: list of tuples
        [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_samples: int or list
        If integer draws the corresponding number of samples in all
        dimensions and test all possible input combinations. If a list then
        the list entries correspond to the number of linearly spaced samples of
        the corresponding input
    noise_var: float
        Variance of the observation noise of the GP function
    interpolation: string
        If 'linear' interpolate linearly between samples, if 'kernel' use the
        corresponding mean RKHS-function of the GP.

    Returns
    -------
    function: object
        function(x, noise=True)
        A function that takes as inputs new locations x to be evaluated and
        returns the corresponding noisy function values. If noise=False is
        set the true function values are returned (useful for plotting).
    """
    inputs = linearly_spaced_combinations(bounds, num_samples)
    cov = kernel.K(inputs) + np.eye(inputs.shape[0]) * 1e-6
    output = np.random.multivariate_normal(np.zeros(inputs.shape[0]),
                                           cov)

    if interpolation == 'linear':

        def evaluate_gp_function_linear(x, noise=True):
            """
            Evaluate the GP sample function with linear interpolation.

            Parameters
            ----------
            x: np.array
                2D array with inputs
            noise: bool
                Whether to include prediction noise
            """
            x = np.atleast_2d(x)
            y = sp.interpolate.griddata(inputs, output, x, method='linear')
            y = np.atleast_2d(y)
            if noise:
                y += np.sqrt(noise_var) * np.random.randn(x.shape[0], 1)
            return y
        return evaluate_gp_function_linear
    elif interpolation == 'kernel':
        cho_factor = sp.linalg.cho_factor(cov)
        alpha = sp.linalg.cho_solve(cho_factor, output)

        def evaluate_gp_function_kernel(x, noise=True):
            """
            Evaluate the GP sample function with kernel interpolation.

            Parameters
            ----------
            x: np.array
                2D array with inputs
            noise: bool
                Whether to include prediction noise
            """
            x = np.atleast_2d(x)
            y = kernel.K(x, inputs).dot(alpha)
            if noise:
                y += np.sqrt(noise_var) * np.random.randn(x.shape[0], 1)
            return y
        return evaluate_gp_function_kernel
