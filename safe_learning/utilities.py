"""
This file defines utilities needed for the experiments, such as creating
parameter grids, computing LQR controllers, Lyapunov functions, sample
functions of Gaussian processes, and plotting ellipses.

Author: Felix Berkenkamp, Learning & Adaptive Systems Group, ETH Zurich
        (GitHub: befelix)
"""


from __future__ import division, print_function

from collections import Sequence

import numpy as np
import scipy as sp
import scipy.linalg
import scipy.interpolate


__all__ = ['combinations', 'linearly_spaced_combinations',
           'lqr', 'quadratic_lyapunov_function', 'sample_gp_function',
           'ellipse_bounds']


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
    x = np.asarray(x)
    return np.sum(x.dot(P) * x, axis=1), 2 * x.dot(P)


def sample_gp_function(kernel, bounds, num_samples, noise_var,
                       interpolation='linear', mean_function=None):
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
    mean_function: callable
        Mean of the sample function

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
            if mean_function is not None:
                y += mean_function(x)
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
            y = y[:, None]
            if mean_function is not None:
                y += mean_function(x)
            if noise:
                y += np.sqrt(noise_var) * np.random.randn(x.shape[0], 1)
            return y
        return evaluate_gp_function_kernel


def ellipse_bounds(P, level, n=100):
    """Compute the bounds of a 2D ellipse.

    The levelset of the ellipsoid is given by
    level = x' P x. Given the coordinates of the first
    dimension, this function computes the corresponding
    lower and upper values of the second dimension and
    removes any values of x0 that are outside of the ellipse.

    Parameters
    ----------
    P: np.array
        The matrix of the ellipsoid
    level: float
        The value of the levelset
    n: int
        Number of data points

    Returns
    -------
    x - np.array
        1D array of x positions of the ellipse
    yu - np.array
        The upper bound of the ellipse
    yl - np.array
        The lower bound of the ellipse

    Notes
    -----
    This can be used as
    ```plt.fill_between(*ellipse_bounds(P, level))```
    """
    # Round up to multiple of 2
    n += n % 2

    # Principal axes of ellipsoid
    eigval, eigvec = np.linalg.eig(P)
    eigvec *= np.sqrt(level / eigval)

    # set zero angle at maximum x
    angle = np.linspace(0, 2 * np.pi, n)[:, None]
    angle += np.arctan(eigvec[0, 1] / eigvec[0, 0])

    # Compute positions
    pos = np.cos(angle) * eigvec[:, 0] + np.sin(angle) * eigvec[:, 1]
    n /= 2

    # Return x-position (symmetric) and upper/lower bounds
    return pos[:n, 0], pos[:n, 1], pos[:n-1:-1, 1]
