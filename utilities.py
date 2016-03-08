from __future__ import division
import numpy as np
import scipy as sp


__all__ = ['combinations', 'line_search_bisection', 'compute_v_dot',
           'get_safe_set', 'find_max_levelset', 'lqr',
           'quadratic_lyapunov_function']


def combinations(arrays):
    """Return a single array with combinations of parameters.

    Parameters
    ----------
    arrays - list of np.array

    Returns
    -------
    array - np.array
        An array that contains all combinations of the input arrays
    """
    return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))


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
