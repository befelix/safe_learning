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
from safe_learning.utilities import concatenate_inputs, get_storage, set_storage
if sys.version_info.major == 2:
    import imp

from collections import OrderedDict
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import colors
import time


__all__ = ['constrained_batch_sampler', 'compute_closedloop_response',
           'import_from_directory', 'InvertedPendulum', 'CartPole']


np_dtype = config.np_dtype
tf_dtype = config.dtype
_STORAGE = {}
            

def debug(lyapunov, true_dynamics, state_norm, do_print=False, newly_safe_only=True, plot=None, 
          old_safe_set=None, fixed_state=(0., 0., 0., 0.)):
    
    storage = get_storage(_STORAGE, index=lyapunov)
    if storage is None:
        tf_states = tf.placeholder(tf_dtype, shape=[None, lyapunov.discretization.ndim], name='states')
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
        print('Dynamics Lipschitz constant (L_f*(L_pi + 1)): {}'.format(lyapunov.lipschitz_dynamics(0).eval()))
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
            values, mean_decrease, error, threshold = session.run([tf_values, tf_mean_decrease, tf_error, tf_threshold],
                                                                  feed_dict)
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
        feed_dict[tf_states] = lyapunov.discretization.all_points
        threshold, mean_decrease, error, true_decrease = session.run([tf_threshold, 
                                                                      tf_mean_decrease,
                                                                      tf_error, 
                                                                      tf_true_decrease],
                                                                     feed_dict)
        threshold = threshold.reshape(lyapunov.discretization.num_points)
        mean_decrease = mean_decrease.reshape(lyapunov.discretization.num_points)
        error = error.reshape(lyapunov.discretization.num_points)
        true_decrease = true_decrease.reshape(lyapunov.discretization.num_points)
        decrease = mean_decrease + error
              
        safe_set = lyapunov.safe_set
        # if old_safe_set is not None:
        #     safe_set = safe_set.astype(np_dtype) + old_safe_set.astype(np_dtype)
        safe_set = safe_set.reshape(lyapunov.discretization.num_points).astype(np_dtype)
            
        if state_norm is not None:
            theta_max, omega_max = state_norm
            scale = np.array([np.rad2deg(theta_max), np.rad2deg(omega_max)]).reshape((-1, 1))
            limits = scale * lyapunov.discretization.limits
        else:
            limits = lyapunov.discretization.limits

        # Figure
        fig, ax = plt.subplots(2, 3, figsize=(17, 12), dpi=100)
        fig.subplots_adjust(wspace=0.5, hspace=0.1)
        
        cmap = colors.ListedColormap(['indigo', 'yellow', 'orange', 'red'])
        bounds = [0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        z = (decrease < threshold) + 2 * safe_set
        ax[0, 0].set_xlabel(r'$\theta$ [deg]')
        ax[0, 0].set_ylabel(r'$\omega$ [deg/s]')
        im = ax[0, 0].imshow(z.T,
                          origin='lower',
                          extent=limits.ravel(),
                          aspect=limits[0, 0] / limits[1, 0],
                          cmap=cmap,
                          norm=norm)        
        cbar = fig.colorbar(im, ax=ax[0, 0], boundaries=bounds, ticks=[0.5, 1.5, 2.5, 3.5])
        cbar.ax.set_yticklabels(['unsafe\n$\Delta v \geq 0$', 'unsafe\n$\Delta v < 0$', 
                                 'safe\n$\Delta v \geq 0$', 'safe\n$\Delta v < 0$'])
        if isinstance(lyapunov.dynamics, UncertainFunction):
            if state_norm is not None:
                theta_max, omega_max = state_norm
                scale = np.array([np.rad2deg(theta_max), np.rad2deg(omega_max)]).ravel()
                X = scale * lyapunov.dynamics.functions[0].X[:, :2]
            else:
                X = lyapunov.dynamics.functions[0].X[:, :2]
            ax[0, 0].plot(X[:, 0], X[:, 1], 'kx')
        
        cmap = plt.get_cmap('viridis')
        cmap.set_over('gold')
        cmap.set_under('indigo')

        z = - decrease
        ax[0, 1].set_xlabel(r'$\theta$ [deg]')
        ax[0, 1].set_ylabel(r'$\omega$ [deg/s]')
        im = ax[0, 1].imshow(z.T,
                          origin='lower',
                          extent=limits.ravel(),
                          aspect=limits[0, 0] / limits[1, 0],
                          cmap=cmap,
                          vmin=0.0,
                          vmax=1.0)
        fig.colorbar(im, ax=ax[0, 1], label=r'$-(v(\mu_\pi(x)) - v(x) + error)$')
        
        z = - (mean_decrease - threshold)
        ax[0, 2].set_xlabel(r'$\theta$ [deg]')
        ax[0, 2].set_ylabel(r'$\omega$ [deg/s]')
        im = ax[0, 2].imshow(z.T,
                          origin='lower',
                          extent=limits.ravel(),
                          aspect=limits[0, 0] / limits[1, 0],
                          cmap=cmap,
                          vmin=0.0,
                          vmax=1.0)
        fig.colorbar(im, ax=ax[0, 2], label=r'$-(v(\mu_\pi(x)) - v(x) + L_{\Delta v}\tau)$')
        
        z = lyapunov._N.reshape(lyapunov.discretization.num_points) * safe_set
        Nmax = 10
        ax[1, 0].set_xlabel(r'$\theta$ [deg]')
        ax[1, 0].set_ylabel(r'$\omega$ [deg/s]')
        im = ax[1, 0].imshow(z.T,
                          origin='lower',
                          extent=limits.ravel(),
                          aspect=limits[0, 0] / limits[1, 0],
                          cmap=cmap,
                          vmin=1,
                          vmax=Nmax)
        fig.colorbar(im, ax=ax[1, 0], label=r'$N$', ticks=np.arange(1, Nmax + 1))
        
        z = - true_decrease
        ax[1, 1].set_xlabel(r'$\theta$ [deg]')
        ax[1, 1].set_ylabel(r'$\omega$ [deg/s]')
        im = ax[1, 1].imshow(z.T,
                          origin='lower',
                          extent=limits.ravel(),
                          aspect=limits[0, 0] / limits[1, 0],
                          cmap=cmap,
                          vmin=0.0,
                          vmax=1.0)
        fig.colorbar(im, ax=ax[1, 1], label=r'$-(v(f(x)) - v(x))$')
        
        z = - (true_decrease - threshold)
        ax[1, 2].set_xlabel(r'$\theta$ [deg]')
        ax[1, 2].set_ylabel(r'$\omega$ [deg/s]')
        im = ax[1, 2].imshow(z.T,
                          origin='lower',
                          extent=limits.ravel(),
                          aspect=limits[0, 0] / limits[1, 0],
                          cmap=cmap,
                          vmin=0.0,
                          vmax=1.0)
        fig.colorbar(im, ax=ax[1, 2], label=r'$-(v(f(x)) - v(x) + L_{\Delta v}\tau)$')
        
        plt.show()

    elif plot=='cartpole':
        safe_set = lyapunov.safe_set
        fixed_state = np.asarray(fixed_state, dtype=np_dtype)
        for i in range(4):
            dist = np.square(lyapunov.discretization.discrete_points[i] - fixed_state[i])
            idx = np.argmin(dist)
            fixed_state[i] = lyapunov.discretization.discrete_points[i][idx]
        x_fix, theta_fix, v_fix, omega_fix = fixed_state
     
        pos_set = np.logical_and(lyapunov.discretization.all_points[:, 1] == theta_fix, 
                                 lyapunov.discretization.all_points[:, 3] == omega_fix)
        vel_set = np.logical_and(lyapunov.discretization.all_points[:, 0] == x_fix,
                                 lyapunov.discretization.all_points[:, 2] == v_fix)
        
        
        feed_dict[tf_states] = lyapunov.discretization.all_points[pos_set, :]
        threshold_pos, dec_pos, true_dec_pos = session.run([tf_threshold, 
                                                            tf_mean_decrease + tf_error, 
                                                            tf_true_decrease], 
                                                           feed_dict)
        threshold_pos = threshold_pos.reshape(lyapunov.discretization.num_points[:2])
        dec_pos = dec_pos.reshape(lyapunov.discretization.num_points[:2])
        true_dec_pos = true_dec_pos.reshape(lyapunov.discretization.num_points[:2])
        safe_pos = safe_set[pos_set].reshape(lyapunov.discretization.num_points[:2]).astype(np_dtype)        
        
        feed_dict[tf_states] = lyapunov.discretization.all_points[vel_set, :]
        threshold_vel, dec_vel, true_dec_vel = session.run([tf_threshold, 
                                                            tf_mean_decrease + tf_error, 
                                                            tf_true_decrease],
                                                           feed_dict)
        threshold_vel = threshold_vel.reshape(lyapunov.discretization.num_points[2:])
        dec_vel = dec_vel.reshape(lyapunov.discretization.num_points[2:])
        true_dec_vel = true_dec_vel.reshape(lyapunov.discretization.num_points[2:])
        safe_vel = safe_set[vel_set].reshape(lyapunov.discretization.num_points[2:]).astype(np_dtype)        
                
        if state_norm is not None:
            x_max, theta_max, v_max, omega_max = state_norm
            scale = np.array([x_max, np.rad2deg(theta_max), 
                              v_max, np.rad2deg(omega_max)]).reshape((-1, 1))
            true_limits = scale * lyapunov.discretization.limits
            x_fix, theta_fix, v_fix, omega_fix = fixed_state * scale.ravel()
        else:
            true_limits = lyapunov.discretization.limits
            
        # Figure
        fig, ax = plt.subplots(4, 2, figsize=(12, 20), dpi=100)
        fig.subplots_adjust(wspace=0.4, hspace=0.1)
        
        ###################################################################################################
        
        # Colormap
        cmap = colors.ListedColormap(['indigo', 'yellow', 'orange', 'red'])
        bounds = [0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        # Safe positions set, with fixed velocities
        z = (dec_pos < threshold_pos) + 2 * safe_pos
        ax[0, 0].set_title(r'$\theta = %.3g$ deg, $\omega = %.3g$ deg/s' % (theta_fix, omega_fix))
        ax[0, 0].set_xlabel(r'$x$ [m]')
        ax[0, 0].set_ylabel(r'$v$ [m/s]')
        im = ax[0, 0].imshow(z.T,
                             origin='lower',
                             extent=true_limits[(0, 2), :].ravel(), 
                             aspect=true_limits[0, 0] / true_limits[2, 0], 
                             cmap=cmap,
                             norm=norm)
        cbar = fig.colorbar(im, ax=ax[0, 0], boundaries=bounds, ticks=[0.5, 1.5, 2.5, 3.5])
        cbar.ax.set_yticklabels(['unsafe\n$\Delta v \geq 0$', 'unsafe\n$\Delta v < 0$', 
                                 'safe\n$\Delta v \geq 0$', 'safe\n$\Delta v < 0$'])

        # Safe velocities set, with fixed positions
        z = (dec_vel < threshold_vel) + 2 * safe_vel
        ax[0, 1].set_title(r'$x = %.3g$ m, $v = %.3g$ m/s' % (x_fix, v_fix))
        ax[0, 1].set_xlabel(r'$\theta$ [deg]')
        ax[0, 1].set_ylabel(r'$\omega$ [deg/s]')
        im = ax[0, 1].imshow(z.T, 
                             origin='lower', 
                             extent=true_limits[(1, 3), :].ravel(), 
                             aspect=true_limits[1, 0] / true_limits[3, 0], 
                             cmap=cmap,
                             norm=norm)
        cbar = fig.colorbar(im, ax=ax[0, 1], boundaries=bounds, ticks=[0.5, 1.5, 2.5, 3.5]) 
        cbar.ax.set_yticklabels(['unsafe\n$\Delta v \geq 0$', 'unsafe\n$\Delta v < 0$', 
                                 'safe\n$\Delta v \geq 0$', 'safe\n$\Delta v < 0$'])
        
        ###################################################################################################
        
        cmap = plt.get_cmap('viridis')
        cmap.set_under('indigo')
        cmap.set_over('gold')
        
        # Safe positions set, with fixed velocities
        z = - dec_pos
        ax[1, 0].set_title(r'$\theta = %.3g$ deg, $\omega = %.3g$ deg/s' % (theta_fix, omega_fix))
        ax[1, 0].set_xlabel(r'$x$ [m]')
        ax[1, 0].set_ylabel(r'$v$ [m/s]')
        im = ax[1, 0].imshow(z.T,
                             origin='lower',
                             extent=true_limits[(0, 2), :].ravel(), 
                             aspect=true_limits[0, 0] / true_limits[2, 0], 
                             cmap=cmap,
                             vmin=0.0,
                             vmax=0.25)
        cbar = fig.colorbar(im, ax=ax[1, 0], label=r'$-(v(\mu(x)) - v(x) + error)$')

        # Safe velocities set, with fixed positions
        z = - dec_vel
        ax[1, 1].set_title(r'$x = %.3g$ m, $v = %.3g$ m/s' % (x_fix, v_fix))
        ax[1, 1].set_xlabel(r'$\theta$ [deg]')
        ax[1, 1].set_ylabel(r'$\omega$ [deg/s]')
        im = ax[1, 1].imshow(z.T, 
                             origin='lower', 
                             extent=true_limits[(1, 3), :].ravel(), 
                             aspect=true_limits[1, 0] / true_limits[3, 0], 
                             cmap=cmap,
                             vmin=0.0,
                             vmax=0.25)
        cbar = fig.colorbar(im, ax=ax[1, 1], label=r'$-(v(\mu(x)) - v(x) + error)$')
        
        ###################################################################################################
        
        # Safe positions set, with fixed velocities
        # z = - true_dec_pos
        z = - (true_dec_pos - threshold_vel)
        ax[2, 0].set_title(r'$\theta = %.3g$ deg, $\omega = %.3g$ deg/s' % (theta_fix, omega_fix))
        ax[2, 0].set_xlabel(r'$x$ [m]')
        ax[2, 0].set_ylabel(r'$v$ [m/s]')
        im = ax[2, 0].imshow(z.T,
                             origin='lower',
                             extent=true_limits[(0, 2), :].ravel(), 
                             aspect=true_limits[0, 0] / true_limits[2, 0], 
                             cmap=cmap,
                             vmin=0.0,
                             vmax=0.25)
        cbar = fig.colorbar(im, ax=ax[2, 0], label=r'$-(v(f(x)) - v(x) + L_{\Delta v}\tau)$')
        
        # Safe velocities set, with fixed positions
        # z = - true_dec_vel
        z = - (true_dec_vel - threshold_vel)
        ax[2, 1].set_title(r'$x = %.3g$ m, $v = %.3g$ m/s' % (x_fix, v_fix))
        ax[2, 1].set_xlabel(r'$\theta$ [deg]')
        ax[2, 1].set_ylabel(r'$\omega$ [deg/s]')
        im = ax[2, 1].imshow(z.T, 
                             origin='lower', 
                             extent=true_limits[(1, 3), :].ravel(), 
                             aspect=true_limits[1, 0] / true_limits[3, 0], 
                             cmap=cmap,                   
                             vmin=0.0,
                             vmax=0.25)
        cbar = fig.colorbar(im, ax=ax[2, 1], label=r'$-(v(f(x)) - v(x) + L_{\Delta v}\tau)$')
        
        ###################################################################################################
    
        Nmax = 5000
        
        # Safe positions set, with fixed velocities
        # z = lyapunov._N[pos_set].reshape(lyapunov.discretization.num_points[:2])
        z = np.clip(np.ceil(threshold_pos / dec_pos), 0, None)
        ax[3, 0].set_title(r'$\theta = %.3g$ deg, $\omega = %.3g$ deg/s' % (theta_fix, omega_fix))
        ax[3, 0].set_xlabel(r'$x$ [m]')
        ax[3, 0].set_ylabel(r'$v$ [m/s]')
        im = ax[3, 0].imshow(z.T,
                             origin='lower',
                             extent=true_limits[(0, 2), :].ravel(), 
                             aspect=true_limits[0, 0] / true_limits[2, 0], 
                             cmap=cmap,
                             vmin=1,
                             vmax=Nmax)
        cbar = fig.colorbar(im, ax=ax[3, 0], label=r'$N$')#, ticks=np.arange(1, Nmax + 1))
        
        # Safe velocities set, with fixed positions
        # z = lyapunov._N[vel_set].reshape(lyapunov.discretization.num_points[2:])
        z = np.clip(np.ceil(threshold_vel / dec_vel), 0, None)
        ax[3, 1].set_title(r'$x = %.3g$ m, $v = %.3g$ m/s' % (x_fix, v_fix))
        ax[3, 1].set_xlabel(r'$\theta$ [deg]')
        ax[3, 1].set_ylabel(r'$\omega$ [deg/s]')
        im = ax[3, 1].imshow(z.T, 
                             origin='lower', 
                             extent=true_limits[(1, 3), :].ravel(), 
                             aspect=true_limits[1, 0] / true_limits[3, 0], 
                             cmap=cmap,                   
                             vmin=1,
                             vmax=Nmax)
        cbar = fig.colorbar(im, ax=ax[3, 1], label=r'$N$')#, ticks=np.arange(1, Nmax + 1))
                
        ###################################################################################################
        
        plt.show()
        

def get_unique_subfolder(data_dir, prefix):
    """
    Return directory string consisting of prefix appended by an underscore and
    then an integer. Ensures string is unique from other similarly prefixed
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


def constrained_batch_sampler(dynamics, policy, state_dim, batch_size, action_limit=None, zero_pad=0):
    batch = tf.random_uniform([int(batch_size), state_dim], -1, 1, dtype=tf_dtype, name='batch_sample')
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
        constrained_batch = tf.zeros([1, state_dim], dtype=tf_dtype)
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
    """
    Only works for a single query point! See approx_local_lipschitz_grid() otherwise.
    """
    # Sample deviations from an epsilon-ball centred at the query point
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
        candidate = np.amax(np.abs(new - old))
        if candidate > max_change:
            max_change = candidate
    return max_change


def compute_closedloop_response(dynamics, policy, state_dim, steps, dt, reference='zero', const=1.0, ic=None):
    """
    """
    action_dim = policy.output_dim

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

    def __init__(self, pendulum_mass, cart_mass, length, rot_friction=0.0,
                 dt=0.01, normalization=None):
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

        inner_euler_steps=10
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
