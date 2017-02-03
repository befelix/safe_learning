import numpy as np
import matplotlib.pyplot as plt


def plot_lyapunov_1d(lyapunov, true_dynamics, legend=False):
    """Plot the lyapunov function of a 1D system

    Parameters
    ----------
    lyapunov : LyapunovFunction
    true_dynamics : callable
    legend : bool, optional
    """
    if lyapunov.is_continuous:
        v_dec_string = '\dot{V}'
    else:
        v_dec_string = '\Delta V'

    threshold = lyapunov.threshold
    # Lyapunov function
    mean, bound = lyapunov.dynamics(lyapunov.discretization, lyapunov.policy)
    v_dot_mean, v_dot_bound = lyapunov.v_decrease_confidence(mean, bound)
    safe_set = lyapunov.safe_set
    extent = [np.min(lyapunov.discretization), np.max(lyapunov.discretization)]

    # Create figure axes
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Format axes
    axes[0].set_title('GP model of the dynamics')
    axes[0].set_xlim(extent)
    axes[1].set_xlim(extent)
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel(r'Upper bound of ${}(x)$'.format(v_dec_string))
    axes[1].set_title(r'Determining stability with ${}(x)$'.format(v_dec_string))

    # Plot dynamics
    axes[0].plot(lyapunov.discretization,
                 true_dynamics(lyapunov.discretization,
                               lyapunov.policy,
                               noise=False),
                 color='black', alpha=0.8)

    axes[0].fill_between(lyapunov.discretization[:, 0],
                         mean[:, 0] - bound[:, 0],
                         mean[:, 0] + bound[:, 0],
                         color=(0.8, 0.8, 1))

    axes[0].plot(lyapunov.dynamics.gaussian_process.X[:, 0],
                 lyapunov.dynamics.gaussian_process.Y[:, 0],
                 'x', ms=8, mew=2)

    # Plot V_dot
    v_dot_est_plot = plt.fill_between(
        lyapunov.discretization.squeeze(),
        v_dot_mean - v_dot_bound,
        v_dot_mean + v_dot_bound,
        color=(0.8, 0.8, 1))

    threshold_plot = plt.plot(extent, [threshold, threshold],
                              'k-.', label=r'Safety threshold ($L \tau$ )')

    # Plot the true V_dot or Delta_V
    evaluated_true_dynamics = true_dynamics(lyapunov.discretization,
                                            lyapunov.policy,
                                            noise=False)
    delta_v, _ = lyapunov.v_decrease_confidence(evaluated_true_dynamics)
    v_dot_true_plot = axes[1].plot(lyapunov.discretization.squeeze(),
                                   delta_v,
                                   color='k',
                                   label=r'True ${}(x)$'.format(v_dec_string))

    # Create twin axis
    ax2 = axes[1].twinx()
    ax2.set_ylabel(r'$V(x)$')
    ax2.set_xlim(extent)

    # Plot Lyapunov function
    V_unsafe = np.ma.masked_where(safe_set, lyapunov.V)
    V_safe = np.ma.masked_where(~safe_set, lyapunov.V)
    unsafe_plot = ax2.plot(lyapunov.discretization, V_unsafe,
                           color='b',
                           label=r'$V(x)$ (unsafe, ${}(x) > L \tau$)'.format(
                               v_dec_string))
    safe_plot = ax2.plot(lyapunov.discretization, V_safe,
                         color='r',
                         label=r'$V(x)$ (safe, ${}(x) \leq L \tau$)'.format(
                             v_dec_string
                         ))

    if legend:
        lns = unsafe_plot + safe_plot + threshold_plot + v_dot_true_plot
        labels = [x.get_label() for x in lns]
        plt.legend(lns, labels, loc=4, fancybox=True, framealpha=0.75)

    # Create helper lines
    if np.any(safe_set):
        max_id = np.argmax(lyapunov.V[safe_set])
        x_safe = lyapunov.discretization[safe_set][max_id]
        y_range = axes[1].get_ylim()
        axes[1].plot([x_safe, x_safe], y_range, 'k-.')
        axes[1].plot([-x_safe, -x_safe], y_range, 'k-.')

    # Show plot
    plt.show()
