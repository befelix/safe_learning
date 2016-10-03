import numpy as np
import matplotlib.pyplot as plt


def plot_lyapunov_1d(lyapunov, true_dynamics, threshold, legend=False):
    """Plot the lyapunov function of a 1D system

    Parameters
    ----------
    lyapunov: LyapunovFunction
    threshold: float
    true_dynamics: callable
    legend: bool, optional
    """

    # Lyapunov function
    mean, var = lyapunov.dynamics_model.predict_noiseless(lyapunov.discretization)
    V_dot_mean, V_dot_var = lyapunov.v_dot_distribution(mean, var)
    safe_set = lyapunov.safe_set
    extent = [np.min(lyapunov.discretization), np.max(lyapunov.discretization)]

    # Create figure axes
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Format axes
    axes[0].set_title('GP model of the dynamics')
    axes[1].set_xlim(extent)
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel(r'Upper bound of $\dot{V}(x)$')
    axes[1].set_title(r'Determining stability with $\dot{V}(x)$')

    # Plot dynamics
    axes[0].plot(lyapunov.discretization,
                 true_dynamics(lyapunov.discretization,
                               noise=False),
                 color='black', alpha=0.8)

    axes[0].fill_between(lyapunov.discretization[:, 0],
                         mean[:, 0] + lyapunov.beta * np.sqrt(var[:, 0]),
                         mean[:, 0] - lyapunov.beta * np.sqrt(var[:, 0]),
                         color=(0.8, 0.8, 1))

    axes[0].plot(lyapunov.dynamics_model.X,
                 lyapunov.dynamics_model.Y,
                 'x', ms=8, mew=2)

    # Plot V_dot
    v_dot_est_plot = plt.fill_between(
        lyapunov.discretization.squeeze(),
        V_dot_mean + lyapunov.beta * np.sqrt(V_dot_var),
        V_dot_mean - lyapunov.beta * np.sqrt(V_dot_var),
        color=(0.8, 0.8, 1))

    threshold_plot = plt.plot(extent, [threshold, threshold],
                              'k-.', label=r'Safety threshold ($L \tau$ )')
    v_dot_true_plot = axes[1].plot(lyapunov.discretization,
                                   lyapunov.dV * true_dynamics(lyapunov.discretization,
                                                               noise=False),
                                   color='k',
                                   label=r'True $\dot{V}(x)$')

    # Create twin axis
    ax2 = axes[1].twinx()
    ax2.set_ylabel(r'$V(x)$')
    ax2.set_xlim(extent)

    # Plot Lyapunov function
    V_unsafe = np.ma.masked_where(safe_set, lyapunov.V)
    V_safe = np.ma.masked_where(~safe_set, lyapunov.V)
    unsafe_plot = ax2.plot(lyapunov.discretization, V_unsafe,
                           color='b',
                           label=r'$V(x)$ (unsafe, $\dot{V}(x) > L \tau$)')
    safe_plot = ax2.plot(lyapunov.discretization, V_safe,
                         color='r',
                         label=r'$V(x)$ (safe, $\dot{V}(x) \leq L \tau$)')

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