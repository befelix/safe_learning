import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import display, HTML


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
    states = lyapunov.discretization
    mean, bound = lyapunov.dynamics(lyapunov.discretization, lyapunov.policy)
    v_dot_mean, v_dot_bound = lyapunov.v_decrease_confidence(states,
                                                             mean,
                                                             bound)
    safe_set = lyapunov.safe_set
    extent = [np.min(states), np.max(states)]

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
    delta_v, _ = lyapunov.v_decrease_confidence(states,
                                                evaluated_true_dynamics)
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


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)),
               id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:100%;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
