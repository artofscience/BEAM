import numpy as np
from matplotlib import pyplot as plt


def plot(crv, u=np.linspace(0, 1, 100), fig=None, ax=None, curve=True, knots=True, control_points=True,
         frenet_serret=False, axis_off=False,
         ticks_off=False):
    """ Create a plot and return the figure and axes handles """

    if fig is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.set_xlabel('$x$ axis', fontsize=12, color='k', labelpad=12)
        ax.set_ylabel('$y$ axis', fontsize=12, color='k', labelpad=12)
        for t in ax.xaxis.get_major_ticks():
            t.label.set_fontsize(12)
        for t in ax.yaxis.get_major_ticks():
            t.label.set_fontsize(12)
        plt.margins(0.2)
        if ticks_off:
            ax.set_xticks([])
            ax.set_yticks([])
        if axis_off:
            ax.axis('off')

    # Add objects to the plot
    if curve:
        plot_curve(crv, u, fig, ax)
    if knots:
        plot_knots(crv, fig, ax)
    if control_points:
        plot_control_points(crv, fig, ax)
    if frenet_serret:
        plot_frenet_serret(crv, fig, ax, frame_scale=1.5)

    ax.set_aspect(1.0)

    # Adjust pad
    plt.tight_layout(pad=5.0, w_pad=None, h_pad=None)

    return fig, ax


def plot_curve(crv, u, fig, ax, linewidth=2, linestyle='-', color='black'):
    """ Plot the coordinates of the NURBS curve """
    X, Y = np.real(crv.value(u))
    line, = ax.plot(X, Y)
    line.set_linewidth(linewidth)
    line.set_linestyle(linestyle)
    line.set_color(color)
    line.set_marker(' ')
    return fig, ax


def plot_control_points(crv, fig, ax, linewidth=1.5, linestyle='-.', color='red', markersize=5, markerstyle='o'):
    """ Plot the control points of the NURBS curve """
    Px, Py = np.real(crv.P)
    line, = ax.plot(Px, Py)
    line.set_linewidth(linewidth)
    line.set_linestyle(linestyle)
    line.set_color(color)
    line.set_marker(markerstyle)
    line.set_markersize(markersize)
    line.set_markeredgewidth(linewidth)
    line.set_markeredgecolor(color)
    line.set_markerfacecolor('w')
    line.set_zorder(4)
    return fig, ax


def plot_knots(crv, fig, ax, color='black', markersize=100, markerstyle='x'):
    """ Plot the knots of the NURBS curve """

    Px, Py = np.real(crv.value(np.unique(crv.U)))
    ax.scatter(Px, Py, markersize, marker=markerstyle, color=color)
    return fig, ax


def plot_frenet_serret(self, fig, ax, frame_number=2, frame_scale=0.10):
    """ Plot some Frenet-Serret reference frames along the NURBS curve """

    # Compute the tangent, normal and binormal unitary vectors
    h = 1e-12
    u = np.linspace(0 + h, 1 - h, frame_number)
    position = np.real(self.value(u))
    tangent = np.real(self.tangent(u))
    normal = np.real(self.normal(u))

    # Plot the frames of reference
    for k in range(frame_number):
        # Plot the tangent vector
        x, y = position[:, k]
        u, v = tangent[:, k]
        ax.quiver(x, y, u, v, color='red', scale=10, width=0.002)

        # Plot the normal vector
        x, y = position[:, k]
        u, v = normal[:, k]
        ax.quiver(x, y, u, v, color='blue', scale=10, width=0.002)

    # Plot the origin of the vectors
    x, y = position
    points, = ax.plot(x, y)
    points.set_linestyle(' ')
    points.set_marker('o')
    points.set_markersize(5)
    points.set_markeredgewidth(1.25)
    points.set_markeredgecolor('k')
    points.set_markerfacecolor('w')
    points.set_zorder(4)
    # points.set_label(' ')

    return fig, ax


def plot_curvature(crv, u=np.linspace(0, 1, 101), fig=None, ax=None, color='black', linestyle='-'):
    # Create the figure
    if fig is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
    ax.set_xlabel('$u$ parameter', fontsize=12, color='k', labelpad=12)
    ax.set_ylabel('Curvature', fontsize=12, color='k', labelpad=12)
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(12)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(12)

    curvature = np.real(crv.curvature(u))
    line, = ax.plot(u, curvature)
    line.set_linewidth(1.25)
    line.set_linestyle(linestyle)
    line.set_color(color)
    line.set_marker(" ")
    line.set_markersize(3.5)
    line.set_markeredgewidth(1)
    line.set_markeredgecolor("k")
    line.set_markerfacecolor("w")

    # Adjust pad
    plt.tight_layout(pad=5.0, w_pad=None, h_pad=None)

    return fig, ax
