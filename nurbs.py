import numba as nb
import numpy as np
from scipy import special, integrate
import matplotlib.pyplot as plt


class Analysis:
    R = np.array([[0, -1], [1, 0]])

    def __init__(self, current_config=None, reference_config=None, integration_points=None):
        self.a2 = None
        self.da1 = None
        self.a1 = None
        self.epsilon = None
        self.sigma = None
        self.A2 = None
        self.dA1 = None
        self.A1 = None
        self.normA1 = None
        self.der_ref = None
        self.cur = current_config
        self.ref = reference_config
        self.int_pts = integration_points
        if self.ref is not None and self.int_pts is not None:
            self.set_ref_props()

    def set_ref(self, curve):
        self.ref = curve

    def set_ref_props(self):
        self.der_ref = self.ref.derivatives(self.int_pts, 2)
        self.normA1 = np.linalg.norm(self.der_ref[1])
        self.A1 = self.der_ref[1] / self.normA1
        self.dA1 = self.der_ref[2] / self.normA1
        self.A2 = self.R @ self.A1

    def set_int_pts(self, int_pts):
        self.int_pts = int_pts

    def strain(self, xsi=None):

        pts = xsi if xsi is not None else self.int_pts

        # Calculate derivatives of current configuration
        der_cur = self.cur.derivatives(pts, 2)
        self.a1 = der_cur[1] / self.normA1
        self.da1 = der_cur[2] / self.normA1
        self.a2 = self.R @ self.a1

        # Calculate strain
        self.epsilon = np.zeros([2, pts.size], dtype=float)
        self.epsilon[0] = (np.einsum('ij,ij->j', self.a1, self.a1) - np.einsum('ij,ij->j', self.A1, self.A1)) / 2
        self.epsilon[1] = np.einsum('ij,ij->j', self.A2, self.dA1) - np.einsum('ij,ij->j', self.a2, self.da1)

        return self.epsilon

    def stress(self, youngs_modulus=1, area=1, second_moment_of_area=1, strain=None):
        epsilon = self.epsilon if strain is None else strain
        self.sigma = np.zeros_like(epsilon)
        self.sigma[0] = youngs_modulus * area * epsilon[0]
        self.sigma[1] = youngs_modulus * second_moment_of_area * epsilon[1]
        return self.sigma

    def bmatrix(self):
        # Size of bmatrix is 2 x 2 per control point per evaluation point
        # Total size is 2n x 2 x neval_pts
        n = self.ref.P.shape[1] - 1
        B = np.zeros([2 * n, 2, self.int_pts.size])
        norma1 = np.linalg.norm(self.a1, axis=0)
        for i in range(n):
            j = 2*i
            k = j + 2
            B[j:k, 0, :] += self.a1
            B[j:k, 1, :] -= self.a2
            B[j:k, 1, :] -= np.multiply(norma1**-3, np.einsum('ij,jik->jk', self.da1, self.R @ np.einsum('i...,j...', self.a1, self.a1)).T)
            B[j:k, 1, :] -= np.multiply(norma1**-1, np.einsum('jl,jk->kl', self.da1, self.R))

        return B


class Curve:
    """ Create a NURBS (Non-Uniform Rational Basis Spline) curve object
        Provide the arrays of control points and weights, degree and knot vector.

        This class includes methods to compute:
            - Curve coordinates
            - Analytic curve derivatives of any order
            - The unitary tangent and normal vectors (Frenet-Serret reference frame)
            - The analytic curvature
            - The arc length of the curve.
                The arc length is computed by numerical quadrature using analytic derivative information

        References
        ----------
        The NURBS Book. See references to equations and algorithms throughout the code
        L. Piegl and W. Tiller
        Springer, second edition

        All references correspond to The NURBS book unless it is explicitly stated that they come from Farin's book
    """

    def __init__(self, ctrlpts=None, weights=None, degree=None, knots=None):
        """
        Parameters
        ----------
        ctrlpts : ndarray with shape (ndim, n+1)
        Array containing the coordinates of the control points
        The first dimension of ´P´ spans the coordinates of the control points (any number of dimensions)
        The second dimension of ´P´ spans the u-direction control points ´(0,1,...,n)´

        weights : ndarray with shape (n+1,)
        Array containing the weight of the control points

        degree : int
        Degree of the B-Spline basis polynomials

        knots : ndarray with shape (r+1=n+p+2,) (r+1 is number of knots)
        Knot vector in the u-direction
        Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline
        """
        # Declare input variables as instance variables
        self.ndim = 2
        self.P = ctrlpts
        self.W = weights if weights is not None else np.ones(self.P.shape[1])
        self.p = degree
        self.U = knots
        # Highest index of the control points (counting from zero)
        self.n = np.shape(self.P)[1] - 1

        if self.P.ndim > 2:
            raise Exception('P must be an array of shape (ndim, n+1)')
        if self.W.ndim > 1:
            raise Exception('W must be an array of shape (n+1,)')
        if not np.isscalar(self.p):
            raise Exception('p must be an scalar')
        if self.U.ndim > 1:
            raise Exception('U must be an array of shape (r+1=n+p+2,)')
        if self.p > self.n:
            raise Exception('p must be equal or lower than the number of basis polynomials')

    def coordinates(self, u):

        """ Evaluate the coordinates of a NURBS curve for the input u parametrization

               This function computes the coordinates of the NURBS curve in homogeneous space using equation 4.5 and then
               maps the coordinates to ordinary space using the perspective map given by equation 1.16. See algorithm A4.1

               Parameters
               ----------

               u : scalar or ndarray with shape (N,)
                   Parameter used to evaluate the curve

               Returns
               -------
               C : ndarray with shape (ndim, N)
                   Array containing the coordinates of the curve
                   The first dimension of ´C´ spans the ´(x,y,z)´ coordinates
                   The second dimension of ´C´ spans the ´u´ parametrization sample points

               """

        # Check the shape of the input parameters
        if np.isscalar(u):
            u = np.asarray(u)
        elif u.ndim > 1:
            raise Exception('u must be a scalar or an array of shape (N,)')

        # Compute the B-Spline basis polynomials
        N_basis = basis_polynomials(self.n, self.p, self.U, u)

        # Map the control points to homogeneous space | P_w = (x*w,y*w,z*w,w)
        P_w = np.concatenate((self.P * self.W[np.newaxis, :], self.W[np.newaxis, :]), axis=0)

        # Compute the coordinates of the NURBS curve in homogeneous space
        # The summations over n is performed exploiting matrix multiplication (vectorized code)
        C_w = np.dot(P_w, N_basis)

        # Map the coordinates back to the ordinary space
        return C_w[0:-1, :] / C_w[-1, :]

    def derivatives(self, u, up_to_order):

        """ Compute the derivatives of a NURBS curve in ordinary space up to the desired order

        This function computes the analytic derivatives of the NURBS curve in ordinary space using equation 4.8 and
        the derivatives of the NURBS curve in homogeneous space obtained from compute_bspline_derivatives()

        The derivatives are computed recursively in a fashion similar to algorithm A4.2

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Parameter used to evaluate the curve

        up_to_order : integer
            Order of the highest derivative

        Returns
        -------
        nurbs_derivatives: ndarray of shape (up_to_order+1, ndim, Nu)
            The first dimension spans the order of the derivatives (0, 1, 2, ...)
            The second dimension spans the coordinates (x,y,z,...)
            The third dimension spans u-parametrization sample points

        """

        # Map the control points to homogeneous space | P_w = (x*w,y*w,z*w,w)
        P_w = np.concatenate((self.P * self.W[np.newaxis, :], self.W[np.newaxis, :]), axis=0)

        # Compute the derivatives of the NURBS curve in homogeneous space
        bspline_derivatives = self.bspline_derivatives(P_w, self.p, self.U, u, up_to_order)
        A_ders = bspline_derivatives[:, 0:-1, :]
        w_ders = bspline_derivatives[:, [-1], :]

        # Initialize array of derivatives
        n_dim, Nu = np.shape(self.P)[0], np.asarray(u).size  # Get sizes
        nurbs_derivatives = np.zeros((up_to_order + 1, n_dim, Nu), dtype=float)  # Initialize array with zeros

        # Compute the derivatives of up to the desired order
        # See algorithm A4.2 from the NURBS book
        for order in range(up_to_order + 1):

            # Update the numerator of equation 4.8 recursively
            temp_numerator = A_ders[[order], ...]
            for i in range(1, order + 1):
                temp_numerator -= special.binom(order, i) * w_ders[[i], ...] * nurbs_derivatives[[order - i], ...]

            # Compute the k-th order NURBS curve derivative
            nurbs_derivatives[order, ...] = temp_numerator / w_ders[[0], ...]

        return nurbs_derivatives

    @staticmethod
    def bspline_derivatives(P, p, U, u, up_to_order):

        """ Compute the derivatives of a B-Spline (or NURBS curve in homogeneous space) up to order `derivative_order`

        This function computes the analytic derivatives of a B-Spline curve using equation 3.3. See algorithm A3.2

        Parameters
        ----------
        P : ndarray with shape (ndim, n+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points ´(x,y,z)´
            The second dimension of ´P´ spans the u-direction control points ´(0,1,...,n)´

        p : int
            Degree of the B-Spline basis polynomials

        U : ndarray with shape (r+1=n+p+2,)
            Knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

        u : scalar or ndarray with shape (N,)
            Parameter used to evaluate the curve

        up_to_order : integer
            Order of the highest derivative

        Returns
        -------
        bspline_derivatives: ndarray of shape (up_to_order+1, ndim, Nu)
            The first dimension spans the order of the derivatives (0, 1, 2, ...)
            The second dimension spans the coordinates (x,y,z,...)
            The third dimension spans u-parametrization sample points

        """

        # Set the data type used to initialize arrays (set `complex` if an argument is complex and `float` if not)
        u = np.asarray(u)

        # Set the B-Spline coordinates as the zero-th derivative
        n_dim, Nu = np.shape(P)[0], np.asarray(u).size
        bspline_derivatives = np.zeros((up_to_order + 1, n_dim, Nu), dtype=float)

        # Compute the derivatives of up to the desired order (start at index 1 and end at index `p`)
        # See algorithm A3.2 from the NURBS book
        for order_u in range(min(p, up_to_order) + 1):
            # Highest index of the control points (counting from zero)
            n = np.shape(P)[1] - 1

            # Compute the B-Spline basis polynomials
            N_basis = basis_polynomials_derivatives(n, p, U, u, order_u)

            # Compute the coordinates of the B-Spline
            # The summations over n is performed exploiting matrix multiplication (vectorized code)
            bspline_derivatives[order_u, :, :] = np.dot(P, N_basis)

        # Note that derivative with order higher than `p` are not computed and are be zero from initialization
        # These zero-derivatives are required to compute the higher order derivatives of rational curves

        return bspline_derivatives

    def tangent(self, u):

        """ Evaluate the unitary tangent vector to the curve for the input u-parametrization

        The definition of the unitary tangent vector is given by equation 10.5 (Farin's textbook)

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Scalar or array containing the u-parameter used to evaluate the function

        Returns
        -------
        tangent : ndarray with shape (ndim, N)
            Array containing the unitary tangent vector
            The first dimension of ´tangent´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´tangent´ spans the ´u´ parametrization sample points

        """

        # Compute the curve derivatives
        dC, = self.derivatives(u, up_to_order=1)[[1], ...]

        # Compute the tangent vector
        return dC / np.linalg.norm(dC, axis=0)

    def normal(self, u):

        """ Evaluate the unitary normal vector using the special formula 2D formula

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Scalar or array containing the u-parameter used to evaluate the function

        Returns
        -------
        normal : ndarray with shape (2, N)
            Array containing the unitary normal vector
            The first dimension of ´normal´ spans the ´(x,y)´ coordinates
            The second dimension of ´normal´ spans the ´u´ parametrization sample points

        """

        # Compute the curve derivatives
        u = np.asarray(u)
        dC = self.derivatives(u, up_to_order=1)[1, ...]

        # Compute the normal vector
        numerator = np.concatenate((-dC[[1], :], dC[[0], :]), axis=0)
        return numerator / np.linalg.norm(numerator)

    def curvature(self, u):

        """ Evaluate the curvature of the curve for the input u-parametrization

        The definition of the curvature is given by equation 10.7 (Farin's textbook)

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Scalar or array containing the u-parameter used to evaluate the curvature

        Returns
        -------
        curvature : scalar or ndarray with shape (N, )
            Scalar or array containing the curvature of the curve

        """

        # Compute the curve derivatives
        u = np.asarray(u)
        dC, ddC = self.derivatives(u, up_to_order=2)[[1, 2], ...]

        # Compute the curvature
        dC = np.concatenate((dC, np.zeros((1, np.asarray(u).size))), axis=0)
        ddC = np.concatenate((ddC, np.zeros((1, np.asarray(u).size))), axis=0)
        numerator = np.sum(np.cross(ddC, dC, axisa=0, axisb=0, axisc=0) ** 2, axis=0) ** (1 / 2)
        denominator = (np.sum(dC ** 2, axis=0)) ** (3 / 2)
        return numerator / denominator

    def arclength(self, u1=0.00, u2=1.00):

        """ Compute the arc length of a parametric curve in the interval [u1,u2] using numerical quadrature

        The definition of the arc length is given by equation 10.3 (Farin's textbook)

        Parameters
        ----------
        u1 : scalar
            Lower limit of integration for the arc length computation

        u2 : scalar
            Upper limit of integration for the arc length computation

        Returns
        -------
        L : scalar
            Arc length of NURBS curve in the interval [u1, u2]

        """

        # Compute the arc length differential analytically
        def get_arclegth_differential(u):
            dCdu = self.derivatives(u, up_to_order=1)[1, ...]
            dLdu = np.sqrt(np.sum(dCdu ** 2, axis=0))  # dL/du = [(dx_0/du)^2 + ... + (dx_n/du)^2]^(1/2)
            return dLdu

        # Compute the arc length of C(t) in the interval [u1, u2] by numerical integration
        arclength = integrate.fixed_quad(get_arclegth_differential, u1, u2, n=10)[0]

        return arclength


@nb.jit(nopython=True, cache=True)
def basis_polynomials(n, p, U, u, return_degree=None):
    """ Evaluate the n-th B-Spline basis polynomials of degree ´p´ for the input u-parametrization

    The basis polynomials are computed from their definition by implementing equation 2.5 directly

    Parameters
    ----------
    n : integer
        Highest index of the basis polynomials (n+1 basis polynomials)

    p : integer
        Degree of the basis polynomials

    U : ndarray with shape (r+1=n+p+2,)
        Knot vector of the basis polynomials

    u : scalar or ndarray with shape (Nu,)
        Parameter used to evaluate the basis polynomials

    return_degree : int
        Degree of the returned basis polynomials

    Returns
    -------
    N : ndarray with shape (n+1, Nu)
        Array containing the basis polynomials of order ´p´ evaluated at ´u´
        The first dimension of ´N´ spans the n-th polynomials
        The second dimension of ´N´ spans the ´u´ parametrization sample points

    """

    # Number of points where the polynomials are evaluated (vectorized computations)
    u = np.asarray(u * 1.0)  # Convert to array of floats
    Nu = u.size

    # Number of basis polynomials at the current step of the recursion
    # Number of basis polynomials equals to number of control points plus degree
    m = n + p + 1

    # Initialize the array of basis polynomials
    N = np.zeros((p + 1, m, Nu), dtype=u.dtype)

    # First step of the recursion formula (p = 0)
    # The case when point_index==n and u==1 is a special case. See the NURBS book section 2.5 and algorithm A2.1
    # The np.real() operator is used such that the function extends to complex u-parameter as well
    for i in range(m):
        N[0, i, :] = 0.0 + 1.0 * (u >= U[i]) * (u < U[i + 1]) + 1.00 * (
            np.logical_and(u == U[-1], i == n))

    # Second and next steps of the recursion formula (p = 1, 2, ...)
    for k in range(1, p + 1):

        # Update the number of basis polynomials
        m = m - 1

        # Compute the basis polynomials using the de Boor recursion formula
        for i in range(m):

            # Compute first factor (avoid division by zero by convention)
            if (U[i + k] - U[i]) == 0:
                n1 = np.zeros((Nu,), dtype=u.dtype)
            else:
                n1 = (u - U[i]) / (U[i + k] - U[i]) * N[k - 1, i, :]

            # Compute second factor (avoid division by zero by convention)
            if (U[i + k + 1] - U[i + 1]) == 0:
                n2 = np.zeros((Nu,), dtype=u.dtype)
            else:
                n2 = (U[i + k + 1] - u) / (U[i + k + 1] - U[i + 1]) * N[k - 1, i + 1, :]

            # Compute basis polynomial (recursion formula 2.5)
            N[k, i, ...] = n1 + n2

    return N[p, 0:n + 1, :] if return_degree is None else N[return_degree, 0:n + 1, :]


@nb.jit(nopython=True, cache=True)
def basis_polynomials_derivatives(n, p, U, u, derivative_order):
    """ Evaluate the derivative of the n-th B-Spline basis polynomials of degree ´p´ for the input u-parametrization

    The basis polynomials derivatives are computed recursively by implementing equations 2.7 and 2.9 directly

    Parameters
    ----------
    n : integer
        Highest index of the basis polynomials (n+1 basis polynomials)

    p : integer
        Degree of the original basis polynomials

    U : ndarray with shape (r+1=n+p+2,)
        Knot vector of the basis polynomials
        Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

    u : scalar or ndarray with shape (Nu,)
        Parameter used to evaluate the basis polynomials

    derivative_order : scalar
        Order of the basis polynomial derivatives

    Returns
    -------
    N_ders : ndarray with shape (n+1, Nu)
        Array containing the basis spline polynomials derivatives evaluated at ´u´
        The first dimension of ´N´ spans the n-th polynomials
        The second dimension of ´N´ spans the ´u´ parametrization sample points

    """

    if derivative_order > p:
        print('The derivative order is higher than the degree of the basis polynomials')

    # Number of points where the polynomials are evaluated (vectorized computations)
    u = np.asarray(u * 1.0)  # Convert to array of floats
    Nu = u.size

    # Compute the basis polynomials to the right hand side of equation 2.9 recursively down to the zeroth derivative
    # Each new call reduces the degree of the basis polynomials by one
    if derivative_order >= 1:
        derivative_order -= 1
        N = basis_polynomials_derivatives(n, p - 1, U, u, derivative_order)
    elif derivative_order == 0:
        N = basis_polynomials(n, p, U, u)
        return N
    else:
        print('Oooopps, something went wrong in compute_basis_polynomials_derivatives()')
        N = basis_polynomials(n, p, U, u)
        return N

    # Initialize the array of basis polynomial derivatives
    N_ders = np.zeros((n + 1, Nu), dtype=u.dtype)

    # Compute the derivatives of the (0, 1, ..., n) basis polynomials using equations 2.7 and 2.9
    for i in range(n + 1):

        # Compute first factor (avoid division by zero by convention)
        if (U[i + p] - U[i]) == 0:
            n1 = np.zeros(Nu, dtype=u.dtype)
        else:
            n1 = p * N[i, :] / (U[i + p] - U[i])

        # Compute second factor (avoid division by zero by convention)
        if (U[i + p + 1] - U[i + 1]) == 0:
            n2 = np.zeros(Nu, dtype=u.dtype)
        else:
            n2 = p * N[i + 1, :] / (U[i + p + 1] - U[i + 1])

        # Compute the derivative of the current basis polynomials
        N_ders[i, :] = n1 - n2

    return N_ders


def plot(crv, u=np.linspace(0, 1, 100), fig=None, ax=None, curve=True, knots=True, control_points=True, frenet_serret=False, axis_off=False,
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

    # Set the scaling of the axes
    rescale_plot(fig, ax)

    return fig, ax


def plot_curve(crv, u, fig, ax, linewidth=2, linestyle='-', color='black'):
    """ Plot the coordinates of the NURBS curve """
    X, Y = np.real(crv.coordinates(u))
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

    Px, Py = np.real(crv.coordinates(np.unique(crv.U)))
    ax.scatter(Px, Py, markersize, marker=markerstyle, color=color)
    return fig, ax


def plot_frenet_serret(self, fig, ax, frame_number=2, frame_scale=0.10):
    """ Plot some Frenet-Serret reference frames along the NURBS curve """

    # Compute the tangent, normal and binormal unitary vectors
    h = 1e-12
    u = np.linspace(0 + h, 1 - h, frame_number)
    position = np.real(self.coordinates(u))
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


def rescale_plot(fig, ax):
    """ Adjust the aspect ratio of the figure """
    # Set the aspect ratio of the data
    ax.set_aspect(1.0)

    # Adjust pad
    plt.tight_layout(pad=5.0, w_pad=None, h_pad=None)


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
