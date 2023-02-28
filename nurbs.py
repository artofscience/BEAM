import numpy as np
from scipy import special

from bspline_basis_functions import basis_polynomials, basis_polynomials_derivatives


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

    def __init__(self, degree, ctrlpts, knots, weights=None):
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
        self.nks = np.unique(knots).size - 1

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

    def value(self, u):

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

        """ Compute the derivatives of a NURBS curve up to the desired order

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

        u = np.asarray(u, dtype=float)

        # Map the control points to homogeneous space | P_w = (x*w,y*w,w)
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
        bspline_derivatives = np.zeros((up_to_order + 1, np.shape(P)[0], u.size), dtype=float)

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

    def store_basis_functions(self, u):

        u = np.asarray(u, dtype=float)
        # Store the B-spline and NURBS basis functions and derivatives (first and second) evaluated at integration points
        self.basis_bspline = np.zeros([3, self.n + 1, u.size], dtype=float)
        self.basis_bspline[0] = basis_polynomials(self.n, self.p, self.U, u)
        self.basis_bspline[1] = basis_polynomials_derivatives(self.n, self.p, self.U, u, 1)
        self.basis_bspline[2] = basis_polynomials_derivatives(self.n, self.p, self.U, u, 2)

        self.basis_nurbs = np.zeros_like(self.basis_bspline)
        for pt in range(u.size):
            sum = np.dot(self.basis_bspline[0, :, pt], self.W)
            dsum = np.dot(self.basis_bspline[1, :, pt], self.W)
            ddsum = np.dot(self.basis_bspline[2, :, pt], self.W)
            for i in range(self.n + 1):
                a = self.basis_bspline[0, i, pt] * self.W[i]
                b = self.basis_bspline[1, i, pt] * self.W[i]
                c = self.basis_bspline[2, i, pt] * self.W[i]
                self.basis_nurbs[0, i, pt] = a / sum
                self.basis_nurbs[1, i, pt] = (b / sum) - (a * dsum / sum ** 2)
                self.basis_nurbs[2, i, pt] = (c / sum) + (2 * a * dsum ** 2 / sum ** 3) - (2 * b * dsum / sum ** 2) - (
                            a * ddsum / sum ** 2)

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
        u = np.asarray(u, dtype=float)

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
        u = np.asarray(u, dtype=float)
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
        u = np.asarray(u, dtype=float)
        dC, ddC = self.derivatives(u, up_to_order=2)[[1, 2], ...]

        # Compute the curvature
        dC = np.concatenate((dC, np.zeros((1, np.asarray(u).size))), axis=0)
        ddC = np.concatenate((ddC, np.zeros((1, np.asarray(u).size))), axis=0)
        numerator = np.sum(np.cross(ddC, dC, axisa=0, axisb=0, axisc=0) ** 2, axis=0) ** (1 / 2)
        denominator = (np.sum(dC ** 2, axis=0)) ** (3 / 2)
        return numerator / denominator


