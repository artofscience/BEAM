import numba as nb
import numpy as np


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
        print('Oooopps, something went wrong in computing the basis_polynomials_derivatives()')
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
