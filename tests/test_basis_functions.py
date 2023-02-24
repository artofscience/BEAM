import numpy as np
import pytest

from nurbs import basis_polynomials, basis_polynomials_derivatives
from nurbs import Curve

@pytest.fixture()
def problem():
    P = np.zeros((2, 5))
    n = 4
    p=3
    P[:, 0] = [0.00, 0.00]
    P[:, 1] = [0.00, 0.30]
    P[:, 2] = [0.25, 0.30]
    P[:, 3] = [0.50, 0.30]
    P[:, 4] = [0.50, 0.10]
    W = np.asarray([1, 1, 3, 1, 1])
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))
    crv = Curve(p, P, U, W)
    crv.u = np.linspace(0, 1, 101)
    return crv


class Problem:
    u = np.linspace(0, 1, 101)

    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))


def test_partition_of_unity(problem):
    """ Test the partition of unity property of the basis polynomials """
    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    N_basis = basis_polynomials(problem.n, problem.p, problem.U, problem.u)
    assert np.sum(N_basis, axis=0) == pytest.approx(1, 1e-6)


def test_basis_function_zeroth_derivative(problem):
    """ Test that the zero-th derivative agrees with the function evaluation """
    # Compute the basis polynomials derivatives analytically
    N = basis_polynomials(problem.n, problem.p, problem.U, problem.u)
    dN = basis_polynomials_derivatives(problem.n, problem.p, problem.U, problem.u, derivative_order=0)

    assert np.allclose(dN, N)


def test_basis_function_first_derivative_cfd(problem):
    """ Test the first derivative of the basis polynomials against central finite differences """
    # Define a new u-parametrization suitable for finite differences
    h = 1e-5
    hh = h + h ** 2
    Nu = 1000
    u = np.linspace(0.00 + hh, 1.00 - hh, Nu)  # Make sure that the limits [0, 1] also work when making changes

    # Compute the basis polynomials derivatives analytically
    dN = basis_polynomials_derivatives(problem.n, problem.p, problem.U, u, derivative_order=1)

    # Compute the basis polynomials derivatives by central finite differences
    a = -1 / 2 * basis_polynomials(problem.n, problem.p, problem.U, u - h)
    b = +1 / 2 * basis_polynomials(problem.n, problem.p, problem.U, u + h)
    dN_fd = (a + b) / h
    assert np.allclose(dN_fd, dN)


def test_basis_function_second_derivative_cfd(problem):
    """ Test the second derivative of the basis polynomials against central finite differences """
    # Define a new u-parametrization suitable for finite differences
    h = 1e-4
    hh = h + h ** 2
    Nu = 1000
    u = np.linspace(0.00 + hh, 1.00 - hh, Nu)  # Make sure that the limits [0, 1] also work when making changes

    # Compute the basis polynomials derivatives
    ddN = basis_polynomials_derivatives(problem.n, problem.p, problem.U, u, derivative_order=2)

    # Check the second derivative against central finite differences
    a = +1 * basis_polynomials(problem.n, problem.p, problem.U, u - h)
    b = -2 * basis_polynomials(problem.n, problem.p, problem.U, u)
    c = +1 * basis_polynomials(problem.n, problem.p, problem.U, u + h)
    ddN_fd = (a + b + c) / h ** 2
    assert np.allclose(ddN, ddN_fd)


def test_nurbs_zeroth_derivative(problem):
    """ Test that the zero-th derivative agrees with the function evaluation """
    # Compute the basis polynomials derivatives analytically
    N = problem.coordinates(problem.u)
    dN = problem.derivatives(problem.u, 0)

    assert np.allclose(dN, N)


def test_nurbs_first_derivative_cfd(problem):
    """ Test the first derivative of the nurbs against central finite differences """
    # Define a new u-parametrization suitable for finite differences
    h = 1e-5
    hh = h + h ** 2
    Nu = 1000
    u = np.linspace(0.00 + hh, 1.00 - hh, Nu)  # Make sure that the limits [0, 1] also work when making changes

    # Compute the basis polynomials derivatives analytically
    dN = problem.derivatives(u, 1)[1]

    # Compute the basis polynomials derivatives by central finite differences
    a = -1 / 2 * problem.coordinates(u - h)
    b = +1 / 2 * problem.coordinates(u + h)
    dN_fd = (a + b) / h
    assert np.allclose(dN_fd, dN)


def test_nurbs_second_derivative_cfd(problem):
    """ Test the second derivative of the nurbs against central finite differences """
    # Define a new u-parametrization suitable for finite differences
    h = 1e-4
    hh = h + h ** 2
    Nu = 1000
    u = np.linspace(0.00 + hh, 1.00 - hh, Nu)  # Make sure that the limits [0, 1] also work when making changes

    # Compute the basis polynomials derivatives
    ddN = problem.derivatives(u, 2)[2]

    # Check the second derivative against central finite differences
    a = +1 * problem.coordinates(u - h)
    b = -2 * problem.coordinates(u)
    c = +1 * problem.coordinates(u + h)
    ddN_fd = (a + b + c) / h ** 2
    assert np.allclose(ddN, ddN_fd, rtol=1e-3)

class TestExample2_2:
    """ Test the basis function value against a known example (Ex2.2 from the NURBS book) """

    @staticmethod
    def analytic_polynomials(u):
        N = np.zeros((8, u.size), dtype=float)
        for i, u in enumerate(u):
            N02 = (1 - u) ** 2 * (0 <= u < 1)
            N12 = (2 * u - 3 / 2 * u ** 2) * (0 <= u < 1) + (1 / 2 * (2 - u) ** 2) * (1 <= u < 2)
            N22 = (1 / 2 * u ** 2) * (0 <= u < 1) + (-3 / 2 + 3 * u - u ** 2) * (1 <= u < 2) + (
                    1 / 2 * (3 - u) ** 2) * (2 <= u < 3)
            N32 = (1 / 2 * (u - 1) ** 2) * (1 <= u < 2) + (-11 / 2 + 5 * u - u ** 2) * (2 <= u < 3) + (
                    1 / 2 * (4 - u) ** 2) * (3 <= u < 4)
            N42 = (1 / 2 * (u - 2) ** 2) * (2 <= u < 3) + (-16 + 10 * u - 3 / 2 * u ** 2) * (3 <= u < 4)
            N52 = (u - 3) ** 2 * (3 <= u < 4) + (5 - u) ** 2 * (4 <= u < 5)
            N62 = (2 * (u - 4) * (5 - u)) * (4 <= u < 5)
            N72 = (u - 4) ** 2 * (4 <= u <= 5)
            N[:, i] = np.asarray([N02, N12, N22, N32, N42, N52, N62, N72])
        return N

    def test_basis_functions(self):
        U = np.asarray([0.00, 0.00, 0.00, 1.00, 2.00, 3.00, 4.00, 4.00, 5.00, 5.00, 5.00])
        uu = np.linspace(0, 5, 21)

        # Evaluate the polynomials numerically
        N_basis = basis_polynomials(7, 2, U, uu)

        # Evaluate the polynomials analytically
        N_analytic = self.analytic_polynomials(uu)

        assert np.allclose(N_basis, N_analytic)
