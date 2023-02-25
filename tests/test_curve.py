from math import sqrt, pi, sin, cos

import numpy as np
import pytest

from nurbs import Curve, basis_polynomials


@pytest.fixture(params=[1, 10, 100])
def crv(request):
    """Create a NURBS circle"""
    a = 1 / sqrt(2)
    knots = np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1], dtype=float)
    ctrlpts = request.param * np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]],
                                       dtype=float)
    weights = np.array([1, a, 1, a, 1, a, 1, a, 1], dtype=float)
    circle = Curve(2, ctrlpts.T, knots, weights)
    return circle, request.param


def test_degree(crv):
    assert crv[0].p == 2


def test_ctrlpts(crv):
    assert np.alltrue(
        crv[0].P.T == crv[1] * np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]],
                                        dtype=float))


def test_weights(crv):
    a = 1 / sqrt(2)
    assert np.alltrue(crv[0].W == np.array([1, a, 1, a, 1, a, 1, a, 1], dtype=float))


def test_knots(crv):
    assert np.alltrue(crv[0].U == np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1], dtype=float))


def test_ndim(crv):
    assert crv[0].ndim == 2


def test_arclength(crv):
    assert crv[0].arclength() == pytest.approx(2 * pi * crv[1], 1e-2)


def test_endpoint_interpolation(crv):
    assert np.allclose(crv[0].coordinates(0).flatten(), crv[0].P[:, 0])
    assert np.allclose(crv[0].coordinates(1).flatten(), crv[0].P[:, -1])


def test_nurbs_zeroth_derivative(crv):
    """ Test that the zero-th derivative agrees with the function evaluation """
    # Compute the basis polynomials derivatives analytically
    u = np.linspace(0, 1, 1000)
    N = crv[0].coordinates(u)
    dN = crv[0].derivatives(u, 0)

    assert np.allclose(dN, N)


def test_nurbs_first_derivative_cfd(crv):
    """ Test the first derivative of the nurbs against central finite differences """
    # Define a new u-parametrization suitable for finite differences
    h = 1e-5
    hh = h + h ** 2
    Nu = 1000
    u = np.linspace(0.00 + hh, 1.00 - hh, Nu)  # Make sure that the limits [0, 1] also work when making changes

    # Compute the basis polynomials derivatives analytically
    dN = crv[0].derivatives(u, 1)[1]

    # Compute the basis polynomials derivatives by central finite differences
    a = -1 / 2 * crv[0].coordinates(u - h)
    b = +1 / 2 * crv[0].coordinates(u + h)
    dN_fd = (a + b) / h
    assert np.allclose(dN_fd, dN, rtol=1e-3)


def test_nurbs_second_derivative_cfd(crv):
    """ Test the second derivative of the nurbs against central finite differences """
    # Define a new u-parametrization suitable for finite differences
    h = 1e-4
    hh = h + h ** 2
    Nu = 1000
    u = np.linspace(0.00 + hh, 1.00 - hh, Nu)  # Make sure that the limits [0, 1] also work when making changes

    # Compute the basis polynomials derivatives
    ddN = crv[0].derivatives(u, 2)[2]

    # Check the second derivative against central finite differences
    a = +1 * crv[0].coordinates(u - h)
    b = -2 * crv[0].coordinates(u)
    c = +1 * crv[0].coordinates(u + h)
    ddN_fd = (a + b + c) / h ** 2
    assert np.allclose(ddN, ddN_fd, rtol=1e-2)


def test_endpoint_first_derivatives(crv):
    dbn = crv[0].derivatives(0, 1)[1]
    den = crv[0].derivatives(1, 1)[1]

    p = crv[0].p
    W = crv[0].W
    P = crv[0].P
    U = crv[0].U
    n = crv[0].n

    dba = (p / U[p + 1]) * (W[1] / W[0]) * (P[:, 1] - P[:, 0])
    dea = (p / (1 - U[n])) * (W[n - 1] / W[n]) * (P[:, n] - P[:, n - 1])

    assert np.allclose(dbn.flatten(), dba)
    assert np.allclose(den.flatten(), dea)


def test_endpoint_second_derivatives(crv):
    dbn = crv[0].derivatives(0, 2)[2]
    den = crv[0].derivatives(1, 2)[2]

    p = crv[0].p
    W = crv[0].W
    P = crv[0].P
    U = crv[0].U
    n = crv[0].n

    dba = p * (p - 1) / U[p + 1] * (
                1 / U[p + 2] * (W[2] / W[0]) * (P[:, 2] - P[:, 0]) - (1 / U[p + 1] + 1 / U[p + 2]) * (W[1] / W[0]) * (
                    P[:, 1] - P[:, 0])) + 2 * (p / U[p + 1]) ** 2 * (W[1] / W[0]) * (1 - W[1] / W[0]) * (
                      P[:, 1] - P[:, 0])
    dea = p * (p - 1) / (1 - U[n]) * (1 / (1 - U[n - 1]) * (W[n - 2] / W[n]) * (P[:, n - 2] - P[:, n]) - (
                1 / (1 - U[n]) + 1 / (1 - U[n - 1])) * (W[n - 1] / W[n]) * (P[:, n - 1] - P[:, n])) + 2 * (
                      p / (1 - U[n])) ** 2 * (W[n - 1] / W[n]) * (1 - W[n - 1] / W[n]) * (P[:, n - 1] - P[:, n])

    assert np.allclose(dbn.flatten(), dba)
    assert np.allclose(den.flatten(), dea)


def test_endpoint_curvature(crv):
    # Get the endpoint curvature numerically
    cbn = crv[0].curvature(0)
    cen = crv[0].curvature(1)

    p = crv[0].p
    W = crv[0].W
    P = crv[0].P
    U = crv[0].U
    n = crv[0].n

    # Get the endpoint curvature analytically
    cba = (p - 1) / p * (U[p + 1] / U[p + 2]) * (W[2] * W[0] / W[1] ** 2) * \
          np.sum(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]) ** 2) ** (1 / 2) * \
          np.sum((P[:, 1] - P[:, 0]) ** 2) ** (-3 / 2)

    cea = (p - 1) / p * (1 - U[n]) / (1 - U[n - 1]) * (W[n] * W[n - 2] / W[n - 1] ** 2) * \
          np.sum(np.cross(P[:, n - 1] - P[:, n], P[:, n - 2] - P[:, n]) ** 2) ** (1 / 2) * \
          np.sum((P[:, n - 1] - P[:, n]) ** 2) ** (-3 / 2)

    assert cbn == pytest.approx(cba)
    assert cen == pytest.approx(cea)






@pytest.mark.parametrize("u", np.linspace(0, 1, 9).tolist())
class TestParametrized:
    def test_curvature(self, crv, u):
        """For a circle the curvature is constant and equal to reciprocal of radius"""
        assert crv[0].curvature(u) == pytest.approx(1 / crv[1])

    def test_coordinates(self, crv, u):
        evalpt = crv[0].coordinates(u)
        angle = u * 2 * pi
        assert np.allclose(evalpt, crv[1] * np.array([[cos(angle)], [sin(angle)]]))

    def test_normal(self, crv, u):
        normalvec = crv[0].normal(u)
        angle = u * 2 * pi
        assert np.allclose(normalvec, np.array([[-cos(angle)], [-sin(angle)]]))

    def test_tangent(self, crv, u):
        tangentvec = crv[0].tangent(u)
        angle = u * 2 * pi
        assert np.allclose(tangentvec, np.array([[-sin(angle)], [cos(angle)]]))

    def test_individual_nurbs_basis_functions_zeroth_derivative(self, crv, u):
        crv[0].store_basis_functions(u)

        basis_bspline = basis_polynomials(crv[0].n, crv[0].p, crv[0].U, u).flatten()
        sum = np.dot(crv[0].W, basis_bspline)
        basis_nurbs = (crv[0].W * basis_bspline) / sum

        assert np.allclose(crv[0].basis_nurbs[0].flatten(), basis_nurbs)


def test_individual_nurbs_basis_functions_first_derivative(crv):
    h = 1e-5
    hh = h + h ** 2
    Nu = 100
    x = np.linspace(0.00 + hh, 1.00 - hh, Nu)  # Make sure that the limits [0, 1] also work when making changes

    crv[0].store_basis_functions(x)

    def basis_nurbs(y):
        basis_bspline = basis_polynomials(crv[0].n, crv[0].p, crv[0].U, y)
        sum = np.einsum('i,ij->j', crv[0].W, basis_bspline)
        R = (crv[0].W[:, np.newaxis] * basis_bspline) / sum
        return R

    # Compute the basis polynomials derivatives by central finite differences
    a = -1 / 2 * basis_nurbs(x - h)
    b = +1 / 2 * basis_nurbs(x + h)
    dR_fd = (a + b) / h

    assert np.allclose(crv[0].basis_nurbs[1], dR_fd)


def test_individual_nurbs_basis_functions_second_derivative(crv):

    h = 1e-4
    hh = h + h ** 2
    Nu = 100
    u = np.linspace(0.00 + hh, 1.00 - hh, Nu)  # Make sure that the limits [0, 1] also work when making changes
    crv[0].store_basis_functions(u)

    def basis_nurbs(u):
        basis_bspline = basis_polynomials(crv[0].n, crv[0].p, crv[0].U, u)
        sum = np.einsum('i,ij->j', crv[0].W, basis_bspline)
        R = (crv[0].W[:, np.newaxis] * basis_bspline) / sum
        return R

    # Compute the basis polynomials derivatives by central finite differences
    a = basis_nurbs(u + h)
    b = -2 * basis_nurbs(u)
    c = basis_nurbs(u - h)

    ddR_fd = (a + b + c) / h ** 2

    assert np.allclose(crv[0].basis_nurbs[2], ddR_fd)
