import pytest
from nurbs import Curve
from math import sqrt, pi, sin, cos
import numpy as np


@pytest.fixture(params=[1, 10, 100])
def crv(request):
    """Create a NURBS circle"""
    a = 1 / sqrt(2)
    knots = np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1], dtype=float)
    ctrlpts = request.param * np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]], dtype=float)
    weights = np.array([1, a, 1, a, 1, a, 1, a, 1], dtype=float)
    circle = Curve(2, ctrlpts.T, knots, weights)
    return circle, request.param


def test_curve_degree(crv):
    assert crv[0].p == 2


def test_curve_ctrlpts(crv):
    assert np.alltrue(
        crv[0].P.T == crv[1] * np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]],
                            dtype=float))


def test_curve_weights(crv):
    a = 1 / sqrt(2)
    assert np.alltrue(crv[0].W == np.array([1, a, 1, a, 1, a, 1, a, 1], dtype=float))


def test_curve_knots(crv):
    assert np.alltrue(crv[0].U == np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1], dtype=float))


def test_curve_ndim(crv):
    assert crv[0].ndim == 2


def test_arclength(crv):
    assert crv[0].arclength() == pytest.approx(2 * pi * crv[1], 1e-2)


@pytest.mark.parametrize("u", np.linspace(0, 1, 9).tolist())
class TestParametrized:
    def test_curvature(self, crv, u):
        """For a circle the curvature is constant and equal to reciprocal of radius"""
        assert crv[0].curvature(u) == pytest.approx(1/crv[1])

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
