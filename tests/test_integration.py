import numpy as np
import pytest
from analysis import Beam
from math import pi, sqrt


@pytest.fixture(params=[2, 3])
def p(request):
    return request.param


@pytest.fixture(params=[0.1, 1, 10])
def L(request):
    return request.param


@pytest.fixture()
def line(p, L):
    ctrlpts = np.zeros([2, p + 1], dtype=float)
    ctrlpts[0, :] = np.linspace(0, L, p + 1)
    knots = np.zeros(2 * (p + 1), dtype=float)
    knots[p+1::] = 1
    myline = Beam(p, ctrlpts, knots)
    return myline, L


@pytest.fixture()
def circle(L):
    ctrlpts = L * np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]], dtype=float)
    a = 1 / sqrt(2)
    weights = np.array([1, a, 1, a, 1, a, 1, a, 1], dtype=float)
    knots = np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1], dtype=float)
    circle2 = Beam(2, ctrlpts.T, knots, weights)

    knots = np.array([0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1], dtype=float)
    ctrlpts = L * np.array([[1, 0], [1, 2], [-1, 2], [-1, 0], [-1, -2], [1, -2], [1, 0]], dtype=float)
    a = 1 / 3
    weights = np.array([1, a, a, 1, a, a, 1], dtype=float)
    circle3 = Beam(3, ctrlpts.T, knots, weights)
    return [circle2, circle3], 2 * pi * L


def test_arclength_line_quadrature(line):
    assert line[0].arclength() == pytest.approx(line[1])


def test_arclength_circle_quadrature(circle):
    for idx, i in enumerate(circle[0]):
        assert i.arclength() == pytest.approx(circle[1], 1e-3)


def test_arclength_line_hpr(line):
    L = 0
    for idx, i in enumerate(line[0].int_pts):
        L += line[0].int_w[idx] * line[0].arclength_differential(i)
    assert L == pytest.approx(line[1])


def test_arclength_circle_hpr(circle):
    for idx, i in enumerate(circle[0]):
        L = 0
        for idx, j in enumerate(i.int_pts):
            L += i.int_w[idx] * i.arclength_differential(j)
        assert L == pytest.approx(circle[1], 1e-2)

