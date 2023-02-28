from math import cos, sin, pi, sqrt

import numpy as np
import pytest

from analysis import Beam


@pytest.fixture()
def beam():
    ctrlpts = np.array([[0, 0], [1 / 3, 0], [2 / 3, 0], [1, 0]], dtype=float)
    knots = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    return Beam(3, ctrlpts.T, knots)

@pytest.fixture(params=[1, 0.1, 10])
def beaml(request):
    ctrlpts = np.array([[0, 0], [request.param / 3, 0], [2* request.param / 3, 0], [request.param, 0]], dtype=float)
    knots = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    return Beam(3, ctrlpts.T, knots), request.param

def test_beam_axis_parametrization(beaml):
    beaml[0].compute_derivatives()
    assert np.allclose(beaml[0].ref.J, beaml[1])


def test_rigid_body_motion(beam):
    beam.P += 1
    beam.set_config(beam.cur)
    strain = beam.compute_strain()
    assert np.allclose(strain, 0.0)


@pytest.mark.parametrize("theta", [pi / 2, pi, 3 / 2 * pi, 2 * pi])
def test_rigid_body_rotation(beam, theta):
    beam.P = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]], dtype=float) @ beam.P
    beam.set_config(beam.cur)
    strain = beam.compute_strain()
    assert np.allclose(strain, 0.0)


@pytest.mark.parametrize("dl", [0.001, 0.01, 0.1, 1])
def test_axial_strain(beam, dl):
    beam.P[0, :] += np.linspace(0, dl, beam.n + 1)
    beam.set_config(beam.cur)
    strain = beam.compute_strain()
    gl = dl + 0.5 * dl ** 2
    assert np.allclose(strain[0], gl)
    assert np.allclose(strain[1], 0.0)


@pytest.fixture()
def circle():
    knots = np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1], dtype=float)
    ctrlpts = np.zeros([2, 9], dtype=float)
    ctrlpts[0] = np.linspace(0, 1, 9)

    return Beam(2, ctrlpts, knots)

def test_bending_strain(circle):
    # fig, ax = plot(circle)
    circle.P = 1 / (2 * pi) * np.array([[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]], dtype=float).T
    circle.P[1] += 1 / (2 * pi)
    # plot(circle, fig=fig, ax=ax)
    # plt.show()
    a = 1 / sqrt(2)
    weights = np.array([1, a, 1, a, 1, a, 1, a, 1], dtype=float)
    circle.set_config(circle.cur)
    strain = circle.compute_strain()
    M = 2 * pi
    pass

def test_b_matrix(beam):
    beam.set_config(beam.cur)
    beam.compute_derivatives()
    b, h = beam.assembly()


    pass

