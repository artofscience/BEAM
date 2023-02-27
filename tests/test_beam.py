import numpy as np
import pytest
from math import cos, sin, pi
from nurbs import Beam


@pytest.fixture()
def beam():
    ctrlpts = np.array([[0, 0], [1/3, 0], [2/3, 0], [1, 0]], dtype=float)
    knots = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    return Beam(3, ctrlpts.T, knots)


def test_rigid_body_motion(beam):
    beam.P += 1
    beam.set_config(beam.cur)
    strain = beam.compute_strain()
    assert np.allclose(strain, 0.0)


@pytest.mark.parametrize("theta", [pi/2, pi, 3/2*pi, 2*pi])
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
