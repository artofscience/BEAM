import numpy as np

from nurbs import Beam, Material

ctrlpts = np.array([[0, 0], [0.25, 0], [0.5, 0], [0.75, 0], [1, 0]], dtype=float)
u = np.array([[0, 2], [0.25, 2], [0.5, 2], [0.75, 2], [1, 2]], dtype=float)
knots = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=float)

beam = Beam(3, ctrlpts.T, knots)
beam.P += u.T

beam.mat = Material()

strain = beam.compute_strain()
stress = beam.compute_stress()
beam.compute_derivatives()
pass
