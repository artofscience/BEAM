import numpy as np
from nurbs import Curve, plot, plot_curvature
import matplotlib.pyplot as plt

knots = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=float)
ctrlpts = np.array([[0, 0], [1, 0], [2, 1], [3, 1], [3, 0]], dtype=float)

beam = Curve(ctrlpts=ctrlpts.T, degree=3, knots=knots)

plot(beam, frenet_serret=True)
plot_curvature(beam)
plt.show()
