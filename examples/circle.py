from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from nurbs import Curve
from plot_functions import plot

a = 1 / sqrt(2)

knots = np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1], dtype=float)

ctrlpts = np.array([[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]], dtype=float)

weights = np.array([1, a, 1, a, 1, a, 1, a, 1], dtype=float)

circle = Curve(2, ctrlpts.T, knots, weights)

plot(circle, frenet_serret=True)
plt.show()

knots = np.array([0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1], dtype=float)
ctrlpts = np.array([[1, 0], [1, 2], [-1, 2], [-1, 0], [-1, -2], [1, -2], [1, 0]],
                                   dtype=float)
a = 1/3
weights = np.array([1, a, a, 1, a, a, 1], dtype=float)
circle = Curve(3, ctrlpts.T, knots, weights)
plot(circle, frenet_serret=True)
plt.show()
