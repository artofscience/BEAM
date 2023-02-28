import matplotlib.pyplot as plt
import numpy as np

from nurbs import Curve
from plot_functions import plot

knots = np.array([0, 0, 0, 0, 0.25, 0.75, 1, 1, 1, 1], dtype=float)
ctrlpts_ref = np.array([[0, 0], [1, 0], [2, -1], [3, -1.5], [3.5, -2], [4.5, -1]], dtype=float)
u = np.array([[0, 0], [0.2, 0], [0, 2], [-0.1, 3], [0.1, 2], [-0.2, 3]], dtype=float)
ctrlpts_cur = ctrlpts_ref + u

beam_ref = Curve(3, ctrlpts_ref.T, knots)
beam_cur = Curve(3, ctrlpts_cur.T, beam_ref.U)

# plot current and deformed config
fig, ax = plot(beam_ref, frenet_serret=True)
plot(beam_cur, fig=fig, ax=ax, frenet_serret=True)

ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')

origin = np.array([0, 0])
xsi = 0.4
point_ref = beam_ref.value(xsi)
point_cur = beam_cur.value(xsi)
ax.arrow(origin[0], origin[1], point_ref[0][0], point_ref[1][0], head_width=0.1, length_includes_head=True)
ax.arrow(origin[0], origin[1], point_cur[0][0], point_cur[1][0], head_width=0.1, length_includes_head=True)
ax.arrow(point_ref[0][0], point_ref[1][0], point_cur[0][0] - point_ref[0][0], point_cur[1][0] - point_ref[1][0],
         head_width=0.1, length_includes_head=True)
ax.arrow(ctrlpts_ref[4][0], ctrlpts_ref[4][1], ctrlpts_cur[4][0] - ctrlpts_ref[4][0],
         ctrlpts_cur[4][1] - ctrlpts_ref[4][1], head_width=0.1, length_includes_head=True)

plt.show()
