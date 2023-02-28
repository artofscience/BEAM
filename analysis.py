from dataclasses import dataclass

import numpy as np
from nurbs import Curve

@dataclass
class Config:
    at = None
    J = None
    an = None
    dat = None


class Beam(Curve):
    R = np.array([[0, -1], [1, 0]])

    def __init__(self, degree=None, ctrlpts=None, knots=None, weights=None):
        super().__init__(degree, ctrlpts, knots, weights)
        self.ref = Config()
        self.cur = Config()
        self.int_pts = np.linspace(0, 1, 11)
        self.set_config(self.ref)
        self.store_basis_functions(self.int_pts)
        self.strain = np.zeros([2, self.int_pts.size], dtype=float)
        self.stress = np.zeros_like(self.strain)

    def set_config(self, config):
        der = self.derivatives(self.int_pts, 2)
        config.at = der[1]
        config.dat = der[2]
        config.J = np.linalg.norm(config.at, axis=0)
        config.an = self.R @ config.at

    def compute_strain(self):

        a1 = self.cur.at
        a2 = self.cur.an
        A1 = self.ref.at
        A2 = self.ref.an
        da1 = self.cur.dat
        dA1 = self.cur.dat

        var = lambda x, y: np.einsum('ij,ij->j', x, y)

        self.strain[0] = 0.5 * (var(a1, a1) - var(A1, A1))  # axial strain
        self.strain[1] = 0.5 * (var(A2, dA1) - var(a2, da1))  # bending strain
        return self.strain

    def compute_stress(self, E=1, A=1, I=1):
        axial = E * A  # axial stiffness
        bending = E * I  # bending stiffness
        self.stress[0] = axial * self.strain[0]
        self.stress[1] = bending * self.strain[1]
        return self.stress

    def compute_derivatives(self):
        self.dRs = np.zeros([self.n + 1, self.int_pts.size], dtype=float)
        self.ddRs = np.zeros([self.n + 1, self.int_pts.size], dtype=float)
        for pt in range(self.int_pts.size):
            for i in range(0, self.n):
                self.dRs[i, pt] = self.ref.J[pt] ** -1 * self.basis_nurbs[1, i, pt]
                self.ddRs[i, pt] = self.basis_nurbs[2, i, pt] / self.ref.J[pt] ** 2 - self.basis_nurbs[1, i, pt] / \
                                   self.ref.J[pt] ** 4 * np.dot(self.ref.at[:, pt], self.ref.dat[:, pt])

    def assembly(self):
        n = 2 * (self.n + 1)
        B = np.zeros([n, 2, self.int_pts.size], dtype=float)
        H = np.zeros([n, n, self.int_pts.size], dtype=float)
        for pt in range(self.int_pts.size):
            at = self.cur.at[:, pt]
            dat = self.cur.dat[:, pt]
            nat = self.cur.J[pt]
            A = np.outer(at, at)
            for i in range(self.n + 1):
                v = 2 * i
                w = v + 2
                B[v:w, 0, pt] += self.dRs[i, pt] * at
                B[v:w, 1, pt] -= self.ddRs[i, pt] * self.cur.an[:, pt]
                B[v:w, 1, pt] -= self.dRs[i, pt] * nat ** -1 * dat @ self.R  # dat transpose?
                B[v:w, 1, pt] += self.dRs[i, pt] * nat ** -3 * dat @ self.R @ np.outer(at, at)  # dat transpose?
                for j in range(self.n + 1):
                    x = 2 * j
                    y = x + 2
                    H[v:w, x:y, pt] += self.ddRs[i, pt] * self.dRs[j, pt] * nat ** -1 * (
                                nat ** -2 * self.R @ A - self.R)
                    H[v:w, x:y, pt] += self.ddRs[j, pt] * self.dRs[i, pt] * nat ** -1 * self.R - nat ** -3 * A @ self.R
                    z = self.dRs[i, pt] * self.dRs[j, pt] * nat ** -3
                    H[v:w, x:y, pt] += z * np.dot(dat, self.R @ at) * np.identity(2)
                    H[v:w, x:y, pt] += z * np.outer(at, dat) @ self.R
                    H[v:w, x:y, pt] += z * 3 * nat ** -2 * np.outer(at, dat) @ self.R @ A
                    H[v:w, x:y, pt] += z * self.R @ np.outer(dat, at)
        self.B = B
        self.H = H
        return B, H

