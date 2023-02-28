from dataclasses import dataclass

import numpy as np
from nurbs import Curve
from scipy import integrate

from scipy import sparse
import warnings

from bspline_basis_functions import basis_polynomials, basis_polynomials_derivatives

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
        self.quad = Quadrature(self.nks)
        self.int_pts = self.quad.pts
        self.int_w = self.quad.w
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

    def bmatrix_axial(self, u):
        n = 2 * (self.n + 1)
        B = np.zeros(2 * (self.n + 1), dtype=float)
        for i in range(self.n + 1):
            B[2 * i:2 * i + 2, 0] += self.basis_nurbs[1, i]
        return B


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


    def arclength(self, u1=0.00, u2=1.00):

        """ Compute the arc length of a parametric curve in the interval [u1,u2] using numerical quadrature

        The definition of the arc length is given by equation 10.3 (Farin's textbook)

        Parameters
        ----------
        u1 : scalar
            Lower limit of integration for the arc length computation

        u2 : scalar
            Upper limit of integration for the arc length computation

        Returns
        -------
        L : scalar
            Arc length of NURBS curve in the interval [u1, u2]

        """

        # Compute the arc length of C(t) in the interval [u1, u2] by numerical integration
        arclength = integrate.quadrature(self.arclength_differential, u1, u2)[0]

        return arclength

    # Compute the arc length differential analytically
    def arclength_differential(self, u):
        dCdu = self.derivatives(u, up_to_order=1)[1, ...]
        dLdu = np.sqrt(np.sum(dCdu ** 2, axis=0))  # dL/du = [(dx_0/du)^2 + ... + (dx_n/du)^2]^(1/2)
        return dLdu


class Quadrature:
    def __init__(self, n):
        self.pts = 0
        self.w = 0
        self.n = n

        if self.n == 2 or self.n == 1:
            self.pts = [0.084001595740497, 0.353667436436311, 0.646332563563689, 0.915998404259503]
            self.w = [0.204166185672591, 0.295833814327409, 0.295833814327409, 0.204166185672591]
        elif self.n == 3:
            self.pts = [0.055307959538964, 0.232008127012761, 0.410698113579587, 0.589301886420413, 0.767991872987239,
                        0.944692040461036]
            self.w = [0.134383670129084, 0.190719210529352, 0.174897119341564, 0.174897119341564, 0.190719210529352,
                      0.134383670129084]
        elif self.n == 4:
            self.pts = [0.042302270496914, 0.178540270746368, 0.335067537628328, 0.500000000000000, 0.664932462371672,
                        0.821459729253632, 0.957697729503086]
            self.w = [0.102836135188702, 0.151209936088574, 0.165363166232141, 0.161181524981166, 0.165363166232141,
                      0.151209936088574, 0.102836135188702]
        elif self.n == 5:
            self.pts = [0.033825647049693, 0.142739413107187, 0.267383546533900, 0.393434348254817, 0.500000000000000,
                        0.606565651745183, 0.732616453466100, 0.857260586892813, 0.966174352950307]
            self.w = [0.082228488484279, 0.120781740225645, 0.131305988133937, 0.112350449822805, 0.106666666666667,
                      0.112350449822805, 0.131305988133937, 0.120781740225645, 0.082228488484279]
        self.pts = np.asarray(self.pts)
        self.w = np.asarray(self.w)
