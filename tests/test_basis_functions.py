import numpy as np
import pytest
from nurbs import basis_polynomials, basis_polynomials_derivatives


# @pytest.mark.parametrize("p", [0, 1, 2, 3, 4])
def test_partition_of_unity():
    """ Test the partition of unity property of the basis polynomials """
    n = 4
    p = 3
    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))
    u = np.linspace(0, 1, 101)
    N_basis = basis_polynomials(n, p, U, u)
    assert np.sum(N_basis, axis=0) == pytest.approx(1, 1e-6)


class TestExample2_2:
    """ Test the basis function value against a known example (Ex2.2 from the NURBS book) """

    @staticmethod
    def analytic_polynomials(u):
        N = np.zeros((8, u.size), dtype=float)
        for i, u in enumerate(u):
            N02 = (1 - u) ** 2 * (0 <= u < 1)
            N12 = (2 * u - 3 / 2 * u ** 2) * (0 <= u < 1) + (1 / 2 * (2 - u) ** 2) * (1 <= u < 2)
            N22 = (1 / 2 * u ** 2) * (0 <= u < 1) + (-3 / 2 + 3 * u - u ** 2) * (1 <= u < 2) + (
                    1 / 2 * (3 - u) ** 2) * (2 <= u < 3)
            N32 = (1 / 2 * (u - 1) ** 2) * (1 <= u < 2) + (-11 / 2 + 5 * u - u ** 2) * (2 <= u < 3) + (
                    1 / 2 * (4 - u) ** 2) * (3 <= u < 4)
            N42 = (1 / 2 * (u - 2) ** 2) * (2 <= u < 3) + (-16 + 10 * u - 3 / 2 * u ** 2) * (3 <= u < 4)
            N52 = (u - 3) ** 2 * (3 <= u < 4) + (5 - u) ** 2 * (4 <= u < 5)
            N62 = (2 * (u - 4) * (5 - u)) * (4 <= u < 5)
            N72 = (u - 4) ** 2 * (4 <= u <= 5)
            N[:, i] = np.asarray([N02, N12, N22, N32, N42, N52, N62, N72])
        return N

    def test_basis_functions(self):
        U = np.asarray([0.00, 0.00, 0.00, 1.00, 2.00, 3.00, 4.00, 4.00, 5.00, 5.00, 5.00])
        uu = np.linspace(0, 5, 21)

        # Evaluate the polynomials numerically
        N_basis = basis_polynomials(7, 2, U, uu)

        # Evaluate the polynomials analytically
        N_analytic = self.analytic_polynomials(uu)

        assert np.allclose(N_basis, N_analytic)
