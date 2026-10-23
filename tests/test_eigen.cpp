#include "numlinalg/eigen.hpp"
#include "numlinalg/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace std;

using nla::Matrix;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        printf("FAIL: %s\n", what);
        ++failures;
    }
}

// Sum of squared off-diagonal entries: the quantity the sweeps drive to zero.
static double off_norm(const Matrix& a) {
    double off = 0.0;
    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = i + 1; j < a.rows(); ++j) {
            off += a(i, j) * a(i, j);
        }
    }
    return off;
}

int main() {
    // 1. Diagonal matrix: eigenvalues are the diagonal entries.
    Matrix diag(3, 3);
    diag(0, 0) = 7; diag(1, 1) = -2; diag(2, 2) = 5;
    {
        Matrix a = diag;
        const vector<double> eig = nla::jacobi_eigenvalues(a);
        vector<double> expected = {7, -2, 5};
        vector<double> got = eig;
        sort(got.begin(), got.end());
        sort(expected.begin(), expected.end());
        check(fabs(got[0] + 2) < 1e-12 && fabs(got[1] - 5) < 1e-12 &&
                  fabs(got[2] - 7) < 1e-12,
              "diagonal matrix eigenvalues");
        check(off_norm(a) < 1e-24, "diagonal matrix stays diagonal");
    }

    // 2. The 2x2 example from the reference derivation.
    {
        Matrix a(2, 2);
        a(0, 0) = 4; a(0, 1) = 1;
        a(1, 0) = 1; a(1, 1) = 3;
        const vector<double> eig = nla::jacobi_eigenvalues(a);
        // char poly: (4-l)(3-l) - 1 = 0 -> l = 3.5 +- sqrt(1.25).
        // The diagonal holds them in arbitrary order, so accept either.
        const double r = sqrt(1.25);
        const bool ok = (fabs(eig[0] - (3.5 - r)) < 1e-12 &&
                         fabs(eig[1] - (3.5 + r)) < 1e-12) ||
                        (fabs(eig[0] - (3.5 + r)) < 1e-12 &&
                         fabs(eig[1] - (3.5 - r)) < 1e-12);
        check(ok, "2x2 eigenvalues match the closed form");
    }

    // 3. Rotation matrix: eigenvalues 1 and -1, eigenvectors at 45 degrees.
    {
        const double c = cos(0.7);
        const double s = sin(0.7);
        Matrix a(2, 2);
        a(0, 0) = c * c - s * s;
        a(0, 1) = 2 * c * s;
        a(1, 0) = 2 * c * s;
        a(1, 1) = s * s - c * c;
        const vector<double> eig = nla::jacobi_eigenvalues(a);
        check(fabs(fabs(eig[0]) - 1.0) < 1e-12 &&
                  fabs(fabs(eig[1]) - 1.0) < 1e-12,
              "reflection-like symmetric matrix has eigenvalues +-1");
    }

    // 4. 4x4 symmetric with known integer eigenvalues 1, 2, 3, 4.
    {
        Matrix v(4, 4);  // eigenvector matrix, orthogonal
        v(0, 0) = 0.5; v(0, 1) = 0.5; v(0, 2) = 0.5; v(0, 3) = 0.5;
        v(1, 0) = 0.5; v(1, 1) = -0.5; v(1, 2) = 0.5; v(1, 3) = -0.5;
        v(2, 0) = 0.5; v(2, 1) = 0.5; v(2, 2) = -0.5; v(2, 3) = -0.5;
        v(3, 0) = 0.5; v(3, 1) = -0.5; v(3, 2) = -0.5; v(3, 3) = 0.5;
        Matrix d(4, 4);
        d(0, 0) = 1; d(1, 1) = 2; d(2, 2) = 3; d(3, 3) = 4;
        // a = V D V^T is symmetric with eigenvalues 1..4.
        Matrix a(4, 4);
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                double s = 0.0;
                for (size_t kk = 0; kk < 4; ++kk) {
                    s += v(i, kk) * d(kk, kk) * v(j, kk);
                }
                a(i, j) = s;
            }
        }
        const vector<double> eig = nla::jacobi_eigenvalues(a);
        vector<double> got = eig;
        sort(got.begin(), got.end());
        const double tol = 1e-10;
        check(fabs(got[0] - 1) < tol && fabs(got[1] - 2) < tol &&
                  fabs(got[2] - 3) < tol && fabs(got[3] - 4) < tol,
              "4x4 Hadamard-built matrix recovers eigenvalues 1..4");
    }

    // 5. Trace is preserved: sum of eigenvalues equals trace of the input.
    {
        Matrix a = diag;
        const vector<double> eig = nla::jacobi_eigenvalues(a);
        check(fabs(eig[0] + eig[1] + eig[2] - 10.0) < 1e-12,
              "trace is preserved");
    }

    // 6. Scale invariance: multiplying A by 1e160 multiplies every eigenvalue
    // by 1e160. An absolute convergence threshold on the off-diagonal sum
    // would stop before the off-diagonal part is negligible, and squaring the
    // Frobenius norm would overflow; the relative criterion and the internal
    // rescaling handle both.
    {
        Matrix base(2, 2);
        base(0, 0) = 4; base(0, 1) = 1;
        base(1, 0) = 1; base(1, 1) = 3;
        const double scale = 1e160;
        Matrix a(2, 2);
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                a(i, j) = scale * base(i, j);
            }
        }
        const vector<double> eig = nla::jacobi_eigenvalues(a);
        const double r = sqrt(1.25);
        const double lo = scale * (3.5 - r);
        const double hi = scale * (3.5 + r);
        const bool ok = (fabs(eig[0] - lo) <= 1e-9 * hi &&
                         fabs(eig[1] - hi) <= 1e-9 * hi) ||
                        (fabs(eig[0] - hi) <= 1e-9 * hi &&
                         fabs(eig[1] - lo) <= 1e-9 * hi);
        check(ok, "eigenvalues scale linearly with the matrix");
    }

    // 7. Non-square input is rejected.
    {
        Matrix bad(2, 3);
        bool threw = false;
        try {
            (void)nla::jacobi_eigenvalues(bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "non-square matrix rejected");
    }

    if (failures == 0) {
        printf("all tests passed\n");
        return 0;
    }
    printf("%d test(s) failed\n", failures);
    return 1;
}
