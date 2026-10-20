#include "numlinalg/cg.hpp"
#include "numlinalg/matrix.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace std;

using nla::Matrix;
using nla::SparseMatrix;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        printf("FAIL: %s\n", what);
        ++failures;
    }
}

// The standard 5-point stencil matrix for the 2D Poisson problem on a k x k
// interior grid: identity and fourth neighbors on the diagonal, 1 elsewhere.
static Matrix poisson_matrix(size_t k) {
    const size_t n = k * k;
    Matrix a(n, n);
    for (size_t i = 0; i < n; ++i) {
        a(i, i) = 4.0;
    }
    for (size_t ix = 0; ix < k; ++ix) {
        for (size_t iy = 0; iy < k; ++iy) {
            const size_t i = iy * k + ix;
            if (ix > 0) {
                a(i, i - 1) = -1.0;
            }
            if (ix + 1 < k) {
                a(i, i + 1) = -1.0;
            }
            if (iy > 0) {
                a(i, i - k) = -1.0;
            }
            if (iy + 1 < k) {
                a(i, i + k) = -1.0;
            }
        }
    }
    return a;
}

int main() {
    // Solve the 12x12 interior Poisson system exactly via CG on the dense
    // matrix converted to CSR. Since A is positive definite and the Krylov
    // space has dimension n, CG must terminate within n iterations.
    const size_t k = 12;
    const Matrix dense = poisson_matrix(k);
    const size_t n = dense.rows();

    const SparseMatrix a(dense);
    check(a.nonzeros() == 4 * k * (k - 1) + n,
          "stencil stores the expected nonzeros");

    // Exact solution: a mix of two Dirichlet eigenmodes of the 12-point
    // interior grid (modes j = 1 and j = 3). Not a single eigenmode, so CG
    // needs several iterations rather than one.
    vector<double> x_exact(n);
    for (size_t ix = 0; ix < k; ++ix) {
        for (size_t iy = 0; iy < k; ++iy) {
            const double u1 =
                sin(M_PI * static_cast<double>(ix + 1) / 13.0) *
                sin(M_PI * static_cast<double>(iy + 1) / 13.0);
            const double u3 =
                sin(3.0 * M_PI * static_cast<double>(ix + 1) / 13.0) *
                sin(3.0 * M_PI * static_cast<double>(iy + 1) / 13.0);
            x_exact[iy * k + ix] = 2.0 * u1 + 0.5 * u3;
        }
    }
    vector<double> b(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < n; ++j) {
            s += dense(i, j) * x_exact[j];
        }
        b[i] = s;
    }

    vector<double> x;
    const size_t used = nla::conjugate_gradient(a, b, x, 1e-12, 100);
    check(used <= n, "CG terminates within n iterations");

    bool converged = true;
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double err = fabs(x[i] - x_exact[i]);
        if (err > max_err) {
            max_err = err;
        }
        if (err > 1e-8) {
            converged = false;
            break;
        }
    }
    check(converged, "CG recovers the exact solution to 1e-8");
    printf("poisson 12x12: %zu iterations, max error %.2e\n", used, max_err);

    // Also solves a small non-Poisson SPD system.
    Matrix s(3, 3);
    s(0, 0) = 5; s(0, 1) = 1; s(0, 2) = 0;
    s(1, 0) = 1; s(1, 1) = 6; s(1, 2) = 2;
    s(2, 0) = 0; s(2, 1) = 2; s(2, 2) = 7;
    const SparseMatrix sp(s);
    const vector<double> xe = {1.0, 2.0, 3.0};
    vector<double> sb(3, 0.0);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            sb[i] += s(i, j) * xe[j];
        }
    }
    vector<double> sx;
    (void)nla::conjugate_gradient(sp, sb, sx, 1e-12, 50);
    check(fabs(sx[0] - 1.0) < 1e-9 && fabs(sx[1] - 2.0) < 1e-9 &&
              fabs(sx[2] - 3.0) < 1e-9,
          "CG solves a small SPD system exactly");

    // Indefinite matrix is rejected: [[4,1],[1,0]] has a zero eigenvalue.
    Matrix bad(2, 2);
    bad(0, 0) = 4; bad(0, 1) = 1;
    bad(1, 0) = 1; bad(1, 1) = 0;
    const SparseMatrix bads(bad);
    vector<double> bx = {1.0, 1.0};
    vector<double> bxx;
    bool threw = false;
    try {
        (void)nla::conjugate_gradient(bads, bx, bxx);
    } catch (const runtime_error&) {
        threw = true;
    }
    check(threw, "indefinite system is rejected");

    if (failures == 0) {
        printf("all tests passed\n");
        return 0;
    }
    printf("%d test(s) failed\n", failures);
    return 1;
}
