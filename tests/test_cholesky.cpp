#include "numlinalg/cholesky.hpp"
#include "numlinalg/matrix.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace std;

using nla::Cholesky;
using nla::Matrix;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        printf("FAIL: %s\n", what);
        ++failures;
    }
}

int main() {
    // 1. Factor a known SPD matrix and verify L L^T == A.
    {
        Matrix a(3, 3);
        a(0, 0) = 4; a(0, 1) = 2; a(0, 2) = -2;
        a(1, 0) = 2; a(1, 1) = 10; a(1, 2) = 4;
        a(2, 0) = -2; a(2, 1) = 4; a(2, 2) = 6;

        const Cholesky chol = nla::cholesky(a);
        const Matrix& l = chol.L;
        check(l.rows() == 3 && l.cols() == 3, "factor has the right shape");

        double err = 0.0;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < 3; ++k) {
                    if (k <= i && k <= j) s += l(i, k) * l(j, k);
                }
                err = max(err, fabs(s - a(i, j)));
            }
        }
        check(err < 1e-12, "L L^T reconstructs A");

        // Strictly lower triangular plus positive diagonal.
        check(l(0, 1) == 0.0 && l(0, 2) == 0.0 && l(1, 2) == 0.0,
              "L is lower triangular");
        check(l(0, 0) > 0 && l(1, 1) > 0 && l(2, 2) > 0,
              "diagonal of L is positive");
    }

    // 2. Solve A x = b and check the residual.
    {
        Matrix a(3, 3);
        a(0, 0) = 4; a(0, 1) = 2; a(0, 2) = -2;
        a(1, 0) = 2; a(1, 1) = 10; a(1, 2) = 4;
        a(2, 0) = -2; a(2, 1) = 4; a(2, 2) = 6;
        const Cholesky chol = nla::cholesky(a);
        const vector<double> x = nla::cholesky_solve(chol, {2, 6, 4});

        // Check A x = b.
        const double r0 = 4 * x[0] + 2 * x[1] - 2 * x[2] - 2;
        const double r1 = 2 * x[0] + 10 * x[1] + 4 * x[2] - 6;
        const double r2 = -2 * x[0] + 4 * x[1] + 6 * x[2] - 4;
        check(fabs(r0) < 1e-12 && fabs(r1) < 1e-12 && fabs(r2) < 1e-12,
              "solution satisfies A x = b");
    }

    // 3. Identity matrix factors to the identity.
    {
        Matrix a(2, 2);
        a(0, 0) = 1; a(1, 1) = 1;
        const Cholesky chol = nla::cholesky(a);
        const Matrix& l = chol.L;
        check(l(0, 0) == 1 && l(1, 1) == 1 && l(1, 0) == 0,
              "identity factors to identity");
    }

    // 4. Symmetric but indefinite matrix is rejected.
    {
        Matrix a(2, 2);
        a(0, 0) = 1; a(0, 1) = 2;
        a(1, 0) = 2; a(1, 1) = 1;  // eigenvalues 3 and -1
        bool threw = false;
        try {
            (void)nla::cholesky(a);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "indefinite matrix rejected");
    }

    // 5. Non-square input is rejected.
    {
        Matrix bad(2, 3);
        bool threw = false;
        try {
            (void)nla::cholesky(bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "non-square matrix rejected");
    }

    // 6. Asymmetric input is rejected instead of silently factoring its
    // lower triangle. The old implementation accepted this and returned a
    // factor for a matrix the caller never passed in.
    {
        Matrix a(2, 2);
        a(0, 0) = 4; a(0, 1) = 2;
        a(1, 0) = -2; a(1, 1) = 10;
        bool threw = false;
        try {
            (void)nla::cholesky(a);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "asymmetric matrix rejected");
    }

    // 7. A symmetric matrix whose symmetry only differs by round-off is
    // accepted (the check uses a relative tolerance).
    {
        Matrix a(2, 2);
        a(0, 0) = 4; a(0, 1) = 2 + 1e-15;
        a(1, 0) = 2; a(1, 1) = 10;
        bool ok = true;
        try {
            (void)nla::cholesky(a);
        } catch (const std::exception&) {
            ok = false;
        }
        check(ok, "round-off-level asymmetry is tolerated");
    }

    if (failures == 0) {
        printf("all tests passed\n");
        return 0;
    }
    printf("%d test(s) failed\n", failures);
    return 1;
}
