#include "numlinalg/eigen.hpp"

#include <cmath>
#include <stdexcept>

#include "numlinalg/matrix.hpp"

using namespace std;

namespace nla {

// One cyclic sweep: rotate every off-diagonal pair (p, q), p < q, once.
// Returns the sum of squared off-diagonal magnitudes after the sweep.
static double jacobi_sweep(Matrix& a, const size_t n) {
    for (size_t p = 0; p + 1 < n; ++p) {
        for (size_t q = p + 1; q < n; ++q) {
            const double apq = a(p, q);
            if (apq == 0.0) {
                continue;
            }

            const double app = a(p, p);
            const double aqq = a(q, q);
            const double theta = (aqq - app) / (2.0 * apq);
            // t = tan(phi) with |t| <= 1: the smaller rotation angle.
            const double t =
                (theta >= 0.0 ? 1.0 : -1.0) /
                (fabs(theta) + sqrt(theta * theta + 1.0));
            const double c = 1.0 / sqrt(t * t + 1.0);
            const double s = t * c;

            // Update rows and columns p and q of A = J^T A J.
            for (size_t i = 0; i < n; ++i) {
                if (i != p && i != q) {
                    const double aip = a(i, p);
                    const double aiq = a(i, q);
                    a(i, p) = c * aip - s * aiq;
                    a(p, i) = a(i, p);
                    a(i, q) = s * aip + c * aiq;
                    a(q, i) = a(i, q);
                }
            }
            a(p, p) = app - t * apq;
            a(q, q) = aqq + t * apq;
            a(p, q) = 0.0;
            a(q, p) = 0.0;
        }
    }

    double off = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double v = a(i, j);
            off += v * v;
        }
    }
    return off;
}

vector<double> jacobi_eigenvalues(Matrix& a) {
    const size_t n = a.rows();
    if (n == 0 || a.cols() != n) {
        throw invalid_argument("matrix must be square and non-empty");
    }

    const double eps = 1e-14;
    for (int sweep = 0; sweep < 100; ++sweep) {
        const double off = jacobi_sweep(a, n);
        if (off <= eps * static_cast<double>(n)) {
            vector<double> eigenvalues(n);
            for (size_t i = 0; i < n; ++i) {
                eigenvalues[i] = a(i, i);
            }
            return eigenvalues;
        }
    }
    throw runtime_error("jacobi iteration did not converge in 100 sweeps");
}

}  // namespace nla
