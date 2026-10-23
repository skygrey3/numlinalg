#include "numlinalg/eigen.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "numlinalg/matrix.hpp"

using namespace std;

namespace nla {

// One cyclic sweep: rotate every off-diagonal pair (p, q), p < q, once.
// Returns the sum of squared off-diagonal magnitudes after the sweep.
static double jacobi_sweep(Matrix& a, const size_t n) {
    for (size_t p = 0; p + 1 < n; ++p) {
        for (size_t q = p + 1; q < n; ++q) {
            const double apq = a.unchecked(p, q);
            if (apq == 0.0) {
                continue;
            }

            const double app = a.unchecked(p, p);
            const double aqq = a.unchecked(q, q);
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
                    const double aip = a.unchecked(i, p);
                    const double aiq = a.unchecked(i, q);
                    a.unchecked(i, p) = c * aip - s * aiq;
                    a.unchecked(p, i) = a.unchecked(i, p);
                    a.unchecked(i, q) = s * aip + c * aiq;
                    a.unchecked(q, i) = a.unchecked(i, q);
                }
            }
            a.unchecked(p, p) = app - t * apq;
            a.unchecked(q, q) = aqq + t * apq;
            a.unchecked(p, q) = 0.0;
            a.unchecked(q, p) = 0.0;
        }
    }

    double off = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double v = a.unchecked(i, j);
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

    // Rescale by the largest entry so the relative test and the squared norm
    // cannot overflow or underflow at extreme scales. Jacobi is homogeneous,
    // so the eigenvalues are scaled back at the end.
    double max_abs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            max_abs = max(max_abs, abs(a.unchecked(i, j)));
        }
    }
    if (max_abs == 0.0) {
        return vector<double>(n, 0.0);
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            a.unchecked(i, j) /= max_abs;
        }
    }

    // Frobenius norm of the scaled input. Orthogonal similarity
    // transformations preserve it, so compute it once: the convergence test
    // compares the off-diagonal norm against this fixed scale.
    double norm_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double v = a.unchecked(i, j);
            norm_sq += v * v;
        }
    }
    const double eps = 1e-12;
    const double threshold = eps * eps * norm_sq;

    for (int sweep = 0; sweep < 100; ++sweep) {
        const double off = jacobi_sweep(a, n);
        if (off <= threshold) {
            vector<double> eigenvalues(n);
            for (size_t i = 0; i < n; ++i) {
                eigenvalues[i] = a.unchecked(i, i) * max_abs;
            }
            return eigenvalues;
        }
    }
    throw runtime_error("jacobi iteration did not converge in 100 sweeps");
}

}  // namespace nla
