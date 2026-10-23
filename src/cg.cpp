#include "numlinalg/cg.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

using namespace std;

namespace nla {

namespace {

double dot(const vector<double>& a, const vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

}  // namespace

CGResult conjugate_gradient(const SparseMatrix& a, const vector<double>& b,
                            vector<double>& x, double tol, size_t max_iter) {
    const size_t n = a.rows();
    if (n == 0 || a.cols() != n) {
        throw invalid_argument("matrix must be square and non-empty");
    }
    if (b.size() != n) {
        throw invalid_argument("right hand side has the wrong length");
    }
    if (max_iter == 0) {
        max_iter = n;
    }

    const double b_norm = sqrt(dot(b, b));
    if (b_norm == 0.0) {
        // The unique solution of A x = 0 for nonsingular A is x = 0.
        x.assign(n, 0.0);
        return CGResult{0, true, 0.0};
    }

    // Warm start: keep a supplied guess, otherwise start from zero. The
    // initial residual is always computed from the actual x, so a supplied
    // guess costs one extra mat-vec but can save many iterations.
    vector<double> r(n);
    if (x.size() != n) {
        x.assign(n, 0.0);
        r = b;
    } else {
        vector<double> ax(n);
        a.multiply_into(x, ax);
        for (size_t i = 0; i < n; ++i) {
            r[i] = b[i] - ax[i];
        }
    }

    vector<double> p = r;
    vector<double> ap(n);  // reused every iteration, no per-iteration allocation
    double rz = dot(r, r);
    double rnorm = sqrt(rz);
    if (rnorm <= tol * b_norm) {
        return CGResult{0, true, rnorm};
    }

    for (size_t it = 1; it <= max_iter; ++it) {
        a.multiply_into(p, ap);

        const double p_ap = dot(p, ap);
        if (p_ap <= 0.0) {
            throw runtime_error(
                "matrix is not positive definite (p^T A p <= 0)");
        }

        const double alpha = rz / p_ap;
        for (size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        const double rr = dot(r, r);
        rnorm = sqrt(rr);
        if (rnorm <= tol * b_norm) {
            return CGResult{it, true, rnorm};
        }

        const double beta = rr / rz;
        for (size_t i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        rz = rr;
    }

    return CGResult{max_iter, false, rnorm};
}

}  // namespace nla
