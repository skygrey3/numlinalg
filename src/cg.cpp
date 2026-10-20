#include "numlinalg/cg.hpp"

#include <stdexcept>

using namespace std;

namespace nla {

size_t conjugate_gradient(const SparseMatrix& a, const vector<double>& b,
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

    x.assign(n, 0.0);

    // x0 = 0, so r0 = b - A x0 = b.
    vector<double> r = b;
    vector<double> p = r;
    double rz = 0.0;
    for (size_t i = 0; i < n; ++i) {
        rz += r[i] * r[i];
    }
    const double b_norm = sqrt(rz);
    if (b_norm == 0.0) {
        return 0;
    }

    for (size_t it = 1; it <= max_iter; ++it) {
        const vector<double> ap = a.multiply(p);

        double p_ap = 0.0;
        for (size_t i = 0; i < n; ++i) {
            p_ap += p[i] * ap[i];
        }
        if (p_ap <= 0.0) {
            throw runtime_error(
                "matrix is not positive definite (p^T A p <= 0)");
        }

        const double alpha = rz / p_ap;

        for (size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        double rr = 0.0;
        for (size_t i = 0; i < n; ++i) {
            rr += r[i] * r[i];
        }
        if (sqrt(rr) <= tol * b_norm) {
            return it;
        }

        const double beta = rr / rz;
        for (size_t i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        rz = rr;
    }
    return max_iter;
}

}  // namespace nla
