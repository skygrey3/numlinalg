#include "numlinalg/cholesky.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "numlinalg/matrix.hpp"

using namespace std;

namespace nla {

Cholesky cholesky(const Matrix& a) {
    const size_t n = a.rows();
    if (n == 0 || a.cols() != n) {
        throw invalid_argument("matrix must be square and non-empty");
    }

    // Cholesky needs A = A^T. Check it explicitly: a non-symmetric matrix can
    // still produce a "successful" factorization of its lower triangle, which
    // is a silent wrong answer. The tolerance is relative to the largest entry
    // so ordinary round-off in a symmetric matrix does not trip the check.
    double scale = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            scale = max(scale, abs(a.unchecked(i, j)));
        }
    }
    const double tol = 1e-12 * max(1.0, scale);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (abs(a.unchecked(i, j) - a.unchecked(j, i)) > tol) {
                throw invalid_argument("matrix must be symmetric");
            }
        }
    }

    Cholesky chol;
    chol.L = Matrix(n, n);
    for (size_t j = 0; j < n; ++j) {
        // Diagonal entry: a_jj - sum of squares of row j so far.
        double s = a.unchecked(j, j);
        for (size_t k = 0; k < j; ++k) {
            const double ljk = chol.L.unchecked(j, k);
            s -= ljk * ljk;
        }
        if (s <= 0.0) {
            throw runtime_error("matrix is not positive definite");
        }
        chol.L.unchecked(j, j) = sqrt(s);

        // Sub-diagonal column j.
        for (size_t i = j + 1; i < n; ++i) {
            double v = a.unchecked(i, j);
            for (size_t k = 0; k < j; ++k) {
                v -= chol.L.unchecked(i, k) * chol.L.unchecked(j, k);
            }
            chol.L.unchecked(i, j) = v / chol.L.unchecked(j, j);
        }
    }
    return chol;
}

vector<double> cholesky_solve(const Cholesky& chol, const vector<double>& b) {
    const Matrix& l = chol.L;
    const size_t n = l.rows();
    if (n == 0 || l.cols() != n || b.size() != n) {
        throw invalid_argument("factor or right-hand side has the wrong size");
    }

    // Forward substitution: L y = b.
    vector<double> y(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (size_t k = 0; k < i; ++k) {
            s -= l.unchecked(i, k) * y[k];
        }
        y[i] = s / l.unchecked(i, i);
    }

    // Back substitution: L^T x = y.
    vector<double> x(n, 0.0);
    for (size_t ii = 0; ii < n; ++ii) {
        const size_t i = n - 1 - ii;
        double s = y[i];
        for (size_t k = i + 1; k < n; ++k) {
            s -= l.unchecked(k, i) * x[k];
        }
        x[i] = s / l.unchecked(i, i);
    }
    return x;
}

}  // namespace nla
