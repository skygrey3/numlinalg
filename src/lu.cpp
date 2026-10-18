#include "numlinalg/lu.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

using namespace std;

namespace nla {

LU lu_decompose(Matrix a) {
    const size_t n = a.rows();
    if (n == 0 || n != a.cols()) {
        throw invalid_argument("lu_decompose expects a non-empty square matrix");
    }

    LU lu;
    lu.perm.resize(n);
    for (size_t i = 0; i < n; ++i) {
        lu.perm[i] = i;
    }

    // Gaussian elimination in place: strict lower triangle becomes L, the
    // upper triangle becomes U.
    for (size_t k = 0; k < n; ++k) {
        size_t p = k;
        for (size_t i = k + 1; i < n; ++i) {
            if (abs(a(i, k)) > abs(a(p, k))) {
                p = i;
            }
        }
        if (a(p, k) == 0.0) {
            throw runtime_error("matrix is singular");
        }
        if (p != k) {
            for (size_t j = 0; j < n; ++j) {
                swap(a(k, j), a(p, j));
            }
            swap(lu.perm[k], lu.perm[p]);
        }
        for (size_t i = k + 1; i < n; ++i) {
            const double m = a(i, k) / a(k, k);
            a(i, k) = m;
            for (size_t j = k + 1; j < n; ++j) {
                a(i, j) -= m * a(k, j);
            }
        }
    }

    lu.L = Matrix::identity(n);
    lu.U = Matrix(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (j < i) {
                lu.L(i, j) = a(i, j);
            } else {
                lu.U(i, j) = a(i, j);
            }
        }
    }
    return lu;
}

vector<double> lu_solve(const LU& lu, const vector<double>& b) {
    const size_t n = lu.L.rows();
    if (b.size() != n) {
        throw invalid_argument("right hand side has the wrong length");
    }

    // L y = P b, unit diagonal so no division needed.
    vector<double> y(n);
    for (size_t i = 0; i < n; ++i) {
        double s = b[lu.perm[i]];
        for (size_t j = 0; j < i; ++j) {
            s -= lu.L(i, j) * y[j];
        }
        y[i] = s;
    }

    // U x = y
    vector<double> x(n);
    for (size_t i = n; i-- > 0;) {
        double s = y[i];
        for (size_t j = i + 1; j < n; ++j) {
            s -= lu.U(i, j) * x[j];
        }
        x[i] = s / lu.U(i, i);
    }
    return x;
}

}  // namespace nla
