#include "numlinalg/lu.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace nla {

LU lu_decompose(Matrix a) {
    const std::size_t n = a.rows();
    if (n == 0 || n != a.cols()) {
        throw std::invalid_argument("lu_decompose expects a non-empty square matrix");
    }

    LU lu;
    lu.perm.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        lu.perm[i] = i;
    }

    // Gaussian elimination in place: strict lower triangle becomes L, the
    // upper triangle becomes U.
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t p = k;
        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::abs(a(i, k)) > std::abs(a(p, k))) {
                p = i;
            }
        }
        if (a(p, k) == 0.0) {
            throw std::runtime_error("matrix is singular");
        }
        if (p != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(a(k, j), a(p, j));
            }
            std::swap(lu.perm[k], lu.perm[p]);
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            const double m = a(i, k) / a(k, k);
            a(i, k) = m;
            for (std::size_t j = k + 1; j < n; ++j) {
                a(i, j) -= m * a(k, j);
            }
        }
    }

    lu.L = Matrix::identity(n);
    lu.U = Matrix(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (j < i) {
                lu.L(i, j) = a(i, j);
            } else {
                lu.U(i, j) = a(i, j);
            }
        }
    }
    return lu;
}

std::vector<double> lu_solve(const LU& lu, const std::vector<double>& b) {
    const std::size_t n = lu.L.rows();
    if (b.size() != n) {
        throw std::invalid_argument("right hand side has the wrong length");
    }

    // L y = P b, unit diagonal so no division needed.
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[lu.perm[i]];
        for (std::size_t j = 0; j < i; ++j) {
            s -= lu.L(i, j) * y[j];
        }
        y[i] = s;
    }

    // U x = y
    std::vector<double> x(n);
    for (std::size_t i = n; i-- > 0;) {
        double s = y[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            s -= lu.U(i, j) * x[j];
        }
        x[i] = s / lu.U(i, i);
    }
    return x;
}

}  // namespace nla
