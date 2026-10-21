#include "numlinalg/cholesky.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "numlinalg/matrix.hpp"

using namespace std;

namespace nla {

vector<double> cholesky(const Matrix& a) {
    const size_t n = a.rows();
    if (n == 0 || a.cols() != n) {
        throw invalid_argument("matrix must be square and non-empty");
    }

    vector<double> l(n * n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        // Diagonal entry: a_jj - sum of squares of row j so far.
        double s = a(j, j);
        for (size_t k = 0; k < j; ++k) {
            s -= l[j * n + k] * l[j * n + k];
        }
        if (s <= 0.0) {
            throw runtime_error("matrix is not positive definite");
        }
        l[j * n + j] = sqrt(s);

        // Sub-diagonal column j.
        for (size_t i = j + 1; i < n; ++i) {
            double v = a(i, j);
            for (size_t k = 0; k < j; ++k) {
                v -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = v / l[j * n + j];
        }
    }
    return l;
}

vector<double> cholesky_solve(const vector<double>& l,
                              const vector<double>& b) {
    const size_t m = l.size();
    // L is n x n stored row-major, so n = sqrt(m); reject non-square input.
    const size_t n = static_cast<size_t>(sqrt(static_cast<double>(m)));
    if (n == 0 || n * n != m || b.size() != n) {
        throw invalid_argument("factor or right-hand side has the wrong size");
    }

    // Forward substitution: L y = b.
    vector<double> y(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (size_t k = 0; k < i; ++k) {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }

    // Back substitution: L^T x = y.
    vector<double> x(n, 0.0);
    for (size_t ii = 0; ii < n; ++ii) {
        const size_t i = n - 1 - ii;
        double s = y[i];
        for (size_t k = i + 1; k < n; ++k) {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    return x;
}

}  // namespace nla
