#include "numlinalg/sparse.hpp"

#include <stdexcept>

#include "numlinalg/matrix.hpp"

using namespace std;

namespace nla {

// Build from a dense matrix, dropping entries whose magnitude is <= threshold.
// threshold 0 keeps every nonzero value.
SparseMatrix::SparseMatrix(const Matrix& dense, double threshold)
    : rows_(dense.rows()), cols_(dense.cols()) {
    if (rows_ == 0 || cols_ == 0) {
        throw invalid_argument(
            "cannot build a sparse matrix from an empty matrix");
    }

    // First pass: count entries kept per row.
    row_ptr_.assign(rows_ + 1, 0);
    size_t total = 0;
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            const double v = dense(i, j);
            const double mag = v < 0.0 ? -v : v;
            if (mag > threshold) {
                ++total;
            }
        }
        row_ptr_[i + 1] = total;
    }

    // Second pass: fill the column indices and values.
    col_idx_.resize(total);
    values_.resize(total);
    size_t k = 0;
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            const double v = dense(i, j);
            const double mag = v < 0.0 ? -v : v;
            if (mag > threshold) {
                col_idx_[k] = j;
                values_[k] = v;
                ++k;
            }
        }
    }
}

double SparseMatrix::at(size_t i, size_t j) const {
    if (i >= rows_ || j >= cols_) {
        throw out_of_range("sparse matrix index out of range");
    }
    for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
        if (col_idx_[k] == j) {
            return values_[k];
        }
    }
    return 0.0;
}

void SparseMatrix::multiply_into(const vector<double>& x,
                                 vector<double>& y) const {
    if (x.size() != cols_) {
        throw invalid_argument("vector length does not match matrix columns");
    }
    y.assign(rows_, 0.0);
    for (size_t i = 0; i < rows_; ++i) {
        double s = 0.0;
        for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            s += values_[k] * x[col_idx_[k]];
        }
        y[i] = s;
    }
}

vector<double> SparseMatrix::multiply(const vector<double>& x) const {
    vector<double> y;
    multiply_into(x, y);
    return y;
}

}  // namespace nla
