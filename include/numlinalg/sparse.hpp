#ifndef NUMLINALG_SPARSE_HPP
#define NUMLINALG_SPARSE_HPP

#include <cstddef>
#include <vector>

using std::size_t;
using std::vector;

namespace nla {

// Compressed sparse row matrix. Construction from a dense matrix keeps only
// entries whose magnitude is strictly greater than threshold (default 0, so
// only exact zeros are dropped); the threshold is a construction-time filter,
// not a property of the storage format. Explicitly stored zeros are therefore
// not preserved.
class SparseMatrix {
public:
    SparseMatrix() = default;

    // Build from a dense matrix, dropping entries with |value| <= threshold.
    SparseMatrix(const class Matrix& dense, double threshold = 0.0);

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // Number of stored entries.
    size_t nonzeros() const { return values_.size(); }

    // Dense access; linear scan within the row, so use for setup, not hot loops.
    double at(size_t i, size_t j) const;

    // y = A x. multiply_into reuses the caller's buffer (resized and zeroed),
    // which avoids a heap allocation per iteration in iterative solvers.
    vector<double> multiply(const vector<double>& x) const;
    void multiply_into(const vector<double>& x, vector<double>& y) const;

    // Number of stored entries in row i.
    size_t row_nnz(size_t i) const {
        return row_ptr_[i + 1] - row_ptr_[i];
    }

private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    vector<size_t> row_ptr_;   // size rows_ + 1
    vector<size_t> col_idx_;
    vector<double> values_;
};

}  // namespace nla

#endif
