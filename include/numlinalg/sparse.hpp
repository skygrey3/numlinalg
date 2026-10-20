#ifndef NUMLINALG_SPARSE_HPP
#define NUMLINALG_SPARSE_HPP

#include <cstddef>
#include <vector>

using std::size_t;
using std::vector;

namespace nla {

// Compressed sparse row matrix. Only the structural nonzeros are stored;
// entries whose value happens to be zero but occupy a slot are allowed.
class SparseMatrix {
public:
    SparseMatrix() = default;

    // Build from a dense matrix, dropping entries with |value| < threshold.
    SparseMatrix(const class Matrix& dense, double threshold = 0.0);

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // Number of stored entries.
    size_t nonzeros() const { return values_.size(); }

    // Dense access; linear scan within the row, so use for setup, not hot loops.
    double at(size_t i, size_t j) const;

    // y = A x
    vector<double> multiply(const vector<double>& x) const;

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
