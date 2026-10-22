#ifndef NUMLINALG_MATRIX_HPP
#define NUMLINALG_MATRIX_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

using std::abs;
using std::invalid_argument;
using std::max;
using std::out_of_range;
using std::size_t;
using std::vector;

namespace nla {

// Dense matrix with row-major storage.
class Matrix {
public:
    Matrix() = default;

    Matrix(size_t rows, size_t cols, double fill = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    static Matrix identity(size_t n) {
        Matrix I(n, n);
        for (size_t i = 0; i < n; ++i) {
            I(i, i) = 1.0;
        }
        return I;
    }

    // Checked access: throws std::out_of_range on an invalid index. The checks
    // are always compiled in, so a release build cannot silently corrupt
    // memory on a bad index.
    double& operator()(size_t i, size_t j) {
        if (i >= rows_ || j >= cols_) {
            throw out_of_range("matrix index out of range");
        }
        return data_[i * cols_ + j];
    }

    double operator()(size_t i, size_t j) const {
        if (i >= rows_ || j >= cols_) {
            throw out_of_range("matrix index out of range");
        }
        return data_[i * cols_ + j];
    }

    // Unchecked access for hot loops whose indices are already known valid.
    // Callers are responsible for the bounds; this is the escape hatch for
    // code where the per-access check measurably matters.
    double& unchecked(size_t i, size_t j) { return data_[i * cols_ + j]; }

    double unchecked(size_t i, size_t j) const {
        return data_[i * cols_ + j];
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    Matrix transpose() const {
        Matrix t(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                t(j, i) = (*this)(i, j);
            }
        }
        return t;
    }

    Matrix& operator+=(const Matrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw invalid_argument("matrix dimensions do not match for +=");
        }
        for (size_t k = 0; k < data_.size(); ++k) {
            data_[k] += other.data_[k];
        }
        return *this;
    }

private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    vector<double> data_;
};

inline Matrix operator+(const Matrix& a, const Matrix& b) {
    Matrix c = a;
    c += b;
    return c;
}

inline Matrix operator*(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows()) {
        throw invalid_argument("matrix dimensions do not match for *");
    }
    Matrix c(a.rows(), b.cols());
    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t k = 0; k < a.cols(); ++k) {
            const double aik = a.unchecked(i, k);
            for (size_t j = 0; j < b.cols(); ++j) {
                c.unchecked(i, j) += aik * b.unchecked(k, j);
            }
        }
    }
    return c;
}

// Relative comparison, the tolerance is scaled by the magnitudes involved.
inline bool approx_equal(double a, double b, double tol = 1e-10) {
    const double scale = max({1.0, abs(a), abs(b)});
    return abs(a - b) <= tol * scale;
}

}  // namespace nla

#endif
