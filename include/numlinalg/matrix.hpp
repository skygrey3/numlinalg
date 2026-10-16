#ifndef NUMLINALG_MATRIX_HPP
#define NUMLINALG_MATRIX_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace nla {

// Dense matrix with row-major storage.
class Matrix {
public:
    Matrix() = default;

    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    static Matrix identity(std::size_t n) {
        Matrix I(n, n);
        for (std::size_t i = 0; i < n; ++i) {
            I(i, i) = 1.0;
        }
        return I;
    }

    double& operator()(std::size_t i, std::size_t j) {
        assert(i < rows_ && j < cols_);
        return data_[i * cols_ + j];
    }

    double operator()(std::size_t i, std::size_t j) const {
        assert(i < rows_ && j < cols_);
        return data_[i * cols_ + j];
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    Matrix transpose() const {
        Matrix t(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                t(j, i) = (*this)(i, j);
            }
        }
        return t;
    }

    Matrix& operator+=(const Matrix& other) {
        assert(rows_ == other.rows_ && cols_ == other.cols_);
        for (std::size_t k = 0; k < data_.size(); ++k) {
            data_[k] += other.data_[k];
        }
        return *this;
    }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

inline Matrix operator+(const Matrix& a, const Matrix& b) {
    Matrix c = a;
    c += b;
    return c;
}

inline Matrix operator*(const Matrix& a, const Matrix& b) {
    assert(a.cols() == b.rows());
    Matrix c(a.rows(), b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t k = 0; k < a.cols(); ++k) {
            const double aik = a(i, k);
            for (std::size_t j = 0; j < b.cols(); ++j) {
                c(i, j) += aik * b(k, j);
            }
        }
    }
    return c;
}

// Relative comparison, the tolerance is scaled by the magnitudes involved.
inline bool approx_equal(double a, double b, double tol = 1e-10) {
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= tol * scale;
}

}  // namespace nla

#endif
