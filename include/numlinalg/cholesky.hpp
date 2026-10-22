#ifndef NUMLINALG_CHOLESKY_HPP
#define NUMLINALG_CHOLESKY_HPP

#include <cstddef>
#include <vector>

#include "numlinalg/matrix.hpp"

using std::size_t;
using std::vector;

namespace nla {

// Cholesky factorization A = L L^T for symmetric positive definite A.
// L is lower triangular with a positive diagonal; A is left unmodified.
//
// Throws std::invalid_argument if A is not square and non-empty, or if A is
// not symmetric within a relative tolerance of 1e-12 (so round-off from the
// caller does not trip the check). Throws std::runtime_error if A is not
// positive definite (a non-positive pivot is encountered).
struct Cholesky {
    Matrix L;
};

Cholesky cholesky(const Matrix& a);

// Solve A x = b given the factorization from cholesky(): forward
// substitution with L, then back substitution with L^T.
vector<double> cholesky_solve(const Cholesky& chol, const vector<double>& b);

}  // namespace nla

#endif  // NUMLINALG_CHOLESKY_HPP
