#ifndef NUMLINALG_CHOLESKY_HPP
#define NUMLINALG_CHOLESKY_HPP

#include <cstddef>
#include <vector>

using std::size_t;
using std::vector;

namespace nla {

class Matrix;

// Cholesky factorization A = L L^T for symmetric positive definite A.
// Throws std::invalid_argument if A is not square, or std::runtime_error
// if A is not positive definite (a non-positive pivot is encountered).
// Returns L (lower triangular, size n x n); A is left unmodified.
vector<double> cholesky(const Matrix& a);

// Solve A x = b given the Cholesky factor L (as returned by cholesky()):
// forward substitution with L, then back substitution with L^T.
vector<double> cholesky_solve(const vector<double>& l,
                              const vector<double>& b);

}  // namespace nla

#endif  // NUMLINALG_CHOLESKY_HPP
