#ifndef NUMLINALG_CG_HPP
#define NUMLINALG_CG_HPP

#include <cmath>
#include <cstddef>
#include <vector>

#include "numlinalg/sparse.hpp"

using std::size_t;
using std::vector;

namespace nla {

// Solve A x = b with the conjugate gradient method. A must be symmetric
// positive definite (its action given through SparseMatrix::multiply).
// Iterates until the relative residual ||r|| / ||b|| <= tol or max_iter is
// reached. Returns the number of iterations performed; throws if b has the
// wrong length or A is not SPD (detected by a non-positive p^T A p).
size_t conjugate_gradient(const SparseMatrix& a, const vector<double>& b,
                          vector<double>& x, double tol = 1e-10,
                          size_t max_iter = 0);

}  // namespace nla

#endif
