#ifndef NUMLINALG_CG_HPP
#define NUMLINALG_CG_HPP

#include <cstddef>
#include <vector>

#include "numlinalg/sparse.hpp"

using std::size_t;
using std::vector;

namespace nla {

// Outcome of a conjugate gradient run.
struct CGResult {
    size_t iterations = 0;  // iterations actually performed
    bool converged = false; // relative residual <= tol at exit
    double residual = 0.0;  // absolute 2-norm of the final residual r = b - A x
};

// Conjugate gradient for a sparse symmetric positive definite system A x = b
// (A's action given through SparseMatrix::multiply).
//
// Warm start: if x already has size n it is used as the initial guess; if it
// has any other size, x is resized and zero-filled first. The starting
// residual is therefore b - A x, not b, for a supplied guess.
//
// Iterates until the relative residual ||r|| / ||b|| <= tol or max_iter is
// reached (max_iter == 0 means n, the exact-arithmetic bound). Returns a
// CGResult; converged == false means the budget ran out. Throws
// std::invalid_argument on a dimension mismatch and std::runtime_error if A
// is not SPD (detected by a non-positive p^T A p).
CGResult conjugate_gradient(const SparseMatrix& a, const vector<double>& b,
                            vector<double>& x, double tol = 1e-10,
                            size_t max_iter = 0);

}  // namespace nla

#endif  // NUMLINALG_CG_HPP
