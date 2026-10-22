#ifndef NUMLINALG_LU_HPP
#define NUMLINALG_LU_HPP

#include <cstddef>
#include <vector>

#include "numlinalg/matrix.hpp"

using std::size_t;
using std::vector;

namespace nla {

// LU factorization with partial pivoting: P A = L U, where L is unit lower
// triangular and U is upper triangular. perm holds the row order, i.e.
// row i of P A is row perm[i] of A.
//
// Throws std::invalid_argument if A is not square and non-empty, and
// std::runtime_error if an exactly zero pivot is found.
//
// Limitation: pivoting only detects an exactly zero pivot. A matrix that is
// merely near-singular (a pivot that is tiny relative to the matrix norm) is
// not flagged, and the computed factors can amplify round-off badly. There is
// no condition estimate and no iterative refinement; check the residual
// ||A x - b|| yourself when the conditioning is unknown.
struct LU {
    Matrix L;
    Matrix U;
    vector<size_t> perm;
};

LU lu_decompose(Matrix a);

// Solve A x = b given a factorization from lu_decompose.
vector<double> lu_solve(const LU& lu, const vector<double>& b);

}  // namespace nla

#endif
