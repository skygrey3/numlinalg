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
