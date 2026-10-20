#ifndef NUMLINALG_EIGEN_HPP
#define NUMLINALG_EIGEN_HPP

#include <cstddef>
#include <vector>

using std::size_t;
using std::vector;

namespace nla {

class Matrix;

// Eigenvalues of a symmetric matrix via cyclic Jacobi rotations.
// a is overwritten with the diagonal matrix of eigenvalues; eigenvalues are
// returned in the vector, in no particular order. Throws on non-square or
// empty input, and if the sweep budget (100 sweeps) is exhausted.
vector<double> jacobi_eigenvalues(Matrix& a);

}  // namespace nla

#endif
