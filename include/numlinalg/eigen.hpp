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
// returned in the vector, in no particular order. Throws std::invalid_argument
// on non-square or empty input, and std::runtime_error if the sweep budget
// (100 sweeps) is exhausted.
//
// Convergence is measured relatively: the Frobenius norm of the off-diagonal
// part must fall to 1e-12 times the Frobenius norm of the input matrix. The
// criterion is therefore invariant under scaling A by a positive constant,
// unlike an absolute threshold on the off-diagonal sum of squares. The input
// is internally rescaled by its largest entry so the squared norm cannot
// overflow or underflow at extreme scales; eigenvalues are scaled back.
vector<double> jacobi_eigenvalues(Matrix& a);

}  // namespace nla

#endif
