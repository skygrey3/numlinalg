# numlinalg

Dense and sparse numerical linear algebra in C++17, implemented from scratch.
The goal is to understand the algorithms and their numerical behaviour, not to
replace Eigen or LAPACK.

## Status

| Implemented | Planned |
|---|---|
| Dense `Matrix` (row-major), transpose, arithmetic | Preconditioned iterative solvers (Jacobi, incomplete lower-upper (ILU)) |
| Lower-upper (LU) decomposition with partial pivoting (`P A = L U`) and solve | QR / least squares |
| Cholesky (`A = L L^T`) with symmetry validation and solve | Benchmark suite against Eigen |
| Compressed sparse row (CSR) storage and sparse matrix-vector (mat-vec) products | 2D Poisson demo application |
| Conjugate gradient (CG) for sparse symmetric positive definite (SPD) systems (warm start) | |
| Jacobi eigensolver for symmetric matrices | |

## Quick start

    cmake -S . -B build
    cmake --build build
    ctest --test-dir build --output-on-failure

To build with AddressSanitizer and UndefinedBehaviorSanitizer:

    cmake -S . -B build -DNLA_SANITIZE=ON
    cmake --build build
    ctest --test-dir build --output-on-failure

## Usage

```cpp
#include "numlinalg/lu.hpp"

nla::Matrix a(2, 2);
a(0, 0) = 4; a(0, 1) = 1;
a(1, 0) = 1; a(1, 1) = 3;

nla::LU lu = nla::lu_decompose(a);              // throws if exactly singular
std::vector<double> x = nla::lu_solve(lu, {1.0, 2.0});
```

For Cholesky the factor is returned as a named struct:

```cpp
#include "numlinalg/cholesky.hpp"

nla::Cholesky chol = nla::cholesky(a);          // requires A symmetric, SPD
std::vector<double> x = nla::cholesky_solve(chol, b);
```

Conjugate gradient returns an outcome object and accepts a warm start:

```cpp
#include "numlinalg/cg.hpp"

nla::SparseMatrix s(a);
std::vector<double> x;                          // empty -> starts from zero
nla::CGResult res = nla::conjugate_gradient(s, b, x);
if (!res.converged) {
    // res.iterations, res.residual describe where the budget ran out.
}
```

## Design notes

- Dense storage is row-major; CSR stores only entries kept at construction.
- Bounds and dimension checks throw in every build, including release. Use
  `Matrix::unchecked()` only in hot loops whose indices are already validated.
- Partial pivoting controls growth for LU; Cholesky validates symmetry and
  requires a positive pivot.
- CG stops on the relative residual `||r|| / ||b|| <= tol` and reports both the
  iteration count and the final absolute residual. A supplied `x` of size `n`
  is used as the initial guess; the starting residual is recomputed as
  `b - A x`.
- Jacobi converges when the off-diagonal Frobenius norm falls to `1e-12` times
  the Frobenius norm of the input, so the test is scale invariant.

## Complexity

Costs are stated in terms of the matrix dimension `n` and the number of stored
nonzeros (`nnz`).

| Routine | Time | Storage |
|---|---|---|
| LU decomposition | `O(2/3 n^3)` | `O(n^2)` |
| Cholesky | `O(1/3 n^3)` | `O(n^2)` |
| Sparse mat-vec | `O(nnz)` | `O(nnz)` |
| Conjugate gradient | `O(nnz)` per iteration | `O(n)` |
| Jacobi sweep | `O(n^3)` | `O(n^2)` |

## References

- Trefethen & Bau, *Numerical Linear Algebra*
- Golub & Van Loan, *Matrix Computations*
