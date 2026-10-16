# numlinalg

A small dense linear algebra library written in C++ from scratch. The point is
to implement the classic dense algorithms myself and understand their stability
and convergence behavior, not to compete with Eigen or LAPACK.

## Implemented

- dense Matrix type (row-major storage)
- LU decomposition with partial pivoting and forward/back substitution

## Planned

- Cholesky factorization for symmetric positive definite matrices
- conjugate gradient for sparse systems (CSR storage)
- Jacobi eigenvalue algorithm
- benchmarks against Eigen
- application: 2D Poisson solver on a regular grid

## Build

    cmake -S . -B build
    cmake --build build
    ./build/test_lu

## References

- Trefethen & Bau, *Numerical Linear Algebra*
- Golub & Van Loan, *Matrix Computations*
