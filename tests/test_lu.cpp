#include "numlinalg/lu.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

using namespace std;

using nla::LU;
using nla::Matrix;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        printf("FAIL: %s\n", what);
        ++failures;
    }
}

// Largest absolute component of A x - b.
static double residual(const Matrix& a, const vector<double>& x,
                       const vector<double>& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.rows(); ++i) {
        double s = b[i];
        for (size_t j = 0; j < a.cols(); ++j) {
            s -= a(i, j) * x[j];
        }
        worst = max(worst, abs(s));
    }
    return worst;
}

static void check_factorization(const Matrix& a, const LU& lu, const char* what) {
    const size_t n = a.rows();
    bool ok = true;
    for (size_t i = 0; i < n && ok; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) {
                s += lu.L(i, k) * lu.U(k, j);
            }
            if (!nla::approx_equal(s, a(lu.perm[i], j))) {
                ok = false;
                break;
            }
        }
    }
    check(ok, what);
}

int main() {
    // Transpose and multiply sanity checks.
    Matrix p(2, 3), q(3, 2);
    p(0, 0) = 1; p(0, 1) = 2; p(0, 2) = 3;
    p(1, 0) = 4; p(1, 1) = 5; p(1, 2) = 6;
    q(0, 0) = 7;  q(0, 1) = 8;
    q(1, 0) = 9;  q(1, 1) = 10;
    q(2, 0) = 11; q(2, 1) = 12;
    Matrix pq = p * q;
    check(nla::approx_equal(pq(0, 0), 58.0) && nla::approx_equal(pq(0, 1), 64.0),
          "matrix multiply");
    check(nla::approx_equal(pq(1, 0), 139.0) && nla::approx_equal(pq(1, 1), 154.0),
          "matrix multiply");
    check(nla::approx_equal(p.transpose()(2, 0), 3.0) &&
              nla::approx_equal(p.transpose()(0, 1), 4.0),
          "transpose");

    // A matrix whose elimination needs a row swap: solving without pivoting
    // would divide by zero on the first step.
    Matrix a(2, 2);
    a(0, 0) = 0; a(0, 1) = 1;
    a(1, 0) = 1; a(1, 1) = 1;
    LU lu = nla::lu_decompose(a);
    check_factorization(a, lu, "P A = L U for the swap case");
    vector<double> x = nla::lu_solve(lu, {3.0, 4.0});
    check(nla::approx_equal(x[0], 1.0) && nla::approx_equal(x[1], 3.0),
          "swap case solution");

    // Classic 3x3 with a known solution x = (1, 2, 3).
    Matrix m(3, 3);
    m(0, 0) = 2; m(0, 1) = 1; m(0, 2) = 1;
    m(1, 0) = 4; m(1, 1) = 3; m(1, 2) = 3;
    m(2, 0) = 8; m(2, 1) = 7; m(2, 2) = 9;
    LU mlu = nla::lu_decompose(m);
    check_factorization(m, mlu, "P A = L U for the 3x3 system");
    vector<double> mx = nla::lu_solve(mlu, {7.0, 19.0, 49.0});
    check(nla::approx_equal(mx[0], 1.0) && nla::approx_equal(mx[1], 2.0) &&
              nla::approx_equal(mx[2], 3.0),
          "3x3 solution");

    // 5x5 Hilbert matrix: badly conditioned, so the residual bound is loose
    // on purpose. A good check that partial pivoting keeps the backward error
    // small even when the solution itself is sensitive.
    const size_t n = 5;
    Matrix h(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            h(i, j) = 1.0 / static_cast<double>(i + j + 1);
        }
    }
    vector<double> hx_exact(n), hb(n);
    for (size_t i = 0; i < n; ++i) {
        hx_exact[i] = static_cast<double>(i + 1);
    }
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < n; ++j) {
            s += h(i, j) * hx_exact[j];
        }
        hb[i] = s;
    }
    LU hlu = nla::lu_decompose(h);
    vector<double> hx = nla::lu_solve(hlu, hb);
    check(residual(h, hx, hb) < 1e-8, "hilbert residual");

    if (failures == 0) {
        printf("all tests passed\n");
        return 0;
    }
    printf("%d test(s) failed\n", failures);
    return 1;
}
