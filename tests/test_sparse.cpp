#include "numlinalg/matrix.hpp"
#include "numlinalg/sparse.hpp"

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace std;

using nla::Matrix;
using nla::SparseMatrix;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        printf("FAIL: %s\n", what);
        ++failures;
    }
}

int main() {
    // Small dense matrix with structural zeros.
    Matrix d(3, 4);
    d(0, 0) = 1; d(0, 1) = 0; d(0, 2) = 2; d(0, 3) = 0;
    d(1, 0) = 0; d(1, 1) = 3; d(1, 2) = 0; d(1, 3) = 4;
    d(2, 0) = 5; d(2, 1) = 0; d(2, 2) = 6; d(2, 3) = 0;

    const SparseMatrix a(d);
    check(a.rows() == 3 && a.cols() == 4, "shape preserved");
    check(a.nonzeros() == 6, "six structural nonzeros stored");
    check(a.row_nnz(0) == 2 && a.row_nnz(1) == 2 && a.row_nnz(2) == 2,
          "two nonzeros per row");
    check(a.at(0, 0) == 1 && a.at(0, 2) == 2, "values readable from CSR");
    check(a.at(1, 0) == 0 && a.at(2, 1) == 0, "structural zeros read as zero");

    // Threshold drops small entries: |values| are 1,2,3,4,5,6; drop <= 3.
    const SparseMatrix t(d, 3.0);
    check(t.nonzeros() == 3, "threshold drops entries at or below 3");
    check(t.at(0, 0) == 0 && t.at(1, 1) == 0 && t.at(1, 3) == 4,
          "dropped entries read as zero, kept ones survive");

    // y = A x against hand-computed result.
    const vector<double> x = {1.0, 1.0, 1.0, 1.0};
    const vector<double> y = a.multiply(x);
    check(y.size() == 3, "result length matches rows");
    check(y[0] == 3 && y[1] == 7 && y[2] == 11, "multiply matches hand result");

    const vector<double> x2 = {2.0, -1.0, 0.5, 3.0};
    const vector<double> y2 = a.multiply(x2);
    check(y2[0] == 3.0 && y2[1] == 9.0 && y2[2] == 13.0,
          "second multiply matches hand result");

    // Mismatched lengths are rejected.
    bool threw = false;
    try {
        (void)a.multiply({1.0, 2.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "length mismatch rejected");

    if (failures == 0) {
        printf("all tests passed\n");
        return 0;
    }
    printf("%d test(s) failed\n", failures);
    return 1;
}
