#include "gtest/gtest.h"

#include "lbl/operators.h"
#include "lbl/utils.h"
#include "utils/testing.h"

namespace oxlm {

TEST(OperatorsTest, TestSetValueOp) {
  VectorReal v(3);
  v << 1, 2, 3;
  VectorReal result = v.unaryExpr(CwiseSetValueOp<Real>(5));
  VectorReal expected_result(3);
  expected_result << 5, 5, 5;
  EXPECT_MATRIX_NEAR(expected_result, result, EPS);
}

TEST(OperatorsTest, TestAdagradUpdateOp) {
  VectorReal gradient(3), adagrad(3);
  gradient << 10, 773, 20;
  adagrad << 25, 0, 16;
  VectorReal result =
      gradient.binaryExpr(adagrad, CwiseAdagradUpdateOp<Real>(0.5));
  VectorReal expected_result(3);
  expected_result << 1, 0, 2.5;
  EXPECT_MATRIX_NEAR(expected_result, result, EPS);
}

} // namespace oxlm
