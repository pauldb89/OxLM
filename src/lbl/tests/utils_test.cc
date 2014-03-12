#include "gtest/gtest.h"

#include "lbl/utils.h"
#include "utils/testing.h"

namespace oxlm {

TEST(TestSoftMax, BasicTest) {
  VectorReal v(3);
  v << 1, 2, 3;
  VectorReal w = softMax(v);
  VectorReal expected_w(3);
  expected_w << 0.09003, 0.244728, 0.665240;
  EXPECT_MATRIX_NEAR(expected_w, w, 1e-5);
}

TEST(TestLogSoftMax, BasicTest) {
  VectorReal v(3);
  v << 1, 2, 3;
  VectorReal w = logSoftMax(v);
  VectorReal expected_w(3);
  expected_w << -2.407605, -1.407605, -0.407605;
  EXPECT_MATRIX_NEAR(expected_w, w, 1e-5);
}

TEST(TestSigmoid, BasicTest) {
  VectorReal v(3);
  v << -1, 0, 1;
  VectorReal w = sigmoid(v);
  VectorReal expected_w(3);
  expected_w << 0.268941, 0.5, 0.731058;
  EXPECT_MATRIX_NEAR(expected_w, w, 1e-5);
}

} // namespace oxlm
