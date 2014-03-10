#include "gtest/gtest.h"

#include "lbl/utils.h"
#include "utils/testing.h"

namespace oxlm {

TEST(TestSigmoid, BasicTest) {
  VectorReal v(3);
  v << -1, 0, 1;
  VectorReal w = sigmoid(v);
  VectorReal expected_w(3);
  expected_w << 0.268941, 0.5, 0.731058;
  EXPECT_MATRIX_NEAR(expected_w, w, 1e-5);
}

} // namespace oxlm
