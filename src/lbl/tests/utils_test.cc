#include "gtest/gtest.h"

#include "lbl/utils.h"
#include "utils/constants.h"
#include "utils/testing.h"

namespace oxlm {

TEST(UtilsTest, TestSoftMax) {
  VectorReal v(3);
  v << 1, 2, 3;
  VectorReal w = softMax(v);
  VectorReal expected_w(3);
  expected_w << 0.09003, 0.244728, 0.665240;
  EXPECT_MATRIX_NEAR(expected_w, w, EPS);
}

TEST(UtilsTest, TestLogSoftMax) {
  VectorReal v(3);
  v << 1, 2, 3;
  Real log_z;

  VectorReal w = logSoftMax(v, log_z);
  VectorReal expected_w(3);
  expected_w << -2.407605, -1.407605, -0.407605;
  EXPECT_MATRIX_NEAR(expected_w, w, EPS);
  EXPECT_NEAR(3.407605964, log_z, EPS);

  w = logSoftMax(v);
  EXPECT_MATRIX_NEAR(expected_w, w, EPS);
}

TEST(UtilsTest, TestLogSoftMaxZero) {
  VectorReal v(1);
  v << 0;
  Real log_z;

  VectorReal w = logSoftMax(v, log_z);
  VectorReal expected_w(1);
  expected_w << 0;
  EXPECT_MATRIX_NEAR(expected_w, w, EPS);
  EXPECT_NEAR(0, log_z, EPS);

  w = logSoftMax(v);
  EXPECT_MATRIX_NEAR(expected_w, w, EPS);
}

TEST(UtilsTest, TestSigmoid) {
  VectorReal v(3);
  v << -1, 0, 1;
  VectorReal w = sigmoid(v);
  VectorReal expected_w(3);
  expected_w << 0.268941, 0.5, 0.731058;
  EXPECT_MATRIX_NEAR(expected_w, w, EPS);
}

TEST(UtilsTest, TestSigmoidDerivative) {
  VectorReal v(5);
  v << 0.5, 1, 0, 0.3, 0.8;
  ArrayReal w = sigmoidDerivative(v);
  ArrayReal expected_w(5);
  expected_w << 0.25, 0, 0, 0.21, 0.16;
  EXPECT_MATRIX_NEAR(expected_w, w, EPS);
}

TEST(UtilsTest, TestMurmurHash) {
  vector<int> data = {0};
  EXPECT_EQ(14961230494313510588ULL, MurmurHash(data, 0));
  EXPECT_EQ(4464299780005672250ULL, MurmurHash(data, 1));

  data = {1};
  EXPECT_EQ(9841952836289088254ULL, MurmurHash(data));

  data = {2};
  EXPECT_EQ(15712189420091562848ULL, MurmurHash(data));

  data = {1, 2};
  EXPECT_EQ(15037832940209189704ULL, MurmurHash(data));

  data = {1, 2, 3};
  EXPECT_EQ(5822962753042209973ULL, MurmurHash(data));
}

} // namespace oxlm
