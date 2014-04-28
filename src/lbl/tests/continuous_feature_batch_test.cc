#include "gtest/gtest.h"

#include "lbl/continuous_feature_batch.h"
#include "utils/constants.h"
#include "utils/testing.h"

namespace oxlm {

TEST(ContinuousFeatureBatchTest, TestBasic) {
  Real buffer[5] = {1, 2, 3, 4, 5};
  ContinuousFeatureBatch batch(buffer, 1, 3);
  VectorReal expected_values(3);
  expected_values << 2, 3, 4;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
}

TEST(ContinuousFeatureBatchTest, TestUpdate) {
  Real buffer[5] = {1, 1, 1, 1, 1};
  ContinuousFeatureBatch batch(buffer, 1, 3);
  VectorReal values(3);
  values << 5, 10, 7;
  batch.update(values);

  VectorReal expected_values(3);
  expected_values << 6, 11, 8;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
  EXPECT_NEAR(6, buffer[1], EPS);
  EXPECT_NEAR(11, buffer[2], EPS);
  EXPECT_NEAR(8, buffer[3], EPS);
}

TEST(ContinuousFeatureBatchTest, TestUpdateSquared) {
  Real buffer[5] = {0, 1, 1, 1, 0};
  ContinuousFeatureBatch batch(buffer, 1, 3);
  VectorReal values(3);
  values << 5, 10, 7;
  batch.updateSquared(values);

  VectorReal expected_values(3);
  expected_values << 26, 101, 50;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
  EXPECT_NEAR(26, buffer[1], EPS);
  EXPECT_NEAR(101, buffer[2], EPS);
  EXPECT_NEAR(50, buffer[3], EPS);
}

TEST(ContinuousFeatureBatchTest, TestUpdateAdaGrad) {
  Real buffer[5] = {0, 1, 2, 3, 0};
  ContinuousFeatureBatch batch(buffer, 1, 3);
  VectorReal values(3);
  values << 25, 36, 9;
  batch.updateAdaGrad(values, values, 0.5);

  VectorReal expected_values(3);
  expected_values << -1.5, -1, 1.5;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
  EXPECT_NEAR(-1.5, buffer[1], EPS);
  EXPECT_NEAR(-1, buffer[2], EPS);
  EXPECT_NEAR(1.5, buffer[3], EPS);
}

TEST(ContinuousFeatureBatchTest, TestL2) {
  Real buffer[5] = {0, 1, 2, 3, 0};
  ContinuousFeatureBatch batch(buffer, 1, 3);
  EXPECT_NEAR(7, batch.l2Objective(0.5), EPS);

  batch.l2Update(0.5);
  VectorReal expected_values(3);
  expected_values << 0.5, 1, 1.5;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
  EXPECT_NEAR(0.5, buffer[1], EPS);
  EXPECT_NEAR(1, buffer[2], EPS);
  EXPECT_NEAR(1.5, buffer[3], EPS);
}

TEST(ContinuousFeatureBatchTest, TestSetZero) {
  Real buffer[5] = {1, 2, 3, 4, 5};
  ContinuousFeatureBatch batch(buffer, 1, 3);
  batch.setZero();

  VectorReal expected_values(3);
  expected_values << 0, 0, 0;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
  EXPECT_NEAR(0, buffer[1], EPS);
  EXPECT_NEAR(0, buffer[2], EPS);
  EXPECT_NEAR(0, buffer[3], EPS);
}

} // namespace oxlm
