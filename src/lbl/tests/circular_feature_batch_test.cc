#include "gtest/gtest.h"

#include "lbl/circular_feature_batch.h"
#include "utils/constants.h"
#include "utils/testing.h"

namespace oxlm {

TEST(CircularFeatureBatchTest, TestBasic) {
  Real buffer[5] = {1, 2, 3, 4, 5};
  CircularFeatureBatch batch(buffer, 3, 4, 5);

  VectorReal expected_values(4);
  expected_values << 4, 5, 1, 2;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
}

TEST(CircularFeatureBatchTest, TestUpdate) {
  Real buffer[5] = {1, 2, 3, 4, 5};
  CircularFeatureBatch batch(buffer, 3, 4, 5);
  VectorReal values(4);
  values << 4, 1, 7, 9;
  batch.update(values);

  VectorReal expected_values(4);
  expected_values << 8, 6, 8, 11;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
  EXPECT_NEAR(8, buffer[3], EPS);
  EXPECT_NEAR(6, buffer[4], EPS);
  EXPECT_NEAR(8, buffer[0], EPS);
  EXPECT_NEAR(11, buffer[1], EPS);
}

TEST(CircularFeatureBatchTest, TestUpdateSquared) {
  Real buffer[5] = {1, 1, 0, 1, 1};
  CircularFeatureBatch batch(buffer, 3, 4, 5);
  VectorReal values(4);
  values << 2, 1, 3, 1;
  batch.updateSquared(values);

  VectorReal expected_values(4);
  expected_values << 5, 2, 10, 2;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
  EXPECT_NEAR(5, buffer[3], EPS);
  EXPECT_NEAR(2, buffer[4], EPS);
  EXPECT_NEAR(10, buffer[0], EPS);
  EXPECT_NEAR(2, buffer[1], EPS);
}

TEST(CircularFeatureBatchTest, TestUpdateAdaGrad) {
  Real buffer[5] = {1, 1, 0, 1, 1};
  CircularFeatureBatch batch(buffer, 3, 4, 5);
  VectorReal values(4);
  values << 4, 25, 9, 16;
  batch.updateAdaGrad(values, values, 0.5);

  VectorReal expected_values(4);
  expected_values << 0, -1.5, -0.5, -1;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
  EXPECT_NEAR(0, buffer[3], EPS);
  EXPECT_NEAR(-1.5, buffer[4], EPS);
  EXPECT_NEAR(-0.5, buffer[0], EPS);
  EXPECT_NEAR(-1, buffer[1], EPS);
}

TEST(CircularFeatureBatchTest, TestL2) {
  Real buffer[5] = {1, 2, 3, 4, 5};
  CircularFeatureBatch batch(buffer, 3, 4, 5);
  EXPECT_EQ(23, batch.l2Objective(0.5));

  batch.l2Update(0.5);
  VectorReal expected_values(4);
  expected_values << 2, 2.5, 0.5, 1;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
  EXPECT_NEAR(2, buffer[3], EPS);
  EXPECT_NEAR(2.5, buffer[4], EPS);
  EXPECT_NEAR(0.5, buffer[0], EPS);
  EXPECT_NEAR(1, buffer[1], EPS);
}

TEST(CircularFeatureBatchTest, TestSetZero) {
  Real buffer[5] = {1, 1, 1, 1, 1};
  CircularFeatureBatch batch(buffer, 3, 4, 5);
  batch.setZero();

  VectorReal expected_values(4);
  expected_values << 0, 0, 0, 0;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
  EXPECT_NEAR(0, buffer[3], EPS);
  EXPECT_NEAR(0, buffer[4], EPS);
  EXPECT_NEAR(0, buffer[0], EPS);
  EXPECT_NEAR(0, buffer[1], EPS);
}

} // namespace oxlm
