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

  VectorReal values(4);
  values << 4, 1, 7, 9;
  batch.add(values);
  expected_values << 8, 6, 8, 11;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
  EXPECT_NEAR(8, buffer[3], EPS);
  EXPECT_NEAR(6, buffer[4], EPS);
  EXPECT_NEAR(8, buffer[0], EPS);
  EXPECT_NEAR(11, buffer[1], EPS);

  batch.setZero();
  expected_values << 0, 0, 0, 0;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
  EXPECT_NEAR(0, buffer[3], EPS);
  EXPECT_NEAR(0, buffer[4], EPS);
  EXPECT_NEAR(0, buffer[0], EPS);
  EXPECT_NEAR(0, buffer[1], EPS);
}

} // namespace oxlm
