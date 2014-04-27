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

  VectorReal values(3);
  values << 5, 10, 7;
  batch.add(values);
  expected_values << 7, 13, 11;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
  EXPECT_NEAR(7, buffer[1], EPS);
  EXPECT_NEAR(13, buffer[2], EPS);
  EXPECT_NEAR(11, buffer[3], EPS);

  batch.setZero();
  expected_values << 0, 0, 0;
  EXPECT_MATRIX_NEAR(expected_values, batch.values(), EPS);
  EXPECT_NEAR(0, buffer[1], EPS);
  EXPECT_NEAR(0, buffer[2], EPS);
  EXPECT_NEAR(0, buffer[3], EPS);
}

} // namespace oxlm
