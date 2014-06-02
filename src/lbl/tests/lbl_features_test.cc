#include "gtest/gtest.h"

#include "lbl/lbl_features.h"
#include "utils/constants.h"

namespace oxlm {

TEST(LBLFeaturesTest, TestBasic) {
  LBLFeatures f1(3, 4);
  LBLFeatures f2(6.5, 8);

  f1 += f2;
  EXPECT_NEAR(9.5, f1.LMScore, EPS);
  EXPECT_NEAR(12, f1.OOVScore, EPS);

  LBLFeatures f3;
  f1 += f3;
  EXPECT_NEAR(9.5, f1.LMScore, EPS);
  EXPECT_NEAR(12, f1.OOVScore, EPS);
}

} // namespace oxlm
