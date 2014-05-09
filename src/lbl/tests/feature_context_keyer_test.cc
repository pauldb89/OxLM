#include "gtest/gtest.h"

#include "lbl/feature_context_keyer.h"

namespace oxlm {

TEST(FeatureContextKeyerTest, TestBasic) {
  FeatureContextKeyer keyer(100);
  vector<int> context = {1};
  EXPECT_EQ(54, keyer.getKey(context));
  context = {1, 2};
  EXPECT_EQ(4, keyer.getKey(context));
  context = {1, 2, 3};
  EXPECT_EQ(73, keyer.getKey(context));
}

} // namespace oxlm
