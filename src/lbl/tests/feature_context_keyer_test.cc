#include "gtest/gtest.h"

#include "lbl/feature_context_keyer.h"

namespace oxlm {

TEST(FeatureContextKeyerTest, TestBasic) {
  FeatureContextKeyer keyer(100, 3);
  vector<int> context = {1, 2, 3};
  vector<int> keys = keyer.getKeys(context);
  EXPECT_EQ(3, keys.size());
  EXPECT_EQ(54, keys[0]);
  EXPECT_EQ(4, keys[1]);
  EXPECT_EQ(73, keys[2]);
}

} // namespace oxlm
