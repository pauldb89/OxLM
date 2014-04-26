#include "gtest/gtest.h"

#include "lbl/feature_context_keyer.h"

namespace oxlm {

TEST(FeatureContextKeyerTest, TestBasic) {
  FeatureContextKeyer keyer(3);
  vector<int> context = {1, 2, 3};
  vector<size_t> keys = keyer.getKeys(context);
  EXPECT_EQ(3, keys.size());
  EXPECT_EQ(5308871539, keys[0]);
  EXPECT_EQ(177902205132, keys[1]);
  EXPECT_EQ(11096476896012, keys[2]);
}

} // namespace oxlm
