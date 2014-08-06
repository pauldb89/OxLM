#include "gtest/gtest.h"

#include "lbl/context_cache.h"

namespace oxlm {

TEST(ContextCacheTest, TestBasic) {
  ContextCache cache;
  vector<int> context = {1, 2, 3};

  EXPECT_EQ(make_pair(0.0f, false), cache.get(context));

  cache.set(context, 0.5);
  EXPECT_EQ(make_pair(0.5f, true), cache.get(context));

  context = {3, 2, 1};
  cache.set(context, 0.25);
  EXPECT_EQ(make_pair(0.25f, true), cache.get(context));

  cache.clear();
  context = {1, 2, 3};
  EXPECT_EQ(make_pair(0.0f, false), cache.get(context));
  context = {3, 2, 1};
  EXPECT_EQ(make_pair(0.0f, false), cache.get(context));
}

} // namespace oxlm
