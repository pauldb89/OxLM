#include "gtest/gtest.h"

#include "lbl/context_cache.h"

namespace oxlm {

TEST(ContextCacheTest, TestBasic) {
  ContextCache cache;
  vector<int> context = {1, 2, 3};

  EXPECT_EQ(make_pair(static_cast<Real>(0), false), cache.get(context));

  cache.set(context, 0.5);
  EXPECT_EQ(make_pair(static_cast<Real>(0.5), true), cache.get(context));

  context = {3, 2, 1};
  cache.set(context, 0.25);
  EXPECT_EQ(make_pair(static_cast<Real>(0.25), true), cache.get(context));

  cache.clear();
  context = {1, 2, 3};
  EXPECT_EQ(make_pair(static_cast<Real>(0), false), cache.get(context));
  context = {3, 2, 1};
  EXPECT_EQ(make_pair(static_cast<Real>(0), false), cache.get(context));
}

} // namespace oxlm
