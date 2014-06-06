#include "gtest/gtest.h"

#include "lbl/bloom_filter.h"
#include "lbl/ngram.h"

namespace oxlm {

TEST(BloomFilterTest, TestBasic) {
  BloomFilter<NGram> bloom_filter(10, 3, 0.01);

  vector<int> context = {1, 2, 3};
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 10; ++i) {
      NGram query(i, context);
      bloom_filter.increment(query);
      EXPECT_FALSE(bloom_filter.contains(query));
    }
  }

  for (int i = 0; i < 10; ++i) {
    NGram query(i, context);
    bloom_filter.increment(query);
    EXPECT_TRUE(bloom_filter.contains(query));
  }
}

} // namespace oxlm
