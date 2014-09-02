#include "gtest/gtest.h"

#include "lbl/class_distribution.h"

namespace oxlm {

TEST(ClassDistributionTest, TestBasic) {
  VectorReal unigram = VectorReal::Zero(4);
  unigram << 0.5, 0, 0.5, 0;

  ClassDistribution dist(unigram);
  for (int i = 0; i < 100; ++i) {
    int class_id = dist.sample();
    EXPECT_TRUE(class_id == 0 || class_id == 2);
  }
}

} // namespace oxlm
