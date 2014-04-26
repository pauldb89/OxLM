#include "gtest/gtest.h"

#include "lbl/feature_context_generator.h"

namespace oxlm {

TEST(FeatureContextGeneratorTest, TestShorterHistory) {
  FeatureContextGenerator generator(3);
  vector<int> history = {1, 2};
  vector<int> expected_data;

  vector<FeatureContext> feature_contexts =
      generator.getFeatureContexts(history);
  EXPECT_EQ(2, feature_contexts.size());
  expected_data = {1};
  EXPECT_EQ(expected_data, feature_contexts[0].data);
  expected_data = {1, 2};
  EXPECT_EQ(expected_data, feature_contexts[1].data);
}

TEST(FeatureContextGeneratorTest, TestLongerHistory) {
  FeatureContextGenerator generator(2);
  vector<int> history = {1, 2, 3};
  vector<int> expected_data;

  vector<FeatureContext> feature_contexts =
      generator.getFeatureContexts(history);
  EXPECT_EQ(2, feature_contexts.size());
  expected_data = {1};
  EXPECT_EQ(expected_data, feature_contexts[0].data);
  expected_data = {1, 2};
  EXPECT_EQ(expected_data, feature_contexts[1].data);
}

} // namespace oxlm
