#include "gtest/gtest.h"

#include "lbl/feature_generator.h"

namespace oxlm {

TEST(FeatureGeneratorTest, TestLongerHistory) {
  FeatureGenerator generator(3);
  vector<int> history = {1, 2, 3, 4, 5};
  vector<FeatureContext> feature_contexts = generator.generate(history);

  EXPECT_EQ(3, feature_contexts.size());
  vector<int> expected_feature_data = {1};
  EXPECT_EQ(expected_feature_data, feature_contexts[0].data);
  EXPECT_EQ(0, feature_contexts[0].feature_type);
  expected_feature_data = {1, 2};
  EXPECT_EQ(expected_feature_data, feature_contexts[1].data);
  EXPECT_EQ(1, feature_contexts[1].feature_type);
  expected_feature_data = {1, 2, 3};
  EXPECT_EQ(expected_feature_data, feature_contexts[2].data);
  EXPECT_EQ(2, feature_contexts[2].feature_type);
}

TEST(FeatureGeneratorTest, TestShorterHistory) {
  FeatureGenerator generator(5);
  vector<int> history = {1, 2, 3};
  vector<FeatureContext> feature_contexts = generator.generate(history);

  EXPECT_EQ(3, feature_contexts.size());
  vector<int> expected_feature_data = {1};
  EXPECT_EQ(expected_feature_data, feature_contexts[0].data);
  EXPECT_EQ(0, feature_contexts[0].feature_type);
  expected_feature_data = {1, 2};
  EXPECT_EQ(expected_feature_data, feature_contexts[1].data);
  EXPECT_EQ(1, feature_contexts[1].feature_type);
  expected_feature_data = {1, 2, 3};
  EXPECT_EQ(expected_feature_data, feature_contexts[2].data);
  EXPECT_EQ(2, feature_contexts[2].feature_type);
}

} // namespace oxlm
