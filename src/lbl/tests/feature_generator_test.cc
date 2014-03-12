#include "gtest/gtest.h"

#include "lbl/feature_generator.h"

namespace oxlm {

TEST(FeatureGeneratorTest, TestLongerHistory) {
  FeatureGenerator generator(3);
  vector<int> history = {1, 2, 3, 4, 5};
  vector<Feature> features = generator.generate(history);

  EXPECT_EQ(3, features.size());
  vector<int> expected_feature_data = {5};
  EXPECT_EQ(expected_feature_data, features[0].data);
  EXPECT_EQ(0, features[0].feature_type);
  expected_feature_data = {5, 4};
  EXPECT_EQ(expected_feature_data, features[1].data);
  EXPECT_EQ(1, features[1].feature_type);
  expected_feature_data = {5, 4, 3};
  EXPECT_EQ(expected_feature_data, features[2].data);
  EXPECT_EQ(2, features[2].feature_type);
}

TEST(FeatureGeneratorTest, TestShorterHistory) {
  FeatureGenerator generator(5);
  vector<int> history = {1, 2, 3};
  vector<Feature> features = generator.generate(history);

  EXPECT_EQ(3, features.size());
  vector<int> expected_feature_data = {3};
  EXPECT_EQ(expected_feature_data, features[0].data);
  EXPECT_EQ(0, features[0].feature_type);
  expected_feature_data = {3, 2};
  EXPECT_EQ(expected_feature_data, features[1].data);
  EXPECT_EQ(1, features[1].feature_type);
  expected_feature_data = {3, 2, 1};
  EXPECT_EQ(expected_feature_data, features[2].data);
  EXPECT_EQ(2, features[2].feature_type);
}

} // namespace oxlm
