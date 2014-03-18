#include "gtest/gtest.h"

#include "lbl/feature_generator.h"

namespace oxlm {

TEST(FeatureGeneratorTest, TestLongerHistory) {
  vector<int> corpus = {2, 3, 4, 5, 6, 1};
  ContextExtractor extractor(corpus, 5, 0, 1);
  FeatureGenerator generator(corpus, extractor, 3);

  vector<int> history = {6, 5, 4, 3, 2};
  vector<FeatureContextId> feature_context_ids =
      generator.getFeatureContextIds(history);
  EXPECT_EQ(3, feature_context_ids.size());
  EXPECT_EQ(15, feature_context_ids[0]);
  EXPECT_EQ(16, feature_context_ids[1]);
  EXPECT_EQ(17, feature_context_ids[2]);
}

TEST(FeatureGeneratorTest, TestShorterHistory) {
  vector<int> corpus = {2, 3, 4, 5, 6, 1};
  ContextExtractor extractor(corpus, 3, 0, 1);
  FeatureGenerator generator(corpus, extractor, 5);

  vector<int> history = {6, 5, 4};
  vector<FeatureContextId> feature_context_ids =
      generator.getFeatureContextIds(history);

  EXPECT_EQ(3, feature_context_ids.size());
  EXPECT_EQ(15, feature_context_ids[0]);
  EXPECT_EQ(16, feature_context_ids[1]);
  EXPECT_EQ(17, feature_context_ids[2]);
}

} // namespace oxlm
