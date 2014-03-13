#include "gtest/gtest.h"

#include "lbl/feature_matcher.h"

namespace oxlm {

TEST(FeatureMatcherTest, TestBasic) {
  vector<int> class_markers = {0, 2, 3, 4};
  Corpus corpus = {2, 3, 3, 1, 2, 2};
  WordToClassIndex index(class_markers);
  ContextExtractor extractor(corpus, 2, 0, 1);
  FeatureGenerator generator(2);

  vector<int> history;
  vector<FeatureContext> feature_contexts;
  FeatureMatcher feature_matcher(corpus, index, extractor, generator);

  MatchingContexts matching_contexts = feature_matcher.getClassFeatures();
  EXPECT_EQ(6, matching_contexts.size());
  history = {0, 0};
  feature_contexts = generator.generate(history);
  EXPECT_EQ(feature_contexts, matching_contexts[0].first);
  EXPECT_EQ(1, matching_contexts[0].second);
  history = {2, 0};
  feature_contexts = generator.generate(history);
  EXPECT_EQ(feature_contexts, matching_contexts[1].first);
  EXPECT_EQ(2, matching_contexts[1].second);
  history = {3, 2};
  feature_contexts = generator.generate(history);
  EXPECT_EQ(feature_contexts, matching_contexts[2].first);
  EXPECT_EQ(2, matching_contexts[2].second);
  history = {3, 3};
  feature_contexts = generator.generate(history);
  EXPECT_EQ(feature_contexts, matching_contexts[3].first);
  EXPECT_EQ(0, matching_contexts[3].second);
  history = {0, 0};
  feature_contexts = generator.generate(history);
  EXPECT_EQ(feature_contexts, matching_contexts[4].first);
  EXPECT_EQ(1, matching_contexts[4].second);
  history = {2, 0};
  feature_contexts = generator.generate(history);
  EXPECT_EQ(feature_contexts, matching_contexts[5].first);
  EXPECT_EQ(1, matching_contexts[5].second);

  matching_contexts = feature_matcher.getWordFeatures(0);
  EXPECT_EQ(1, matching_contexts.size());
  history = {3, 3};
  feature_contexts = generator.generate(history);
  EXPECT_EQ(feature_contexts, matching_contexts[0].first);
  EXPECT_EQ(1, matching_contexts[0].second);

  matching_contexts = feature_matcher.getWordFeatures(1);
  EXPECT_EQ(3, matching_contexts.size());
  history = {0, 0};
  feature_contexts = generator.generate(history);
  EXPECT_EQ(feature_contexts, matching_contexts[0].first);
  EXPECT_EQ(0, matching_contexts[0].second);
  history = {0, 0};
  feature_contexts = generator.generate(history);
  EXPECT_EQ(feature_contexts, matching_contexts[1].first);
  EXPECT_EQ(0, matching_contexts[1].second);
  history = {2, 0};
  feature_contexts = generator.generate(history);
  EXPECT_EQ(feature_contexts, matching_contexts[2].first);
  EXPECT_EQ(0, matching_contexts[2].second);

  matching_contexts = feature_matcher.getWordFeatures(2);
  EXPECT_EQ(2, matching_contexts.size());
  history = {2, 0};
  feature_contexts = generator.generate(history);
  EXPECT_EQ(feature_contexts, matching_contexts[0].first);
  EXPECT_EQ(0, matching_contexts[0].second);
  history = {3, 2};
  feature_contexts = generator.generate(history);
  EXPECT_EQ(feature_contexts, matching_contexts[1].first);
  EXPECT_EQ(0, matching_contexts[1].second);
}

} // namespace oxlm
