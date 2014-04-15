#include "gtest/gtest.h"

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/feature_matcher.h"

namespace oxlm {

class FeatureMatcherTest : public testing::Test {
 protected:
  void checkFeatureContexts(
      const FeatureIndexes& feature_indexes,
      const vector<FeatureContextId>& feature_context_ids,
      int feature_index) const {
    for (const FeatureContextId& feature_context_id: feature_context_ids) {
      EXPECT_TRUE(feature_indexes.count(feature_context_id));
      EXPECT_TRUE(feature_indexes.at(feature_context_id).count(feature_index));
    }
  }
};

TEST_F(FeatureMatcherTest, TestBasic) {
  vector<int> class_markers = {0, 2, 3, 4};
  Corpus corpus = {2, 3, 3, 1, 2, 2};
  WordToClassIndex index(class_markers);
  ContextProcessor processor(corpus, 2, 0, 1);
  boost::shared_ptr<FeatureContextExtractor> extractor =
      boost::make_shared<FeatureContextExtractor>(corpus, processor, 2);

  vector<int> history;
  vector<FeatureContextId> feature_context_ids;
  FeatureMatcher feature_matcher(corpus, index, processor, extractor);

  FeatureIndexes feature_indexes = feature_matcher.getClassFeatures();
  EXPECT_EQ(7, feature_indexes.size());
  history = {0, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 1);
  history = {2, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 2);
  history = {3, 2};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 2);
  history = {3, 3};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 0);
  history = {0, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 1);
  history = {2, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 1);

  feature_indexes = feature_matcher.getWordFeatures(0);
  EXPECT_EQ(2, feature_indexes.size());
  history = {3, 3};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 1);

  feature_indexes = feature_matcher.getWordFeatures(1);
  EXPECT_EQ(4, feature_indexes.size());
  history = {0, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 0);
  history = {0, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 0);
  history = {2, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 0);

  feature_indexes = feature_matcher.getWordFeatures(2);
  EXPECT_EQ(4, feature_indexes.size());
  history = {2, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 0);
  history = {3, 2};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 0);
}

} // namespace oxlm
