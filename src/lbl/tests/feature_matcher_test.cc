#include "gtest/gtest.h"

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/feature_matcher.h"

namespace oxlm {

class FeatureMatcherTest : public testing::Test {
 protected:
  void SetUp() {
    vector<int> class_markers = {0, 2, 3, 4};
    corpus = {2, 3, 3, 1, 3, 2};
    index = WordToClassIndex(class_markers);
    processor = boost::make_shared<ContextProcessor>(corpus, 2, 0, 1);
    extractor = boost::make_shared<FeatureContextExtractor>(
        corpus, *processor, 2);
    feature_matcher = boost::make_shared<FeatureMatcher>(
        corpus, index, *processor, extractor);
  }

  void checkFeatureContexts(
      FeatureIndexesPtr feature_indexes,
      const vector<FeatureContextId>& feature_context_ids,
      int feature_index) const {
    for (const FeatureContextId& feature_context_id: feature_context_ids) {
      EXPECT_TRUE(feature_indexes->count(feature_context_id));
      EXPECT_TRUE(feature_indexes->at(feature_context_id).count(feature_index));
    }
  }

  Corpus corpus;
  WordToClassIndex index;
  boost::shared_ptr<ContextProcessor> processor;
  boost::shared_ptr<FeatureContextExtractor> extractor;
  boost::shared_ptr<FeatureMatcher> feature_matcher;
};

TEST_F(FeatureMatcherTest, TestBasic) {
  vector<int> history;
  vector<FeatureContextId> feature_context_ids;
  FeatureIndexesPairPtr feature_indexes_pair = feature_matcher->getFeatures();

  FeatureIndexesPtr feature_indexes = feature_indexes_pair->getClassIndexes();
  EXPECT_EQ(8, feature_indexes->size());
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
  checkFeatureContexts(feature_indexes, feature_context_ids, 2);
  history = {3, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 1);

  feature_indexes = feature_indexes_pair->getWordIndexes(0);
  EXPECT_EQ(2, feature_indexes->size());
  history = {3, 3};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 1);

  feature_indexes = feature_indexes_pair->getWordIndexes(1);
  EXPECT_EQ(4, feature_indexes->size());
  history = {0, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 0);
  history = {3, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 0);

  feature_indexes = feature_indexes_pair->getWordIndexes(2);
  EXPECT_EQ(6, feature_indexes->size());
  history = {2, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 0);
  history = {3, 2};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 0);
  history = {0, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 0);
}

TEST_F(FeatureMatcherTest, TestSubset) {
  vector<int> minibatch_indexes = {1, 4};

  vector<int> history;
  vector<FeatureContextId> feature_context_ids;
  boost::shared_ptr<FeatureIndexesPair> feature_indexes_pair =
      feature_matcher->getFeatures(minibatch_indexes);
  boost::shared_ptr<FeatureIndexes> feature_indexes =
      feature_indexes_pair->getClassIndexes();
  EXPECT_EQ(4, feature_indexes->size());
  history = {2, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 2);
  history = {0, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  // The starting 2 (belonging to class 1) is not explicitly selected, but it
  // follows the context [0, 0].
  checkFeatureContexts(feature_indexes, feature_context_ids, 1);
  checkFeatureContexts(feature_indexes, feature_context_ids, 2);

  feature_indexes = feature_indexes_pair->getWordIndexes(0);
  EXPECT_EQ(0, feature_indexes->size());

  feature_indexes = feature_indexes_pair->getWordIndexes(1);
  EXPECT_EQ(0, feature_indexes->size());

  feature_indexes = feature_indexes_pair->getWordIndexes(2);
  EXPECT_EQ(4, feature_indexes->size());
  history = {0, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 0);
  history = {2, 0};
  feature_context_ids = extractor->getFeatureContextIds(history);
  checkFeatureContexts(feature_indexes, feature_context_ids, 0);
}

} // namespace oxlm
