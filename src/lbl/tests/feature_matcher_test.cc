#include "gtest/gtest.h"

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/feature_matcher.h"

namespace oxlm {

class FeatureMatcherTest : public testing::Test {
 protected:
  void SetUp() {
    vector<int> class_markers = {0, 2, 3, 4};
    vector<int> data = {2, 3, 3, 1, 3, 2};
    corpus = boost::make_shared<Corpus>(data);
    boost::shared_ptr<WordToClassIndex> index =
        boost::make_shared<WordToClassIndex>(class_markers);
    boost::shared_ptr<ContextProcessor> processor =
        boost::make_shared<ContextProcessor>(corpus, 2, 0, 1);
    generator = boost::make_shared<FeatureContextGenerator>(2);
    boost::shared_ptr<NGramFilter> filter =
        boost::make_shared<NGramFilter>(corpus, index, processor, generator);
    feature_matcher = boost::make_shared<FeatureMatcher>(
        corpus, index, processor, generator, filter);
  }

  void checkFeatureContexts(
      FeatureIndexesPtr feature_indexes,
      const vector<Hash>& context_hashes,
      int feature_index) const {
    for (Hash context_hash: context_hashes) {
      vector<int>& indexes = feature_indexes->at(context_hash);
      EXPECT_TRUE(find(indexes.begin(), indexes.end(), feature_index) != indexes.end());
    }
  }

  boost::shared_ptr<Corpus> corpus;
  boost::shared_ptr<FeatureContextGenerator> generator;
  boost::shared_ptr<FeatureMatcher> feature_matcher;
};

TEST_F(FeatureMatcherTest, TestGlobal) {
  vector<int> history;
  vector<Hash> context_hashes;
  auto feature_indexes_pair = feature_matcher->getFeatureIndexes();

  FeatureIndexesPtr feature_indexes = feature_indexes_pair->getClassIndexes();
  EXPECT_EQ(8, feature_indexes->size());
  history = {0, 0};
  context_hashes = generator->getFeatureContextHashes(history);
  checkFeatureContexts(feature_indexes, context_hashes, 1);
  history = {2, 0};
  context_hashes = generator->getFeatureContextHashes(history);
  checkFeatureContexts(feature_indexes, context_hashes, 2);
  history = {3, 2};
  context_hashes = generator->getFeatureContextHashes(history);
  checkFeatureContexts(feature_indexes, context_hashes, 2);
  history = {3, 3};
  context_hashes = generator->getFeatureContextHashes(history);
  checkFeatureContexts(feature_indexes, context_hashes, 0);
  history = {0, 0};
  context_hashes = generator->getFeatureContextHashes(history);
  checkFeatureContexts(feature_indexes, context_hashes, 2);
  history = {3, 0};
  context_hashes = generator->getFeatureContextHashes(history);
  checkFeatureContexts(feature_indexes, context_hashes, 1);

  feature_indexes = feature_indexes_pair->getWordIndexes(0);
  EXPECT_EQ(2, feature_indexes->size());
  history = {3, 3};
  context_hashes = generator->getFeatureContextHashes(history);
  checkFeatureContexts(feature_indexes, context_hashes, 1);

  feature_indexes = feature_indexes_pair->getWordIndexes(1);
  EXPECT_EQ(4, feature_indexes->size());
  history = {0, 0};
  context_hashes = generator->getFeatureContextHashes(history);
  checkFeatureContexts(feature_indexes, context_hashes, 0);
  history = {3, 0};
  context_hashes = generator->getFeatureContextHashes(history);
  checkFeatureContexts(feature_indexes, context_hashes, 0);

  feature_indexes = feature_indexes_pair->getWordIndexes(2);
  EXPECT_EQ(6, feature_indexes->size());
  history = {2, 0};
  context_hashes = generator->getFeatureContextHashes(history);
  checkFeatureContexts(feature_indexes, context_hashes, 0);
  history = {3, 2};
  context_hashes = generator->getFeatureContextHashes(history);
  checkFeatureContexts(feature_indexes, context_hashes, 0);
  history = {0, 0};
  context_hashes = generator->getFeatureContextHashes(history);
  checkFeatureContexts(feature_indexes, context_hashes, 0);
}

} // namespace oxlm
