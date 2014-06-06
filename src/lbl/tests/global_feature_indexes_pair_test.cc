#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/global_feature_indexes_pair.h"

namespace oxlm {

TEST(GlobalFeatureIndexesPairTest, TestBasic) {
  vector<int> data = {2, 3, 3, 1};
  vector<int> classes = {0, 2, 4};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, 2, 0, 1);
  boost::shared_ptr<FeatureContextGenerator> generator =
      boost::make_shared<FeatureContextGenerator>(2);
  boost::shared_ptr<NGramFilter> filter =
      boost::make_shared<NGramFilter>(corpus, index, processor, generator);
  boost::shared_ptr<FeatureContextMapper> mapper =
      boost::make_shared<FeatureContextMapper>(
          corpus, index, processor, generator, filter);
  GlobalFeatureIndexesPair indexes_pair(index, mapper);

  EXPECT_EQ(7, indexes_pair.getClassIndexes()->size());
  EXPECT_EQ(2, indexes_pair.getWordIndexes(0)->size());
  EXPECT_EQ(6, indexes_pair.getWordIndexes(1)->size());

  indexes_pair.addClassIndex(3, 1);
  indexes_pair.addWordIndex(1, 3, 5);
  vector<int> expected_indexes = {1};
  EXPECT_EQ(expected_indexes, indexes_pair.getClassFeatures(3));
  expected_indexes = {5};
  EXPECT_EQ(expected_indexes, indexes_pair.getWordFeatures(1, 3));
}

} // namespace oxlm
