#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/corpus.h"
#include "lbl/feature_indexes_pair.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

TEST(FeatureIndexesPairTest, TestBasic) {
  vector<int> data = {2, 3, 3, 1};
  vector<int> classes = {0, 2, 4};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);
  FeatureIndexesPair indexes_pair(index);

  indexes_pair.addClassIndex(3, 1);
  indexes_pair.addWordIndex(1, 3, 5);
  vector<int> expected_indexes = {1};
  EXPECT_EQ(expected_indexes, indexes_pair.getClassFeatures(3));
  expected_indexes = {5};
  EXPECT_EQ(expected_indexes, indexes_pair.getWordFeatures(1, 3));
}

} // namespace oxlm
