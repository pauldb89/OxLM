#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/minibatch_feature_indexes_pair.h"

namespace oxlm {

TEST(MinibatchFeatureIndexesPairTest, TestBasic) {
  vector<int> classes = {0, 2, 4};
  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);
  MinibatchFeatureIndexesPair indexes_pair(index);

  EXPECT_EQ(0, indexes_pair.getClassIndexes()->size());
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(0, indexes_pair.getWordIndexes(i)->size());
  }

  vector<int> new_indexes = {1};
  indexes_pair.setClassIndexes(2, new_indexes);
  new_indexes = {3};
  indexes_pair.setWordIndexes(1, 2, new_indexes);

  vector<int> expected_indexes = {1};
  EXPECT_EQ(expected_indexes, indexes_pair.getClassIndexes()->at(2));
  expected_indexes = {3};
  EXPECT_EQ(expected_indexes, indexes_pair.getWordIndexes(1)->at(2));
}

} // namespace oxlm
