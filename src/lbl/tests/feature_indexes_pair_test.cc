#include "gtest/gtest.h"

#include "lbl/feature_indexes_pair.h"

namespace oxlm {

TEST(FeatureIndexesPairTest, TestBasic) {
  int num_classes = 3;
  FeatureIndexesPair indexes_pair(num_classes);

  EXPECT_EQ(0, indexes_pair.getClassIndexes()->size());
  for (int i = 0; i < num_classes; ++i) {
    EXPECT_EQ(0, indexes_pair.getWordIndexes(i)->size());
  }

  indexes_pair.addClassIndex(1, 2);
  indexes_pair.addWordIndex(2, 1, 5);
  EXPECT_EQ(1, indexes_pair.getClassIndexes()->size());
  EXPECT_EQ(1, indexes_pair.getWordIndexes(2)->size());

  unordered_set<int> expected_indexes = {2};
  EXPECT_EQ(expected_indexes, indexes_pair.getClassFeatures(1));
  expected_indexes = {5};
  EXPECT_EQ(expected_indexes, indexes_pair.getWordFeatures(2, 1));

  unordered_set<int> new_indexes = {1};
  indexes_pair.setClassIndexes(2, new_indexes);
  new_indexes = {3};
  indexes_pair.setWordIndexes(1, 2, new_indexes);

  expected_indexes = {1};
  EXPECT_EQ(expected_indexes, indexes_pair.getClassFeatures(2));
  expected_indexes = {3};
  EXPECT_EQ(expected_indexes, indexes_pair.getWordFeatures(1, 2));
}

} // namespace oxlm
