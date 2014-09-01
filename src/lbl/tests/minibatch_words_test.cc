#include "gtest/gtest.h"

#include "lbl/minibatch_words.h"

namespace oxlm {

TEST(MinibatchWordsTest, TestBasic) {
  MinibatchWords words;

  for (int i = 0; i < 3; ++i) {
    words.addContextWord(i);
    words.addOutputWord(i);
  }

  for (int i = 2; i < 5; ++i) {
    words.addContextWord(i);
    words.addOutputWord(i);
  }

  unordered_set<int> expected_set = {0, 1, 2, 3, 4};
  EXPECT_EQ(expected_set, words.getContextWordsSet());
  EXPECT_EQ(expected_set, words.getOutputWordsSet());

  words.transform();
  vector<int> expected_words = {0, 1, 2, 3, 4};
  EXPECT_EQ(expected_words, words.getContextWords());
  EXPECT_EQ(expected_words, words.getOutputWords());
}

TEST(MinibatchWordsTest, TestMerge) {
  MinibatchWords w1, w2;

  for (int i = 0; i < 3; ++i) {
    w1.addContextWord(i);
    w1.addOutputWord(i);
  }

  for (int i = 2; i < 5; ++i) {
    w2.addContextWord(i);
    w2.addOutputWord(i);
  }

  w1.merge(w2);

  unordered_set<int> expected_set = {0, 1, 2, 3, 4};
  EXPECT_EQ(expected_set, w1.getContextWordsSet());
  EXPECT_EQ(expected_set, w1.getOutputWordsSet());

  w1.transform();
  vector<int> expected_words = {0, 1, 2, 3, 4};
  EXPECT_EQ(expected_words, w1.getContextWords());
  EXPECT_EQ(expected_words, w1.getOutputWords());
}

} // namespace oxlm
