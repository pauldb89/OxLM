#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/feature_context_hasher.h"

namespace oxlm {

TEST(FeatureContextHasherTest, TestLongerHistory) {
  vector<int> data = {2, 3, 4, 5, 6, 1};
  vector<int> classes = {0, 2, 4, 7};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, 5, 0, 1);
  FeatureContextHasher hasher(corpus, index, processor, 3);

  vector<int> history = {6, 5, 4, 3, 2};
  vector<int> class_context_ids = hasher.getClassContextIds(history);
  EXPECT_EQ(3, class_context_ids.size());
  EXPECT_EQ(15, class_context_ids[0]);
  EXPECT_EQ(16, class_context_ids[1]);
  EXPECT_EQ(17, class_context_ids[2]);

  vector<int> word_context_ids = hasher.getWordContextIds(0, history);
  EXPECT_EQ(3, word_context_ids.size());
  EXPECT_EQ(0, word_context_ids[0]);
  EXPECT_EQ(1, word_context_ids[1]);
  EXPECT_EQ(2, word_context_ids[2]);

  EXPECT_EQ(36, hasher.getNumContexts());
  EXPECT_EQ(18, hasher.getNumClassContexts());
  EXPECT_EQ(3, hasher.getNumWordContexts(0));
  EXPECT_EQ(6, hasher.getNumWordContexts(1));
  EXPECT_EQ(9, hasher.getNumWordContexts(2));
}

TEST(FeatureContextHasherTest, TestShorterHistory) {
  vector<int> classes = {0, 2, 4, 7};
  vector<int> data = {2, 3, 4, 5, 6, 1};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, 3, 0, 1);
  FeatureContextHasher hasher(corpus, index, processor, 5);

  vector<int> history = {6, 5, 4};
  vector<int> class_context_ids = hasher.getClassContextIds(history);
  EXPECT_EQ(3, class_context_ids.size());
  EXPECT_EQ(15, class_context_ids[0]);
  EXPECT_EQ(16, class_context_ids[1]);
  EXPECT_EQ(17, class_context_ids[2]);

  vector<int> word_context_ids = hasher.getWordContextIds(0, history);
  EXPECT_EQ(3, word_context_ids.size());
  EXPECT_EQ(0, word_context_ids[0]);
  EXPECT_EQ(1, word_context_ids[1]);
  EXPECT_EQ(2, word_context_ids[2]);
}

} // namespace oxlm
