#include "gtest/gtest.h"

#include "lbl/parallel_corpus.h"
#include "lbl/parallel_vocabulary.h"

namespace oxlm {

TEST(ParallelCorpusTest, TestBasic) {
  boost::shared_ptr<ParallelVocabulary> vocab =
      boost::make_shared<ParallelVocabulary>();
  ParallelCorpus corpus("training.fr-en", "training.gdfa", vocab, false);

  EXPECT_EQ(13059, corpus.size());
  EXPECT_EQ(14441, corpus.sourceSize());

  EXPECT_EQ(vocab->convert("$"), corpus.at(0));
  EXPECT_EQ(vocab->convert("10,000"), corpus.at(1));
  EXPECT_EQ(vocab->convert("gold"), corpus.at(2));
  EXPECT_EQ(vocab->convert("?"), corpus.at(3));
  EXPECT_EQ(vocab->convert("</s>"), corpus.at(4));

  EXPECT_EQ(vocab->convertSource("l'or"), corpus.sourceAt(0));
  EXPECT_EQ(vocab->convertSource("à"), corpus.sourceAt(1));
  EXPECT_EQ(vocab->convertSource("10.000"), corpus.sourceAt(2));
  EXPECT_EQ(vocab->convertSource("dollars"), corpus.sourceAt(3));
  EXPECT_EQ(vocab->convertSource("l'once"), corpus.sourceAt(4));
  EXPECT_EQ(vocab->convertSource("?"), corpus.sourceAt(5));
  EXPECT_EQ(vocab->convertSource("</s>"), corpus.sourceAt(6));

  vector<long long> expected_links = {1, 3};
  EXPECT_EQ(expected_links, corpus.getLinks(0));
  expected_links = {2};
  EXPECT_EQ(expected_links, corpus.getLinks(1));
  expected_links = {4};
  EXPECT_EQ(expected_links, corpus.getLinks(2));
  expected_links = {5};
  EXPECT_EQ(expected_links, corpus.getLinks(3));
  expected_links = {6};
  EXPECT_EQ(expected_links, corpus.getLinks(4));

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(corpus.isAligned(i));
  }
}

} // namespace oxlm
