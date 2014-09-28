#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/corpus.h"

namespace oxlm {

TEST(CorpusTest, TestBasic) {
  boost::shared_ptr<Vocabulary> vocab = boost::make_shared<Vocabulary>();
  Corpus corpus("training.en", vocab, false);

  EXPECT_EQ(13554, corpus.size());
  EXPECT_EQ(vocab->convert("musharraf"), corpus.at(0));
  EXPECT_EQ(vocab->convert("'s"), corpus.at(1));
  EXPECT_EQ(vocab->convert("last"), corpus.at(2));
  EXPECT_EQ(vocab->convert("act"), corpus.at(3));
  EXPECT_EQ(vocab->convert("?"), corpus.at(4));
  EXPECT_EQ(vocab->convert("</s>"), corpus.at(5));
}

} // namespace oxlm
