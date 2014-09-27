#include "gtest/gtest.h"

#include "lbl/parallel_vocabulary.h"

namespace oxlm {

TEST(ParallelVocabularyTest, TestTarget) {
  ParallelVocabulary vocab;

  EXPECT_EQ(2, vocab.convert("anna"));
  EXPECT_EQ(3, vocab.convert("has"));
  EXPECT_EQ(4, vocab.convert("apples"));

  EXPECT_EQ(0, vocab.convert("<s>"));
  EXPECT_EQ(1, vocab.convert("</s>"));
  EXPECT_EQ(2, vocab.convert("anna"));
  EXPECT_EQ(3, vocab.convert("has"));
  EXPECT_EQ(4, vocab.convert("apples"));

  EXPECT_EQ("<s>", vocab.convert(0));
  EXPECT_EQ("</s>", vocab.convert(1));
  EXPECT_EQ("anna", vocab.convert(2));
  EXPECT_EQ("has", vocab.convert(3));
  EXPECT_EQ("apples", vocab.convert(4));

  EXPECT_EQ(5, vocab.size());
}

TEST(ParallelVocabularyTest, TestSource) {
  ParallelVocabulary vocab;

  EXPECT_EQ(2, vocab.convertSource("anna"));
  EXPECT_EQ(3, vocab.convertSource("has"));
  EXPECT_EQ(4, vocab.convertSource("apples"));

  EXPECT_EQ(0, vocab.convertSource("<s>"));
  EXPECT_EQ(1, vocab.convertSource("</s>"));
  EXPECT_EQ(2, vocab.convertSource("anna"));
  EXPECT_EQ(3, vocab.convertSource("has"));
  EXPECT_EQ(4, vocab.convertSource("apples"));

  EXPECT_EQ("<s>", vocab.convertSource(0));
  EXPECT_EQ("</s>", vocab.convertSource(1));
  EXPECT_EQ("anna", vocab.convertSource(2));
  EXPECT_EQ("has", vocab.convertSource(3));
  EXPECT_EQ("apples", vocab.convertSource(4));

  EXPECT_EQ(5, vocab.sourceSize());
}

} // namespace oxlm
