#include "gtest/gtest.h"

#include "lbl/vocabulary.h"

namespace oxlm {

TEST(VocabularyTest, TestBasic) {
  Vocabulary vocab;

  EXPECT_TRUE(vocab.contains("<s>"));
  EXPECT_TRUE(vocab.contains("</s>"));

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

} // namespace oxlm
