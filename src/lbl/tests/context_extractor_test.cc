#include "gtest/gtest.h"

#include "lbl/context_extractor.h"

namespace oxlm {

TEST(ContextExtractorTest, TestExtract) {
  Corpus corpus = {2, 3, 4, 1, 5, 6};
  ContextExtractor extractor(corpus, 3, 0, 1);

  vector<WordId> expected_context = {0, 0, 0};
  EXPECT_EQ(expected_context, extractor.extract(0));
  expected_context = {2, 0, 0};
  EXPECT_EQ(expected_context, extractor.extract(1));
  expected_context = {3, 2, 0};
  EXPECT_EQ(expected_context, extractor.extract(2));
  expected_context = {4, 3, 2};
  EXPECT_EQ(expected_context, extractor.extract(3));
  expected_context = {0, 0, 0};
  EXPECT_EQ(expected_context, extractor.extract(4));
  expected_context = {5, 0, 0};
  EXPECT_EQ(expected_context, extractor.extract(5));
}

} // namespace oxlm
