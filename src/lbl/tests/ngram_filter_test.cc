#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/ngram_filter.h"

namespace oxlm {

TEST(NGramFilterTest, TestFilterEnabled) {
  vector<Hash> context_hashes = {1, 2, 3};

  hash<HashedNGram> hasher;
  unordered_set<size_t> valid_ngrams;
  for (size_t i = 0; i < context_hashes.size(); ++i) {
    size_t ngram_hash = hasher(HashedNGram(1, 1, context_hashes[i]));
    valid_ngrams.insert(ngram_hash);
  }

  NGramFilter filter(valid_ngrams);

  EXPECT_EQ(3, filter.filter(1, 1, context_hashes).size());
  EXPECT_EQ(0, filter.filter(0, 1, context_hashes).size());
  EXPECT_EQ(0, filter.filter(1, 0, context_hashes).size());
  vector<size_t> other_hashes = {4, 5, 6};
  EXPECT_EQ(0, filter.filter(1, 1, other_hashes).size());
  vector<size_t> mixed_hashes = {1, 5, 3};
  EXPECT_EQ(2, filter.filter(1, 1, mixed_hashes).size());
}

TEST(NGramFilterTest, TestDisabled) {
  NGramFilter filter;

  vector<Hash> context_hashes = {2};
  EXPECT_EQ(1, filter.filter(2, 1, context_hashes).size());
  EXPECT_EQ(1, filter.filter(3, 2, context_hashes).size());
  EXPECT_EQ(1, filter.filter(4, 2, context_hashes).size());
}

} // namespace oxlm
