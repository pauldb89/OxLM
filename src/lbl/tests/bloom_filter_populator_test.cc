#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/bloom_filter_populator.h"

namespace oxlm {

TEST(BloomFilterPopulatorTest, TestBasic) {
  vector<int> data = {2, 3, 3, 1, 2, 2};
  vector<int> classes = {0, 2, 3, 4};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  boost::shared_ptr<WordToClassIndex> index =
      boost::make_shared<WordToClassIndex>(classes);
  ModelData config;
  config.ngram_order = 3;
  config.feature_context_size = 2;
  config.filter_error_rate = 0.1;
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(corpus, config.ngram_order - 1);
  boost::shared_ptr<FeatureContextHasher> hasher =
      boost::make_shared<FeatureContextHasher>(
          corpus, index, processor, config.feature_context_size);
  BloomFilterPopulator populator(corpus, index, hasher, config);

  boost::shared_ptr<BloomFilter<NGramQuery>> bloom_filter = populator.get();

  // Class checks.
  vector<int> context = {0};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(1, context)));
  context = {0, 0};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(1, context)));
  context = {2};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(2, context)));
  context = {2, 0};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(2, context)));
  context = {3};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(2, context)));
  context = {3, 2};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(2, context)));
  context = {3};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(0, context)));
  context = {3, 3};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(0, context)));
  context = {2};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(1, context)));
  context = {2, 0};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(1, context)));

  context = {1};
  EXPECT_FALSE(bloom_filter->contains(NGramQuery(1, context)));
  context = {1, 3};
  EXPECT_FALSE(bloom_filter->contains(NGramQuery(1, context)));

  // Word checks.
  context = {5, 0};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(0, context)));
  context = {5, 0, 0};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(0, context)));
  context = {6, 2};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(0, context)));
  context = {6, 2, 0};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(0, context)));
  context = {6, 2};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(0, context)));
  context = {6, 3, 2};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(0, context)));
  context = {4, 3};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(1, context)));
  context = {4, 3, 3};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(1, context)));
  context = {5, 2};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(0, context)));
  context = {5, 2, 0};
  EXPECT_TRUE(bloom_filter->contains(NGramQuery(0, context)));

  context = {5, 1};
  EXPECT_FALSE(bloom_filter->contains(NGramQuery(0, context)));
  context = {5, 1, 3};
  EXPECT_FALSE(bloom_filter->contains(NGramQuery(0, context)));
}

} // namespace oxlm
