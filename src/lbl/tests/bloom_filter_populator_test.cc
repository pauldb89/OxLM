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
  boost::shared_ptr<FeatureContextGenerator> generator =
      boost::make_shared<FeatureContextGenerator>(config.feature_context_size);
  boost::shared_ptr<NGramFilter> filter =
      boost::make_shared<NGramFilter>(corpus, index, processor, generator);
  boost::shared_ptr<FeatureContextMapper> mapper =
      boost::make_shared<FeatureContextMapper>(
          corpus, index, processor, generator, filter);
  BloomFilterPopulator populator(corpus, index, mapper, config);

  boost::shared_ptr<BloomFilter<NGram>> bloom_filter = populator.get();

  // Class checks.
  vector<int> context = {0};
  EXPECT_TRUE(bloom_filter->contains(NGram(1, context)));
  context = {0, 0};
  EXPECT_TRUE(bloom_filter->contains(NGram(1, context)));
  context = {2};
  EXPECT_TRUE(bloom_filter->contains(NGram(2, context)));
  context = {2, 0};
  EXPECT_TRUE(bloom_filter->contains(NGram(2, context)));
  context = {3};
  EXPECT_TRUE(bloom_filter->contains(NGram(2, context)));
  context = {3, 2};
  EXPECT_TRUE(bloom_filter->contains(NGram(2, context)));
  context = {3};
  EXPECT_TRUE(bloom_filter->contains(NGram(0, context)));
  context = {3, 3};
  EXPECT_TRUE(bloom_filter->contains(NGram(0, context)));
  context = {2};
  EXPECT_TRUE(bloom_filter->contains(NGram(1, context)));
  context = {2, 0};
  EXPECT_TRUE(bloom_filter->contains(NGram(1, context)));

  context = {1};
  EXPECT_FALSE(bloom_filter->contains(NGram(1, context)));
  context = {1, 3};
  EXPECT_FALSE(bloom_filter->contains(NGram(1, context)));

  // Word checks.
  context = {0};
  EXPECT_TRUE(bloom_filter->contains(NGram(0, 1, context)));
  context = {0, 0};
  EXPECT_TRUE(bloom_filter->contains(NGram(0, 1, context)));
  context = {2};
  EXPECT_TRUE(bloom_filter->contains(NGram(0, 2, context)));
  context = {2, 0};
  EXPECT_TRUE(bloom_filter->contains(NGram(0, 2, context)));
  context = {2};
  EXPECT_TRUE(bloom_filter->contains(NGram(0, 2, context)));
  context = {3, 2};
  EXPECT_TRUE(bloom_filter->contains(NGram(0, 2, context)));
  context = {3};
  EXPECT_TRUE(bloom_filter->contains(NGram(1, 0, context)));
  context = {3, 3};
  EXPECT_TRUE(bloom_filter->contains(NGram(1, 0, context)));
  context = {2};
  EXPECT_TRUE(bloom_filter->contains(NGram(0, 1, context)));
  context = {2, 0};
  EXPECT_TRUE(bloom_filter->contains(NGram(0, 1, context)));

  context = {1};
  EXPECT_FALSE(bloom_filter->contains(NGram(0, 1, context)));
  context = {1, 3};
  EXPECT_FALSE(bloom_filter->contains(NGram(0, 1, context)));
}

} // namespace oxlm
