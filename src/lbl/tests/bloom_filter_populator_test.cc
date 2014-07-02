#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>

#include "lbl/bloom_filter_populator.h"

namespace ar = boost::archive;

namespace oxlm {

class BloomFilterPopulatorTest : public testing::Test {
 protected:
  void SetUp() {
    vector<int> data = {2, 3, 3, 1, 2, 2};
    vector<int> classes = {0, 2, 3, 4};
    boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
    boost::shared_ptr<WordToClassIndex> index =
        boost::make_shared<WordToClassIndex>(classes);
    boost::shared_ptr<ModelData> config = boost::make_shared<ModelData>();
    config->ngram_order = 3;
    config->feature_context_size = 2;
    config->filter_error_rate = 0.1;
    boost::shared_ptr<ContextProcessor> processor =
        boost::make_shared<ContextProcessor>(corpus, config->ngram_order - 1);
    boost::shared_ptr<FeatureContextGenerator> generator =
        boost::make_shared<FeatureContextGenerator>(config->feature_context_size);
    boost::shared_ptr<NGramFilter> filter =
        boost::make_shared<NGramFilter>(corpus, index, processor, generator);
    boost::shared_ptr<FeatureContextMapper> mapper =
        boost::make_shared<FeatureContextMapper>(
            corpus, index, processor, generator, filter);
    populator = BloomFilterPopulator(corpus, index, mapper, config);
  }

  BloomFilterPopulator populator;
};

TEST_F(BloomFilterPopulatorTest, TestBasic) {
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

TEST_F(BloomFilterPopulatorTest, TestSerialization) {
  BloomFilterPopulator populator_copy;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << populator;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> populator_copy;

  EXPECT_EQ(populator, populator_copy);
}

} // namespace oxlm
