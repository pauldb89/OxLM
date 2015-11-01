#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "lbl/feature_context_generator.h"
#include "lbl/feature_filter.h"
#include "lbl/feature_matcher.h"
#include "lbl/hash_builder.h"
#include "lbl/ngram_filter.h"
#include "lbl/word_to_class_index.h"

namespace ar = boost::archive;

namespace oxlm {

class FeatureFilterTest : public testing::Test {
 protected:
  void SetUp() {
    vector<int> data = {2, 3, 1};
    vector<int> classes = {0, 2, 5};
    boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
    boost::shared_ptr<WordToClassIndex> index =
        boost::make_shared<WordToClassIndex>(classes);
    boost::shared_ptr<ContextProcessor> processor =
        boost::make_shared<ContextProcessor>(corpus, 1);
    boost::shared_ptr<FeatureContextGenerator> generator =
        boost::make_shared<FeatureContextGenerator>(1);
    boost::shared_ptr<NGramFilter> ngram_filter =
        boost::make_shared<NGramFilter>();
    boost::shared_ptr<FeatureMatcher> matcher =
        boost::make_shared<FeatureMatcher>(
            corpus, index, processor, generator, ngram_filter);
    FeatureIndexesPairPtr feature_indexes = matcher->getFeatureIndexes();
    filter = boost::make_shared<FeatureFilter>(
        feature_indexes->getClassIndex());
  }

  boost::shared_ptr<FeatureFilter> filter;
  HashBuilder hashBuilder;
};

TEST_F(FeatureFilterTest, TestBasic) {
  vector<int> context = {0};
  Hash context_hash = hashBuilder.compute(context);
  vector<int> expected_indexes = {1};
  EXPECT_EQ(expected_indexes, filter->getIndexes(context_hash));
  EXPECT_FALSE(filter->hasIndex(context_hash, 0));
  EXPECT_TRUE(filter->hasIndex(context_hash, 1));

  context = {2};
  context_hash = hashBuilder.compute(context);
  expected_indexes = {1};
  EXPECT_EQ(expected_indexes, filter->getIndexes(context_hash));
  EXPECT_FALSE(filter->hasIndex(context_hash, 0));
  EXPECT_TRUE(filter->hasIndex(context_hash, 1));

  context = {3};
  context_hash = hashBuilder.compute(context);
  expected_indexes = {0};
  EXPECT_EQ(expected_indexes, filter->getIndexes(context_hash));
  EXPECT_TRUE(filter->hasIndex(context_hash, 0));
  EXPECT_FALSE(filter->hasIndex(context_hash, 1));
}

TEST_F(FeatureFilterTest, TestSerialization) {
  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << filter;

  boost::shared_ptr<FeatureFilter> filter_copy;
  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> filter_copy;

  vector<int> context = {0};
  Hash context_hash = hashBuilder.compute(context);
  vector<int> expected_indexes = {1};
  EXPECT_EQ(expected_indexes, filter_copy->getIndexes(context_hash));
  context = {2};
  context_hash = hashBuilder.compute(context);
  expected_indexes = {1};
  EXPECT_EQ(expected_indexes, filter_copy->getIndexes(context_hash));
  context = {3};
  context_hash = hashBuilder.compute(context);
  expected_indexes = {0};
  EXPECT_EQ(expected_indexes, filter_copy->getIndexes(context_hash));
}

} // namespace oxlm
