#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/ngram_filter.h"

namespace oxlm {

class NGramFilterTest : public testing::Test {
 protected:
  void SetUp() {
    vector<int> data = {2, 3, 2, 3, 1, 3, 3, 4, 2, 2, 4, 3, 3, 2, 2, 1};
    vector<int> classes = {0, 2, 3, 5};
    corpus = boost::make_shared<Corpus>(data);
    index = boost::make_shared<WordToClassIndex>(classes);
    processor = boost::make_shared<ContextProcessor>(corpus, 1);
    generator = boost::make_shared<FeatureContextGenerator>(1);
  }

  boost::shared_ptr<Corpus> corpus;
  boost::shared_ptr<WordToClassIndex> index;
  boost::shared_ptr<ContextProcessor> processor;
  boost::shared_ptr<FeatureContextGenerator> generator;
};

TEST_F(NGramFilterTest, TestFilterMaxCount) {
  NGramFilter filter(corpus, index, processor, generator, 3);

  vector<int> context = {2};
  FeatureContext feature_context(context);
  vector<FeatureContext> feature_contexts = {feature_context};
  EXPECT_EQ(0, filter.filter(2, 1, feature_contexts).size());
  EXPECT_EQ(1, filter.filter(3, 2, feature_contexts).size());
  EXPECT_EQ(0, filter.filter(4, 2, feature_contexts).size());

  context = {3};
  feature_context = FeatureContext(context);
  feature_contexts = {feature_context};
  EXPECT_EQ(1, filter.filter(2, 1, feature_contexts).size());
  EXPECT_EQ(1, filter.filter(3, 2, feature_contexts).size());
  EXPECT_EQ(0, filter.filter(4, 2, feature_contexts).size());

  context = {4};
  feature_context = FeatureContext(context);
  feature_contexts = {feature_context};
  EXPECT_EQ(0, filter.filter(2, 1, feature_contexts).size());
  EXPECT_EQ(0, filter.filter(3, 2, feature_contexts).size());
  EXPECT_EQ(0, filter.filter(4, 2, feature_contexts).size());
}

TEST_F(NGramFilterTest, TestFilterMinFreq) {
  NGramFilter filter(corpus, index, processor, generator, 0, 2);

  vector<int> context = {2};
  FeatureContext feature_context(context);
  vector<FeatureContext> feature_contexts = {feature_context};
  EXPECT_EQ(1, filter.filter(2, 1, feature_contexts).size());
  EXPECT_EQ(1, filter.filter(3, 2, feature_contexts).size());
  EXPECT_EQ(0, filter.filter(4, 2, feature_contexts).size());

  context = {3};
  feature_context = FeatureContext(context);
  feature_contexts = {feature_context};
  EXPECT_EQ(1, filter.filter(2, 1, feature_contexts).size());
  EXPECT_EQ(1, filter.filter(3, 2, feature_contexts).size());
  EXPECT_EQ(0, filter.filter(4, 2, feature_contexts).size());

  context = {4};
  feature_context = FeatureContext(context);
  feature_contexts = {feature_context};
  EXPECT_EQ(0, filter.filter(2, 1, feature_contexts).size());
  EXPECT_EQ(0, filter.filter(3, 2, feature_contexts).size());
  EXPECT_EQ(0, filter.filter(4, 2, feature_contexts).size());
}

TEST_F(NGramFilterTest, TestDisabled) {
  NGramFilter filter(corpus, index, processor, generator);

  vector<int> context = {2};
  FeatureContext feature_context(context);
  vector<FeatureContext> feature_contexts = {feature_context};
  EXPECT_EQ(1, filter.filter(2, 1, feature_contexts).size());
  EXPECT_EQ(1, filter.filter(3, 2, feature_contexts).size());
  EXPECT_EQ(1, filter.filter(4, 2, feature_contexts).size());

  context = {3};
  feature_context = FeatureContext(context);
  feature_contexts = {feature_context};
  EXPECT_EQ(1, filter.filter(2, 1, feature_contexts).size());
  EXPECT_EQ(1, filter.filter(3, 2, feature_contexts).size());
  EXPECT_EQ(1, filter.filter(4, 2, feature_contexts).size());

  context = {4};
  feature_context = FeatureContext(context);
  feature_contexts = {feature_context};
  EXPECT_EQ(1, filter.filter(2, 1, feature_contexts).size());
  EXPECT_EQ(1, filter.filter(3, 2, feature_contexts).size());
  EXPECT_EQ(1, filter.filter(4, 2, feature_contexts).size());
}

} // namespace oxlm
