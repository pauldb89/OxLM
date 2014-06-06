#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/class_context_extractor.h"
#include "lbl/context_processor.h"
#include "lbl/feature_context_mapper.h"
#include "lbl/feature_exact_filter.h"
#include "lbl/word_to_class_index.h"

namespace ar = boost::archive;

namespace oxlm {

class FeatureExactFilterTest : public testing::Test {
 protected:
  void SetUp() {
    GlobalFeatureIndexesPtr feature_indexes =
        boost::make_shared<GlobalFeatureIndexes>(3);
    (*feature_indexes)[0] = {0, 2, 5};
    (*feature_indexes)[1] = {1, 6, 8};
    (*feature_indexes)[2] = {2, 3, 5};

    // Mini-corpus for producing a few feature contexts, unrelated to the feature
    // indexes defined above.
    vector<int> data = {2, 3, 4};
    vector<int> classes = {0, 2, 5};
    boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
    boost::shared_ptr<WordToClassIndex> index =
        boost::make_shared<WordToClassIndex>(classes);
    boost::shared_ptr<ContextProcessor> processor =
        boost::make_shared<ContextProcessor>(corpus, 1);
    boost::shared_ptr<FeatureContextGenerator> generator =
        boost::make_shared<FeatureContextGenerator>(1);
    boost::shared_ptr<NGramFilter> ngram_filter =
        boost::make_shared<NGramFilter>(corpus, index, processor, generator);
    boost::shared_ptr<FeatureContextMapper> mapper =
        boost::make_shared<FeatureContextMapper>(
            corpus, index, processor, generator, ngram_filter);
    boost::shared_ptr<ClassContextExtractor> extractor =
        boost::make_shared<ClassContextExtractor>(mapper);
    filter = boost::make_shared<FeatureExactFilter>(feature_indexes, extractor);
  }

  boost::shared_ptr<FeatureFilter> filter;
};

TEST_F(FeatureExactFilterTest, TestBasic) {
  // Context [0] maps to 0.
  vector<int> context = {0};
  FeatureContext feature_context(context);
  vector<int> expected_indexes = {0, 2, 5};
  EXPECT_EQ(expected_indexes, filter->getIndexes(feature_context));
  // Context [2] maps to 1.
  context = {2};
  feature_context = FeatureContext(context);
  expected_indexes = {1, 6, 8};
  EXPECT_EQ(expected_indexes, filter->getIndexes(feature_context));
  // Context [3] maps to 2.
  context = {3};
  feature_context = FeatureContext(context);
  expected_indexes = {2, 3, 5};
  EXPECT_EQ(expected_indexes, filter->getIndexes(feature_context));
}

TEST_F(FeatureExactFilterTest, TestSerialization) {
  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << filter;

  boost::shared_ptr<FeatureFilter> filter_copy;
  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> filter_copy;

  // Context [0] maps to 0.
  vector<int> context = {0};
  FeatureContext feature_context(context);
  vector<int> expected_indexes = {0, 2, 5};
  EXPECT_EQ(expected_indexes, filter_copy->getIndexes(feature_context));
  // Context [2] maps to 1.
  context = {2};
  feature_context = FeatureContext(context);
  expected_indexes = {1, 6, 8};
  EXPECT_EQ(expected_indexes, filter_copy->getIndexes(feature_context));
  // Context [3] maps to 2.
  context = {3};
  feature_context = FeatureContext(context);
  expected_indexes = {2, 3, 5};
  EXPECT_EQ(expected_indexes, filter_copy->getIndexes(feature_context));
}

} // namespace oxlm
