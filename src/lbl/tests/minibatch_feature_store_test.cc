#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/class_context_hasher.h"
#include "lbl/context_processor.h"
#include "lbl/corpus.h"
#include "lbl/feature_filter.h"
#include "lbl/feature_matcher.h"
#include "lbl/minibatch_feature_store.h"
#include "lbl/ngram_filter.h"
#include "lbl/word_to_class_index.h"
#include "utils/constants.h"
#include "utils/testing.h"

namespace oxlm {

class MinibatchFeatureStoreTest : public testing::Test {
 protected:
  void SetUp() {
    int vector_size = 3;
    int hash_space = 10000000;
    vector<int> data = {4, 3, 2, 1, 4, 3, 2, 2, 4, 3, 2, 3};
    vector<int> classes = {0, 2, 3, 5};
    boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
    boost::shared_ptr<WordToClassIndex> index =
        boost::make_shared<WordToClassIndex>(classes);
    boost::shared_ptr<ContextProcessor> processor =
        boost::make_shared<ContextProcessor>(corpus, 3);
    boost::shared_ptr<FeatureContextGenerator> generator =
        boost::make_shared<FeatureContextGenerator>(3);
    boost::shared_ptr<NGramFilter> ngram_filter =
        boost::make_shared<NGramFilter>();
    boost::shared_ptr<ClassContextHasher> hasher =
        boost::make_shared<ClassContextHasher>();
    boost::shared_ptr<FeatureMatcher> feature_matcher =
        boost::make_shared<FeatureMatcher>(
            corpus, index, processor, generator, ngram_filter);

    auto feature_indexes_pair = feature_matcher->getFeatureIndexes();
    auto feature_indexes = feature_indexes_pair->getClassIndexes();
    boost::shared_ptr<FeatureFilter> filter =
        boost::make_shared<FeatureFilter>(feature_indexes);

    store = boost::make_shared<MinibatchFeatureStore>(
        vector_size, hash_space, 3, hasher, filter);

    g_store = boost::make_shared<MinibatchFeatureStore>(
        vector_size, hash_space, 3, hasher, filter);

    context = {2, 3, 4};
    VectorReal values(3);
    values << 4, 2, 5;
    g_store->update(context, values);
  }

  vector<int> context;
  boost::shared_ptr<MinibatchFeatureStore> store;
  boost::shared_ptr<MinibatchFeatureStore> g_store;
};

TEST_F(MinibatchFeatureStoreTest, TestBasic) {
  VectorReal expected_values = VectorReal::Zero(3);
  EXPECT_MATRIX_NEAR(expected_values, store->get(context), EPS);

  VectorReal values(3);
  values << 4, 2, 5;
  store->update(context, values);
  // Due to collisions we don't get 3 x values.
  expected_values << 12, 6, 15;
  EXPECT_MATRIX_NEAR(expected_values, store->get(context), EPS);
  EXPECT_EQ(9, store->size());
}

TEST_F(MinibatchFeatureStoreTest, TestUpdateValue) {
  VectorReal expected_values = VectorReal::Zero(3);
  EXPECT_MATRIX_NEAR(expected_values, store->get(context), EPS);

  store->updateValue(0, context, 3);
  expected_values << 9, 0, 0;
  EXPECT_MATRIX_NEAR(expected_values, store->get(context), EPS);
}

TEST_F(MinibatchFeatureStoreTest, TestGradientUpdate) {
  store->update(g_store);

  VectorReal expected_values(3);
  expected_values << 12, 6, 15;
  EXPECT_MATRIX_NEAR(expected_values, store->get(context), EPS);
  EXPECT_EQ(9, store->size());
}

TEST_F(MinibatchFeatureStoreTest, TestClear) {
  EXPECT_EQ(9, g_store->size());

  g_store->clear();
  EXPECT_EQ(0, g_store->size());
  EXPECT_EQ(VectorReal::Zero(3), g_store->get(context));
}

} // namespace oxlm
