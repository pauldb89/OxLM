#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "lbl/class_context_extractor.h"
#include "lbl/class_context_hasher.h"
#include "lbl/collision_global_feature_store.h"
#include "lbl/context_processor.h"
#include "lbl/corpus.h"
#include "lbl/feature_context_mapper.h"
#include "lbl/feature_exact_filter.h"
#include "lbl/feature_matcher.h"
#include "lbl/ngram_filter.h"
#include "lbl/word_to_class_index.h"
#include "utils/constants.h"
#include "utils/testing.h"

namespace ar = boost::archive;

namespace oxlm {

class CollisionGlobalFeatureStoreTest : public testing::Test {
 protected:
  void SetUp() {
    int vector_size = 3;
    int hash_space = 10;
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
        boost::make_shared<NGramFilter>(corpus, index, processor, generator);
    boost::shared_ptr<FeatureContextMapper> mapper =
        boost::make_shared<FeatureContextMapper>(
            corpus, index, processor, generator, ngram_filter);
    boost::shared_ptr<ClassContextExtractor> extractor =
        boost::make_shared<ClassContextExtractor>(mapper);
    boost::shared_ptr<GlobalCollisionSpace> space =
        boost::make_shared<GlobalCollisionSpace>(hash_space);
    boost::shared_ptr<FeatureMatcher> feature_matcher =
        boost::make_shared<FeatureMatcher>(
            corpus, index, processor, generator, ngram_filter, mapper);

    auto feature_indexes_pair = feature_matcher->getGlobalFeatures();
    auto feature_indexes = feature_indexes_pair->getClassIndexes();
    boost::shared_ptr<FeatureExactFilter> filter =
        boost::make_shared<FeatureExactFilter>(feature_indexes, extractor);
    boost::shared_ptr<ClassContextHasher> hasher =
        boost::make_shared<ClassContextHasher>(hash_space);

    CollisionMinibatchFeatureStore g_store(
        vector_size, hash_space, 3, hasher, filter);

    context = {2, 3, 4};
    VectorReal values(3);
    values << 4, 2, 5;
    g_store.update(context, values);

    store = CollisionGlobalFeatureStore(
        vector_size, hash_space, 3, space, hasher, filter);
    gradient_store = boost::make_shared<CollisionMinibatchFeatureStore>(
        g_store);
  }

  vector<int> context;
  CollisionGlobalFeatureStore store;
  boost::shared_ptr<MinibatchFeatureStore> gradient_store;
};

TEST_F(CollisionGlobalFeatureStoreTest, TestUpdateSquared) {
  store.updateSquared(gradient_store);

  VectorReal expected_values(3);
  expected_values << 113, 12, 131;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context), EPS);
  EXPECT_NEAR(113, store.getValue(0, context), EPS);
  EXPECT_NEAR(12, store.getValue(1, context), EPS);
  EXPECT_NEAR(131, store.getValue(2, context), EPS);
}

TEST_F(CollisionGlobalFeatureStoreTest, TestUpdateAdaGrad) {
  store.updateSquared(gradient_store);
  boost::shared_ptr<GlobalFeatureStore> adagrad_store =
      boost::make_shared<CollisionGlobalFeatureStore>(store);

  store.updateAdaGrad(gradient_store, adagrad_store, 1);
  VectorReal expected_values(3);
  expected_values << 110, 9, 128;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context), EPS);
  EXPECT_NEAR(110, store.getValue(0, context), EPS);
  EXPECT_NEAR(9, store.getValue(1, context), EPS);
  EXPECT_NEAR(128, store.getValue(2, context), EPS);
}

TEST_F(CollisionGlobalFeatureStoreTest, TestUpdateRegularizer) {
  store.updateSquared(gradient_store);
  store.l2GradientUpdate(gradient_store, 0.5);

  EXPECT_NEAR(2092.75, store.l2Objective(gradient_store, 1), EPS);
  VectorReal expected_values(3);
  expected_values << 56.5, 6, 65.5;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context), EPS);
  EXPECT_NEAR(56.5, store.getValue(0, context), EPS);
  EXPECT_NEAR(6, store.getValue(1, context), EPS);
  EXPECT_NEAR(65.5, store.getValue(2, context), EPS);
}

TEST_F(CollisionGlobalFeatureStoreTest, TestSerialization) {
  store.updateSquared(gradient_store);
  boost::shared_ptr<GlobalFeatureStore> store_ptr =
      boost::make_shared<CollisionGlobalFeatureStore>(store);
  boost::shared_ptr<GlobalFeatureStore> store_copy_ptr;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive output_stream(stream);
  output_stream << store_ptr;

  ar::binary_iarchive input_stream(stream);
  input_stream >> store_copy_ptr;

  boost::shared_ptr<CollisionGlobalFeatureStore> expected_ptr =
      CollisionGlobalFeatureStore::cast(store_ptr);
  boost::shared_ptr<CollisionGlobalFeatureStore> actual_ptr =
      CollisionGlobalFeatureStore::cast(store_copy_ptr);

  EXPECT_NE(nullptr, expected_ptr);
  EXPECT_NE(nullptr, actual_ptr);
  EXPECT_EQ(*expected_ptr, *actual_ptr);

  VectorReal expected_values(3);
  expected_values << 113, 12, 131;
  EXPECT_MATRIX_NEAR(expected_values, actual_ptr->get(context), EPS);
}

} // namespace oxlm
