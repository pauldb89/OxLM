#include "gtest/gtest.h"

#include <sstream>
#include <vector>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/class_context_extractor.h"
#include "lbl/context_processor.h"
#include "lbl/feature_context_mapper.h"
#include "lbl/unconstrained_feature_store.h"
#include "lbl/word_to_class_index.h"
#include "utils/constants.h"
#include "utils/testing.h"

using namespace std;
namespace ar = boost::archive;

namespace oxlm {

class UnconstrainedFeatureStoreTest : public ::testing::Test {
 protected:

  virtual void SetUp() {
    vector<int> data = {2, 3, 4, 5, 6};
    vector<int> classes = {0, 2, 7};
    boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
    boost::shared_ptr<WordToClassIndex> index =
        boost::make_shared<WordToClassIndex>(classes);
    boost::shared_ptr<ContextProcessor> processor =
        boost::make_shared<ContextProcessor>(corpus, 1);
    boost::shared_ptr<FeatureContextGenerator> generator =
        boost::make_shared<FeatureContextGenerator>(1);
    boost::shared_ptr<NGramFilter> filter =
        boost::make_shared<NGramFilter>(corpus, index, processor, generator);
    boost::shared_ptr<FeatureContextMapper> mapper =
        boost::make_shared<FeatureContextMapper>(
            corpus, index, processor, generator, filter);
    extractor = boost::make_shared<ClassContextExtractor>(mapper);

    store = UnconstrainedFeatureStore(3, extractor);
    gradient_store =
        boost::make_shared<UnconstrainedFeatureStore>(3, extractor);

    VectorReal values(3);

    context1 = {2};
    values << 1, 2, 3;
    store.update(context1, values);

    context2 = {3};
    values << 6, 5, 4;
    store.update(context2, values);

    values << 1.5, 1.25, 1;
    gradient_store->update(context2, values);

    context3 = {4};
    values << 0.5, 0.75, 1;
    gradient_store->update(context3, values);
  }

  vector<int> context1, context2, context3;
  boost::shared_ptr<FeatureContextExtractor> extractor;
  UnconstrainedFeatureStore store;
  boost::shared_ptr<UnconstrainedFeatureStore> gradient_store;
};

TEST_F(UnconstrainedFeatureStoreTest, TestBasic) {
  UnconstrainedFeatureStore feature_store(3, extractor);

  EXPECT_MATRIX_NEAR(
      VectorReal::Zero(3), feature_store.get(context1), EPS);

  VectorReal values(3);
  values << 1, 2, 3;
  feature_store.update(context1, values);
  EXPECT_MATRIX_NEAR(values, feature_store.get(context1), EPS);

  feature_store.update(context1, values);
  EXPECT_MATRIX_NEAR(2 * values, feature_store.get(context1), EPS);

  EXPECT_EQ(1, feature_store.size());
  feature_store.clear();
  EXPECT_EQ(0, feature_store.size());
}

TEST_F(UnconstrainedFeatureStoreTest, TestUpdateRegularizer) {
  store.l2GradientUpdate(gradient_store, 0.5);
  EXPECT_NEAR(22.75, store.l2Objective(gradient_store, 1), EPS);

  VectorReal expected_values(3);
  expected_values << 0.5, 1, 1.5;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context1), EPS);

  expected_values << 3, 2.5, 2;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context2), EPS);
}

TEST_F(UnconstrainedFeatureStoreTest, TestUpdateStore) {
  store.update(gradient_store);

  EXPECT_EQ(3, store.size());
  VectorReal expected_values(3);
  expected_values << 1, 2, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context1), EPS);
  expected_values << 7.5, 6.25, 5;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context2), EPS);
  expected_values << 0.5, 0.75, 1;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context3), EPS);
}

TEST_F(UnconstrainedFeatureStoreTest, TestUpdateSquared) {
  store.updateSquared(gradient_store);

  EXPECT_EQ(3, store.size());
  VectorReal expected_values(3);
  expected_values << 1, 2, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context1), EPS);
  expected_values << 8.25, 6.5625, 5;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context2), EPS);
  expected_values << 0.25, 0.5625, 1;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context3), EPS);
}

TEST_F(UnconstrainedFeatureStoreTest, TestUpdateAdaGrad) {
  store.updateAdaGrad(gradient_store, gradient_store, 1);

  EXPECT_EQ(3, store.size());
  VectorReal expected_values(3);
  expected_values << 1, 2, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context1), EPS);
  expected_values << 4.775255, 3.881966, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context2), EPS);
  expected_values << -0.707106, -0.866025, -1;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context3), EPS);
}

TEST_F(UnconstrainedFeatureStoreTest, TestSerialization) {
  boost::shared_ptr<GlobalFeatureStore> store_ptr =
      boost::make_shared<UnconstrainedFeatureStore>(store);
  boost::shared_ptr<GlobalFeatureStore> store_copy_ptr;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive output_stream(stream, ar::no_header);
  output_stream << store_ptr;

  ar::binary_iarchive input_stream(stream, ar::no_header);
  input_stream >> store_copy_ptr;

  boost::shared_ptr<UnconstrainedFeatureStore> expected_ptr =
      UnconstrainedFeatureStore::cast(store_ptr);
  boost::shared_ptr<UnconstrainedFeatureStore> actual_ptr =
      UnconstrainedFeatureStore::cast(store_copy_ptr);

  EXPECT_NE(nullptr, expected_ptr);
  EXPECT_NE(nullptr, actual_ptr);
  EXPECT_EQ(*expected_ptr, *actual_ptr);
}

}
