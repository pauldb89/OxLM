#include "gtest/gtest.h"

#include <sstream>
#include <vector>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/sparse_global_feature_store.h"
#include "lbl/sparse_minibatch_feature_store.h"
#include "utils/constants.h"
#include "utils/testing.h"

using namespace std;
namespace ar = boost::archive;

namespace oxlm {

class SparseGlobalFeatureStoreTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    store = SparseGlobalFeatureStore(5, 4);
    SparseMinibatchFeatureStore g_store(5);

    VectorReal values(5);

    feature_context_ids1 = {2};
    values = SparseVectorReal(5);
    store.hintFeatureIndex(feature_context_ids1[0], 0);
    store.hintFeatureIndex(feature_context_ids1[0], 1);

    values = SparseVectorReal(5);
    values << 5, 3, 0, 0, 0;
    g_store.hintFeatureIndex(feature_context_ids1[0], 0);
    g_store.hintFeatureIndex(feature_context_ids1[0], 1);
    g_store.update(feature_context_ids1, values);

    feature_context_ids2 = {3};
    store.hintFeatureIndex(feature_context_ids2[0], 2);
    store.hintFeatureIndex(feature_context_ids2[0], 3);

    values = SparseVectorReal(5);
    values << 0, 0, 2, 1, 0;
    g_store.hintFeatureIndex(feature_context_ids2[0], 2);
    g_store.hintFeatureIndex(feature_context_ids2[0], 3);
    g_store.update(feature_context_ids2, values);

    gradient_store = boost::make_shared<SparseMinibatchFeatureStore>(g_store);
  }

  SparseGlobalFeatureStore store;
  boost::shared_ptr<MinibatchFeatureStore> gradient_store;
  vector<int> feature_context_ids1, feature_context_ids2;
};

TEST_F(SparseGlobalFeatureStoreTest, TestUpdateSquared) {
  store.updateSquared(gradient_store);

  EXPECT_EQ(4, store.size());
  VectorReal expected_values(5);
  expected_values << 25, 9, 0, 0, 0;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids1), EPS);
  expected_values << 0, 0, 4, 1, 0;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids2), EPS);
}

TEST_F(SparseGlobalFeatureStoreTest, TestUpdateAdaGrad) {
  store.updateSquared(gradient_store);
  boost::shared_ptr<GlobalFeatureStore> adagrad_store =
      boost::make_shared<SparseGlobalFeatureStore>(store);

  store.updateAdaGrad(gradient_store, adagrad_store, 1);

  EXPECT_EQ(4, store.size());
  VectorReal expected_values(5);
  expected_values << 24, 8, 0, 0, 0;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids1), EPS);
  expected_values << 0, 0, 3, 0, 0;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids2), EPS);

}

TEST_F(SparseGlobalFeatureStoreTest, TestUpdateRegularizer) {
  store.updateSquared(gradient_store);
  store.l2GradientUpdate(gradient_store, 0.5);
  EXPECT_NEAR(180.75, store.l2Objective(gradient_store, 1), EPS);

  VectorReal expected_values(5);
  expected_values << 12.5, 4.5, 0, 0, 0;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids1), EPS);
  expected_values << 0, 0, 2, 0.5, 0;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids2), EPS);
}

TEST_F(SparseGlobalFeatureStoreTest, TestSerialization) {
  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive output_stream(stream);
  output_stream << store;

  SparseGlobalFeatureStore store_copy;
  ar::binary_iarchive input_stream(stream);
  input_stream >> store_copy;

  EXPECT_EQ(store, store_copy);
}

}
