#include "gtest/gtest.h"

#include <sstream>
#include <vector>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/sparse_feature_store.h"
#include "utils/constants.h"
#include "utils/testing.h"

using namespace std;
namespace ar = boost::archive;

namespace oxlm {

class SparseFeatureStoreTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    store = SparseFeatureStore(5);
    SparseFeatureStore g_store(5);

    VectorReal values(5);

    feature_context_ids1 = {1};
    store.hintFeatureIndex(feature_context_ids1, 1);
    g_store.hintFeatureIndex(feature_context_ids1, 1);
    store.hintFeatureIndex(feature_context_ids1, 4);
    g_store.hintFeatureIndex(feature_context_ids1, 4);
    values << 0, 2, 0, 0, 4;
    store.update(feature_context_ids1, values);

    feature_context_ids2 = {2};
    values = SparseVectorReal(5);
    values << 1, 0, 0, 0, 3;
    store.hintFeatureIndex(feature_context_ids2, 0);
    store.hintFeatureIndex(feature_context_ids2, 1);
    store.hintFeatureIndex(feature_context_ids2, 4);
    store.update(feature_context_ids2, values);

    values = SparseVectorReal(5);
    values << 5, 3, 0, 0, 0;
    g_store.hintFeatureIndex(feature_context_ids2, 0);
    g_store.hintFeatureIndex(feature_context_ids2, 1);
    g_store.hintFeatureIndex(feature_context_ids2, 4);
    g_store.update(feature_context_ids2, values);

    feature_context_ids3 = {3};
    values = SparseVectorReal(5);
    values << 0, 0, 2, 1, 0;
    store.hintFeatureIndex(feature_context_ids3, 2);
    g_store.hintFeatureIndex(feature_context_ids3, 2);
    store.hintFeatureIndex(feature_context_ids3, 3);
    g_store.hintFeatureIndex(feature_context_ids3, 3);
    g_store.update(feature_context_ids3, values);

    gradient_store = boost::make_shared<SparseFeatureStore>(g_store);
  }

  SparseFeatureStore store;
  boost::shared_ptr<FeatureStore> gradient_store;
  vector<FeatureContextId> feature_context_ids1;
  vector<FeatureContextId> feature_context_ids2;
  vector<FeatureContextId> feature_context_ids3;
};

TEST_F(SparseFeatureStoreTest, TestBasic) {
  SparseFeatureStore feature_store(5);
  EXPECT_MATRIX_NEAR(
      VectorReal::Zero(5), feature_store.get(feature_context_ids1), EPS);

  feature_store.hintFeatureIndex(feature_context_ids1, 1);
  feature_store.hintFeatureIndex(feature_context_ids1, 3);
  feature_store.hintFeatureIndex(feature_context_ids1, 4);
  VectorReal values(5), expected_values(5);
  values << 10, 1, 20, 3, 4;
  expected_values << 0, 1, 0, 3, 4;
  feature_store.update(feature_context_ids1, values);
  EXPECT_MATRIX_NEAR(
      expected_values, feature_store.get(feature_context_ids1), EPS);
  feature_store.update(feature_context_ids1, values);
  expected_values *= 2;
  EXPECT_MATRIX_NEAR(
      expected_values, feature_store.get(feature_context_ids1), EPS);
}

TEST_F(SparseFeatureStoreTest, TestCombined) {
  vector<FeatureContextId> feature_context_ids = {1, 2};
  VectorReal expected_values(5);
  expected_values << 1, 2, 0, 0, 7;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids), EPS);
}

TEST_F(SparseFeatureStoreTest, TestUpdateRegularizer) {
  EXPECT_NEAR(7.5, store.updateRegularizer(0.5), EPS);

  VectorReal expected_values(5);
  expected_values << 0, 1, 0, 0, 2;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids1), EPS);
  expected_values << 0.5, 0, 0, 0, 1.5;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids2), EPS);
}

TEST_F(SparseFeatureStoreTest, TestUpdateStore) {
  store.update(gradient_store);

  EXPECT_EQ(3, store.size());
  VectorReal expected_values(5);
  expected_values << 0, 2, 0, 0, 4;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids1), EPS);
  expected_values << 6, 3, 0, 0, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids2), EPS);
  expected_values << 0, 0, 2, 1, 0;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids3), EPS);
}

TEST_F(SparseFeatureStoreTest, TestUpdateSquared) {
  store.updateSquared(gradient_store);

  EXPECT_EQ(3, store.size());
  VectorReal expected_values(5);
  expected_values << 0, 2, 0, 0, 4;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids1), EPS);
  expected_values << 26, 9, 0, 0, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids2), EPS);
  expected_values << 0, 0, 4, 1, 0;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids3), EPS);
}

TEST_F(SparseFeatureStoreTest, TestUpdateAdaGrad) {
  store.updateAdaGrad(gradient_store, gradient_store, 1);

  EXPECT_EQ(3, store.size());
  VectorReal expected_values(5);
  expected_values << 0, 2, 0, 0, 4;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids1), EPS);
  expected_values << -1.236067, -1.732050, 0, 0, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids2), EPS);
  expected_values << 0, 0, -1.41421, -1, 0;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids3), EPS);

}

TEST_F(SparseFeatureStoreTest, TestClear) {
  store.clear();
  EXPECT_EQ(3, store.size());

  EXPECT_MATRIX_NEAR(VectorReal::Zero(5), store.get(feature_context_ids1), EPS);

  VectorReal values(5);
  values << 1, 2, 3, 4, 5;
  store.update(feature_context_ids1, values);
  VectorReal expected_values(5);
  expected_values << 0, 2, 0, 0, 5;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids1), EPS);
}

TEST_F(SparseFeatureStoreTest, TestSerialization) {
  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive output_stream(stream);
  output_stream << store;

  SparseFeatureStore store_copy;
  ar::binary_iarchive input_stream(stream);
  input_stream >> store_copy;

  EXPECT_EQ(store, store_copy);
}

}
