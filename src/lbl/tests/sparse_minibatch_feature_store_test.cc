#include "gtest/gtest.h"

#include <sstream>
#include <vector>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/sparse_minibatch_feature_store.h"
#include "utils/constants.h"
#include "utils/testing.h"

using namespace std;
namespace ar = boost::archive;

namespace oxlm {

class SparseMinibatchFeatureStoreTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    store = SparseMinibatchFeatureStore(5);
    SparseMinibatchFeatureStore g_store(5);

    VectorReal values(5);

    feature_context_ids1 = {1};
    store.hintFeatureIndex(feature_context_ids1[0], 1);
    g_store.hintFeatureIndex(feature_context_ids1[0], 1);
    store.hintFeatureIndex(feature_context_ids1[0], 4);
    g_store.hintFeatureIndex(feature_context_ids1[0], 4);
    values << 0, 2, 0, 0, 4;
    store.update(feature_context_ids1, values);

    feature_context_ids2 = {2};
    values = SparseVectorReal(5);
    values << 1, 0, 0, 0, 3;
    store.hintFeatureIndex(feature_context_ids2[0], 0);
    store.hintFeatureIndex(feature_context_ids2[0], 1);
    store.hintFeatureIndex(feature_context_ids2[0], 4);
    store.update(feature_context_ids2, values);

    values = SparseVectorReal(5);
    values << 5, 3, 0, 0, 0;
    g_store.hintFeatureIndex(feature_context_ids2[0], 0);
    g_store.hintFeatureIndex(feature_context_ids2[0], 1);
    g_store.hintFeatureIndex(feature_context_ids2[0], 4);
    g_store.update(feature_context_ids2, values);

    feature_context_ids3 = {3};
    values = SparseVectorReal(5);
    values << 0, 0, 2, 1, 0;
    store.hintFeatureIndex(feature_context_ids3[0], 2);
    g_store.hintFeatureIndex(feature_context_ids3[0], 2);
    store.hintFeatureIndex(feature_context_ids3[0], 3);
    g_store.hintFeatureIndex(feature_context_ids3[0], 3);
    g_store.update(feature_context_ids3, values);

    gradient_store = boost::make_shared<SparseMinibatchFeatureStore>(g_store);
  }

  SparseMinibatchFeatureStore store;
  boost::shared_ptr<MinibatchFeatureStore> gradient_store;
  vector<int> feature_context_ids1, feature_context_ids2, feature_context_ids3;
};

TEST_F(SparseMinibatchFeatureStoreTest, TestBasic) {
  SparseMinibatchFeatureStore feature_store(5);
  EXPECT_MATRIX_NEAR(
      VectorReal::Zero(5), feature_store.get(feature_context_ids1), EPS);

  feature_store.hintFeatureIndex(feature_context_ids1[0], 1);
  feature_store.hintFeatureIndex(feature_context_ids1[0], 3);
  feature_store.hintFeatureIndex(feature_context_ids1[0], 4);
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

TEST_F(SparseMinibatchFeatureStoreTest, TestCombined) {
  vector<int> feature_context_ids = {1, 2};
  VectorReal expected_values(5);
  expected_values << 1, 2, 0, 0, 7;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids), EPS);
}

TEST_F(SparseMinibatchFeatureStoreTest, TestUpdateStore) {
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

TEST_F(SparseMinibatchFeatureStoreTest, TestClear) {
  store.clear();
  EXPECT_EQ(0, store.size());
}

}
