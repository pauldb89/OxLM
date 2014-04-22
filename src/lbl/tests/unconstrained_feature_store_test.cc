#include "gtest/gtest.h"

#include <sstream>
#include <vector>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/unconstrained_feature_store.h"
#include "utils/constants.h"
#include "utils/testing.h"

using namespace std;
namespace ar = boost::archive;

namespace oxlm {

class UnconstrainedFeatureStoreTest : public ::testing::Test {
 protected:

  virtual void SetUp() {
    store = UnconstrainedFeatureStore(3);
    gradient_store = boost::make_shared<UnconstrainedFeatureStore>();

    VectorReal values(3);

    feature_context_ids1 = {1};
    values << 1, 2, 3;
    store.update(feature_context_ids1, values);

    feature_context_ids2 = {2};
    values << 6, 5, 4;
    store.update(feature_context_ids2, values);

    values << 1.5, 1.25, 1;
    gradient_store->update(feature_context_ids2, values);

    feature_context_ids3 = {3};
    values << 0.5, 0.75, 1;
    gradient_store->update(feature_context_ids3, values);
  }

  vector<int> feature_context_ids1, feature_context_ids2, feature_context_ids3;
  UnconstrainedFeatureStore store;
  boost::shared_ptr<UnconstrainedFeatureStore> gradient_store;
};

TEST_F(UnconstrainedFeatureStoreTest, TestBasic) {
  UnconstrainedFeatureStore feature_store(3);

  EXPECT_MATRIX_NEAR(
      VectorReal::Zero(3), feature_store.get(feature_context_ids1), EPS);

  VectorReal values(3);
  values << 1, 2, 3;
  feature_store.update(feature_context_ids1, values);
  EXPECT_MATRIX_NEAR(values, feature_store.get(feature_context_ids1), EPS);

  feature_store.update(feature_context_ids1, values);
  EXPECT_MATRIX_NEAR(2 * values, feature_store.get(feature_context_ids1), EPS);

  EXPECT_EQ(1, feature_store.size());
  feature_store.clear();
  EXPECT_EQ(0, feature_store.size());
}

TEST_F(UnconstrainedFeatureStoreTest, TestCombined) {
  vector<int> feature_context_ids = {1, 2};

  VectorReal expected_values(3);
  expected_values << 7, 7, 7;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids), EPS);
}

TEST_F(UnconstrainedFeatureStoreTest, TestUpdateRegularizer) {
  store.l2GradientUpdate(0.5);
  EXPECT_NEAR(22.75, store.l2Objective(1), EPS);

  VectorReal expected_values(3);
  expected_values << 0.5, 1, 1.5;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids1), EPS);

  expected_values << 3, 2.5, 2;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids2), EPS);
}

TEST_F(UnconstrainedFeatureStoreTest, TestUpdateStore) {
  store.update(gradient_store);

  EXPECT_EQ(3, store.size());
  VectorReal expected_values(3);
  expected_values << 1, 2, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids1), EPS);
  expected_values << 7.5, 6.25, 5;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids2), EPS);
  expected_values << 0.5, 0.75, 1;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids3), EPS);
}

TEST_F(UnconstrainedFeatureStoreTest, TestUpdateSquared) {
  store.updateSquared(gradient_store);

  EXPECT_EQ(3, store.size());
  VectorReal expected_values(3);
  expected_values << 1, 2, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids1), EPS);
  expected_values << 8.25, 6.5625, 5;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids2), EPS);
  expected_values << 0.25, 0.5625, 1;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids3), EPS);
}

TEST_F(UnconstrainedFeatureStoreTest, TestUpdateAdaGrad) {
  store.updateAdaGrad(gradient_store, gradient_store, 1);

  EXPECT_EQ(3, store.size());
  VectorReal expected_values(3);
  expected_values << 1, 2, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids1), EPS);
  expected_values << 4.775255, 3.881966, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids2), EPS);
  expected_values << -0.707106, -0.866025, -1;
  EXPECT_MATRIX_NEAR(expected_values, store.get(feature_context_ids3), EPS);
}

TEST_F(UnconstrainedFeatureStoreTest, TestSerialization) {
  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive output_stream(stream, ar::no_header);
  output_stream << store;

  UnconstrainedFeatureStore store_copy;
  ar::binary_iarchive input_stream(stream, ar::no_header);
  input_stream >> store_copy;

  EXPECT_EQ(store, store_copy);
}

}
