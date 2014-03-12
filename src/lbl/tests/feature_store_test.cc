#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "lbl/feature_store.h"
#include "utils/constants.h"
#include "utils/testing.h"

namespace ar = boost::archive;

namespace oxlm {

class UnconstrainedFeatureStoreTest : public ::testing::Test {
 protected:
  UnconstrainedFeatureStoreTest() : store(3) {}

  virtual void SetUp() {
    VectorReal values(3);

    vector<int> feature_data = {1};
    features1 = {Feature(0, feature_data)};
    values << 1, 2, 3;
    store.update(features1, values);

    feature_data = {2};
    features2 = {Feature(0, feature_data)};
    values << 6, 5, 4;
    store.update(features2, values);

    values << 1.5, 1.25, 1;
    gradient_store.update(features2, values);

    feature_data = {3};
    features3 = {Feature(0, feature_data)};
    values << 0.5, 0.75, 1;
    gradient_store.update(features3, values);
  }

  vector<Feature> features1, features2, features3;
  UnconstrainedFeatureStore store, gradient_store;
};

TEST_F(UnconstrainedFeatureStoreTest, BasicTest) {
  UnconstrainedFeatureStore feature_store(3);

  EXPECT_MATRIX_NEAR(VectorReal::Zero(3), feature_store.get(features1), EPS);

  VectorReal values(3);
  values << 1, 2, 3;
  feature_store.update(features1, values);
  EXPECT_MATRIX_NEAR(values, feature_store.get(features1), EPS);

  feature_store.update(features1, values);
  EXPECT_MATRIX_NEAR(2 * values, feature_store.get(features1), EPS);

  EXPECT_EQ(1, feature_store.size());
  feature_store.clear();
  EXPECT_EQ(0, feature_store.size());
}

TEST_F(UnconstrainedFeatureStoreTest, UpdateRegularizerTest) {
  EXPECT_EQ(22.75, store.updateRegularizer(0.5));

  VectorReal expected_values(3);
  expected_values << 0.5, 1, 1.5;
  store.get(features1);
  EXPECT_MATRIX_NEAR(expected_values, store.get(features1), EPS);

  expected_values << 3, 2.5, 2;
  EXPECT_MATRIX_NEAR(expected_values, store.get(features2), EPS);
}

TEST_F(UnconstrainedFeatureStoreTest, UpdateStoreTest) {
  store.update(gradient_store);

  EXPECT_EQ(3, store.size());
  VectorReal expected_values(3);
  expected_values << 1, 2, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(features1), EPS);
  expected_values << 7.5, 6.25, 5;
  EXPECT_MATRIX_NEAR(expected_values, store.get(features2), EPS);
  expected_values << 0.5, 0.75, 1;
  EXPECT_MATRIX_NEAR(expected_values, store.get(features3), EPS);
}

TEST_F(UnconstrainedFeatureStoreTest, UpdateSquaredTest) {
  store.updateSquared(gradient_store);

  EXPECT_EQ(3, store.size());
  VectorReal expected_values(3);
  expected_values << 1, 2, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(features1), EPS);
  expected_values << 8.25, 6.5625, 5;
  EXPECT_MATRIX_NEAR(expected_values, store.get(features2), EPS);
  expected_values << 0.25, 0.5625, 1;
  EXPECT_MATRIX_NEAR(expected_values, store.get(features3), EPS);
}

TEST_F(UnconstrainedFeatureStoreTest, UpdateAdaGradTest) {
  store.updateAdaGrad(gradient_store, gradient_store, 1);

  EXPECT_EQ(3, store.size());
  VectorReal expected_values(3);
  expected_values << 1, 2, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(features1), EPS);
  expected_values << 4.775255, 3.881966, 3;
  EXPECT_MATRIX_NEAR(expected_values, store.get(features2), EPS);
  expected_values << -0.707106, -0.866025, -1;
  EXPECT_MATRIX_NEAR(expected_values, store.get(features3), EPS);
}

TEST_F(UnconstrainedFeatureStoreTest, SerializationTest) {
  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive output_stream(stream, ar::no_header);
  output_stream << store;

  UnconstrainedFeatureStore store_copy;
  ar::binary_iarchive input_stream(stream, ar::no_header);
  input_stream >> store_copy;

  EXPECT_EQ(store, store_copy);
}

}
