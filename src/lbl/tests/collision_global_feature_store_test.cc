#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "lbl/class_context_hasher.h"
#include "lbl/collision_global_feature_store.h"
#include "lbl/feature_no_op_filter.h"
#include "utils/constants.h"
#include "utils/testing.h"

namespace ar = boost::archive;

namespace oxlm {

class CollisionGlobalFeatureStoreTest : public testing::Test {
 protected:
  void SetUp() {
    int vector_size = 3;
    int hash_space = 10;
    boost::shared_ptr<GlobalCollisionSpace> space =
        boost::make_shared<GlobalCollisionSpace>(hash_space);
    boost::shared_ptr<FeatureNoOpFilter> filter =
        boost::make_shared<FeatureNoOpFilter>(vector_size);
    boost::shared_ptr<ClassContextHasher> hasher =
        boost::make_shared<ClassContextHasher>(hash_space);

    CollisionMinibatchFeatureStore g_store(
        vector_size, hash_space, 3, hasher, filter);

    context = {1, 2, 3};
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
  expected_values << 216, 262, 281;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context), EPS);
}

TEST_F(CollisionGlobalFeatureStoreTest, TestUpdateAdaGrad) {
  store.updateSquared(gradient_store);
  boost::shared_ptr<GlobalFeatureStore> adagrad_store =
      boost::make_shared<CollisionGlobalFeatureStore>(store);

  store.updateAdaGrad(gradient_store, adagrad_store, 1);
  VectorReal expected_values(3);
  expected_values << 213, 259, 278;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context), EPS);
}

TEST_F(CollisionGlobalFeatureStoreTest, TestUpdateRegularizer) {
  store.updateSquared(gradient_store);
  store.l2GradientUpdate(gradient_store, 0.5);

  EXPECT_NEAR(6704.25, store.l2Objective(gradient_store, 1), EPS);
  VectorReal expected_values(3);
  expected_values << 108, 131, 140.5;
  EXPECT_MATRIX_NEAR(expected_values, store.get(context), EPS);
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
  expected_values << 216, 262, 281;
  EXPECT_MATRIX_NEAR(expected_values, actual_ptr->get(context), EPS);
}

} // namespace oxlm
