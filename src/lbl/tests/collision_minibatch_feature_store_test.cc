#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/collision_minibatch_feature_store.h"
#include "utils/constants.h"
#include "utils/testing.h"

namespace oxlm {

class CollisionMinibatchFeatureStoreTest : public testing::Test {
 protected:
  void SetUp() {
    context = {1, 2, 3};
    store = boost::make_shared<CollisionMinibatchFeatureStore>(3, 10, 3);

    g_store = boost::make_shared<CollisionMinibatchFeatureStore>(3, 10, 3);
    VectorReal values(3);
    values << 4, 2, 5;
    g_store->update(context, values);
  }

  vector<int> context;
  boost::shared_ptr<CollisionMinibatchFeatureStore> store;
  boost::shared_ptr<MinibatchFeatureStore> g_store;
};

TEST_F(CollisionMinibatchFeatureStoreTest, TestBasic) {
  VectorReal expected_values = VectorReal::Zero(3);
  EXPECT_MATRIX_NEAR(expected_values, store->get(context), EPS);

  VectorReal values(3);
  values << 4, 2, 5;
  store->update(context, values);
  // Due to collisions we don't get 3 x values.
  expected_values << 20, 10, 25;
  EXPECT_MATRIX_NEAR(expected_values, store->get(context), EPS);
  EXPECT_EQ(6, store->size());
}

TEST_F(CollisionMinibatchFeatureStoreTest, TestGradientUpdate) {
  store->update(g_store);

  VectorReal expected_values(3);
  expected_values << 20, 10, 25;
  EXPECT_MATRIX_NEAR(expected_values, store->get(context), EPS);
  EXPECT_EQ(6, store->size());
}

TEST_F(CollisionMinibatchFeatureStoreTest, TestClear) {
  EXPECT_EQ(6, g_store->size());

  g_store->clear();
  EXPECT_EQ(0, g_store->size());
  EXPECT_EQ(VectorReal::Zero(3), g_store->get(context));
}

} // namespace oxlm
