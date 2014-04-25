#include "gtest/gtest.h"

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/feature_store_initializer.h"
#include "lbl/sparse_global_feature_store.h"
#include "lbl/sparse_minibatch_feature_store.h"
#include "lbl/unconstrained_feature_store.h"

namespace oxlm {

class FeatureStoreInitializerTest : public testing::Test {
 protected:
  void SetUp() {
    vector<int> classes = {0, 2};
    config.classes = 1;

    corpus = boost::make_shared<Corpus>();
    index = boost::make_shared<WordToClassIndex>(classes);
  }

  ModelData config;
  boost::shared_ptr<Corpus> corpus;
  boost::shared_ptr<WordToClassIndex> index;
};

TEST_F(FeatureStoreInitializerTest, TestUnconstrainedGlobal) {
  config.sparse_features = false;
  FeatureStoreInitializer initializer(config, corpus, index);

  boost::shared_ptr<GlobalFeatureStore> U;
  vector<boost::shared_ptr<GlobalFeatureStore>> V;
  initializer.initialize(U, V);
  EXPECT_TRUE(typeid(*U.get()) == typeid(UnconstrainedFeatureStore));
  EXPECT_EQ(1, V.size());
  EXPECT_TRUE(typeid(*V[0].get()) == typeid(UnconstrainedFeatureStore));
}

TEST_F(FeatureStoreInitializerTest, TestUnconstrainedMinibatch) {
  config.sparse_features = false;
  FeatureStoreInitializer initializer(config, corpus, index);

  boost::shared_ptr<MinibatchFeatureStore> U;
  vector<boost::shared_ptr<MinibatchFeatureStore>> V;
  vector<int> minibatch_indices;
  initializer.initialize(U, V, minibatch_indices);
  EXPECT_TRUE(typeid(*U.get()) == typeid(UnconstrainedFeatureStore));
  EXPECT_EQ(1, V.size());
  EXPECT_TRUE(typeid(*V[0].get()) == typeid(UnconstrainedFeatureStore));
}

TEST_F(FeatureStoreInitializerTest, TestSparseGlobal) {
  config.sparse_features = true;
  FeatureStoreInitializer initializer(config, corpus, index);

  boost::shared_ptr<GlobalFeatureStore> U;
  vector<boost::shared_ptr<GlobalFeatureStore>> V;
  initializer.initialize(U, V);
  EXPECT_TRUE(typeid(*U.get()) == typeid(SparseGlobalFeatureStore));
  EXPECT_EQ(1, V.size());
  EXPECT_TRUE(typeid(*V[0].get()) == typeid(SparseGlobalFeatureStore));
}

TEST_F(FeatureStoreInitializerTest, TestSparseMinibatch) {
  config.sparse_features = true;
  FeatureStoreInitializer initializer(config, corpus, index);

  boost::shared_ptr<MinibatchFeatureStore> U;
  vector<boost::shared_ptr<MinibatchFeatureStore>> V;
  vector<int> minibatch_indices;
  initializer.initialize(U, V, minibatch_indices);
  EXPECT_TRUE(typeid(*U.get()) == typeid(SparseMinibatchFeatureStore));
  EXPECT_EQ(1, V.size());
  EXPECT_TRUE(typeid(*V[0].get()) == typeid(SparseMinibatchFeatureStore));
}

} // namespace oxlm
