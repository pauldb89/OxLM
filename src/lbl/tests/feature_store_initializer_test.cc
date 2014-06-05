#include "gtest/gtest.h"

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/collision_global_feature_store.h"
#include "lbl/collision_minibatch_feature_store.h"
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
    boost::shared_ptr<ContextProcessor> processor =
        boost::make_shared<ContextProcessor>(corpus, 2);
    boost::shared_ptr<FeatureContextGenerator> generator =
        boost::make_shared<FeatureContextGenerator>(2);
    boost::shared_ptr<NGramFilter> filter =
        boost::make_shared<NGramFilter>(corpus, index, processor, generator);
    hasher = boost::make_shared<FeatureContextHasher>(
        corpus, index, processor, generator, filter);
    matcher = boost::make_shared<FeatureMatcher>(
        corpus, index, processor, generator, filter, hasher);
  }

  ModelData config;
  boost::shared_ptr<Corpus> corpus;
  boost::shared_ptr<WordToClassIndex> index;
  boost::shared_ptr<FeatureContextHasher> hasher;
  boost::shared_ptr<FeatureMatcher> matcher;
  boost::shared_ptr<BloomFilterPopulator> populator;
};

TEST_F(FeatureStoreInitializerTest, TestUnconstrainedGlobal) {
  config.sparse_features = false;
  FeatureStoreInitializer initializer(
      config, corpus, index, hasher, matcher, populator);

  boost::shared_ptr<GlobalFeatureStore> U;
  vector<boost::shared_ptr<GlobalFeatureStore>> V;
  initializer.initialize(U, V);
  EXPECT_TRUE(typeid(*U.get()) == typeid(UnconstrainedFeatureStore));
  EXPECT_EQ(1, V.size());
  EXPECT_TRUE(typeid(*V[0].get()) == typeid(UnconstrainedFeatureStore));
}

TEST_F(FeatureStoreInitializerTest, TestUnconstrainedMinibatch) {
  config.sparse_features = false;
  FeatureStoreInitializer initializer(
      config, corpus, index, hasher, matcher, populator);

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
  FeatureStoreInitializer initializer(
      config, corpus, index, hasher, matcher, populator);

  boost::shared_ptr<GlobalFeatureStore> U;
  vector<boost::shared_ptr<GlobalFeatureStore>> V;
  initializer.initialize(U, V);
  EXPECT_TRUE(typeid(*U.get()) == typeid(SparseGlobalFeatureStore));
  EXPECT_EQ(1, V.size());
  EXPECT_TRUE(typeid(*V[0].get()) == typeid(SparseGlobalFeatureStore));
}

TEST_F(FeatureStoreInitializerTest, TestSparseMinibatch) {
  config.sparse_features = true;
  FeatureStoreInitializer initializer(
      config, corpus, index, hasher, matcher, populator);

  boost::shared_ptr<MinibatchFeatureStore> U;
  vector<boost::shared_ptr<MinibatchFeatureStore>> V;
  vector<int> minibatch_indices;
  initializer.initialize(U, V, minibatch_indices);
  EXPECT_TRUE(typeid(*U.get()) == typeid(SparseMinibatchFeatureStore));
  EXPECT_EQ(1, V.size());
  EXPECT_TRUE(typeid(*V[0].get()) == typeid(SparseMinibatchFeatureStore));
}

TEST_F(FeatureStoreInitializerTest, TestCollisionGlobal) {
  config.hash_space = 1000000;
  FeatureStoreInitializer initializer(
      config, corpus, index, hasher, matcher, populator);

  boost::shared_ptr<GlobalFeatureStore> U;
  vector<boost::shared_ptr<GlobalFeatureStore>> V;
  initializer.initialize(U, V);
  EXPECT_TRUE(typeid(*U.get()) == typeid(CollisionGlobalFeatureStore));
  EXPECT_EQ(1, V.size());
  EXPECT_TRUE(typeid(*V[0].get()) == typeid(CollisionGlobalFeatureStore));
}

TEST_F(FeatureStoreInitializerTest, TestCollisionMinibatch) {
  config.hash_space = 1000000;
  FeatureStoreInitializer initializer(
      config, corpus, index, hasher, matcher, populator);

  boost::shared_ptr<MinibatchFeatureStore> U;
  vector<boost::shared_ptr<MinibatchFeatureStore>> V;
  vector<int> minibatch_indices;
  initializer.initialize(U, V, minibatch_indices);
  EXPECT_TRUE(typeid(*U.get()) == typeid(CollisionMinibatchFeatureStore));
  EXPECT_EQ(1, V.size());
  EXPECT_TRUE(typeid(*V[0].get()) == typeid(CollisionMinibatchFeatureStore));
}

} // namespace oxlm
