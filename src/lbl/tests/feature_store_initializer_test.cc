#include "gtest/gtest.h"

#include "lbl/feature_store_initializer.h"
#include "lbl/sparse_feature_store.h"
#include "lbl/unconstrained_feature_store.h"

namespace oxlm {

TEST(FeatureStoreInitializerTest, TestBasic) {
  vector<int> corpus;
  vector<int> classes = {0, 2};
  ModelData config;
  config.classes = 1;
  WordToClassIndex index(classes);
  ContextExtractor extractor(corpus, config.ngram_order, 0, 1);
  FeatureGenerator generator(config.feature_context_size);
  FeatureMatcher matcher(corpus, index, extractor, generator);
  FeatureStoreInitializer initializer(config, index, matcher);

  boost::shared_ptr<FeatureStore> U;
  vector<boost::shared_ptr<FeatureStore>> V;

  config.sparse_features = false;
  initializer.initialize(U, V);
  EXPECT_TRUE(typeid(*U.get()) == typeid(UnconstrainedFeatureStore));
  EXPECT_EQ(1, V.size());
  EXPECT_TRUE(typeid(*V[0].get()) == typeid(UnconstrainedFeatureStore));

  config.sparse_features = true;
  initializer.initialize(U, V);
  EXPECT_TRUE(typeid(*U.get()) == typeid(SparseFeatureStore));
  EXPECT_EQ(1, V.size());
  EXPECT_TRUE(typeid(*V[0].get()) == typeid(SparseFeatureStore));
}

} // namespace oxlm
