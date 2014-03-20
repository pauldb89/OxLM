#include "gtest/gtest.h"

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

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
  ContextProcessor processor(corpus, config.ngram_order, 0, 1);
  boost::shared_ptr<FeatureContextExtractor> extractor =
      boost::make_shared<FeatureContextExtractor>(corpus, processor, 0);
  FeatureMatcher matcher(corpus, index, processor, extractor);
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
