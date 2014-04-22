#include "lbl/feature_store_initializer.h"

#include <boost/make_shared.hpp>

#include "lbl/sparse_global_feature_store.h"
#include "lbl/sparse_minibatch_feature_store.h"
#include "lbl/unconstrained_feature_store.h"

namespace oxlm {

FeatureStoreInitializer::FeatureStoreInitializer(
    const ModelData& config,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<FeatureMatcher>& matcher)
    : config(config), index(index), matcher(matcher) {}

void FeatureStoreInitializer::initialize(
    boost::shared_ptr<GlobalFeatureStore>& U,
    vector<boost::shared_ptr<GlobalFeatureStore>>& V) const {
  if (config.sparse_features) {
    auto feature_indexes_pair = matcher->getGlobalFeatures();
    U = boost::make_shared<SparseGlobalFeatureStore>(config.classes, feature_indexes_pair->getClassIndexes());
    V.resize(config.classes);
    for (int i = 0; i < config.classes; ++i) {
      V[i] = boost::make_shared<SparseGlobalFeatureStore>(
          index->getClassSize(i),
          feature_indexes_pair->getWordIndexes(i));
    }
  } else {
    U = boost::make_shared<UnconstrainedFeatureStore>(config.classes);
    V.resize(config.classes);
    for (int i = 0; i < config.classes; ++i) {
      V[i] = boost::make_shared<UnconstrainedFeatureStore>(
          index->getClassSize(i));
    }
  }
}

void FeatureStoreInitializer::initialize(
    boost::shared_ptr<MinibatchFeatureStore>& U,
    vector<boost::shared_ptr<MinibatchFeatureStore>>& V,
    const vector<int>& minibatch_indices) const {
  if (config.sparse_features) {
    auto feature_indexes_pair = matcher->getMinibatchFeatures(minibatch_indices);
    U = boost::make_shared<SparseMinibatchFeatureStore>(
        config.classes, feature_indexes_pair->getClassIndexes());
    V.resize(config.classes);
    for (int i = 0; i < config.classes; ++i) {
      V[i] = boost::make_shared<SparseMinibatchFeatureStore>(
          index->getClassSize(i),
          feature_indexes_pair->getWordIndexes(i));
    }
  } else {
    U = boost::make_shared<UnconstrainedFeatureStore>(config.classes);
    V.resize(config.classes);
    for (int i = 0; i < config.classes; ++i) {
      V[i] = boost::make_shared<UnconstrainedFeatureStore>(
          index->getClassSize(i));
    }
  }
}

} // namespace oxlm
