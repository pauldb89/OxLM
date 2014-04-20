#include "lbl/feature_store_initializer.h"

#include <boost/make_shared.hpp>

#include "lbl/sparse_feature_store.h"
#include "lbl/unconstrained_feature_store.h"

namespace oxlm {

FeatureStoreInitializer::FeatureStoreInitializer(
    const ModelData& config,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<FeatureMatcher>& matcher)
    : config(config), index(index), matcher(matcher) {}

void FeatureStoreInitializer::initialize(
    boost::shared_ptr<FeatureStore>& U,
    vector<boost::shared_ptr<FeatureStore>>& V) const {
  if (config.sparse_features) {
    initializeSparseStores(U, V, matcher->getFeatures());
  } else {
    initializeUnconstrainedStores(U, V);
  }
}

void FeatureStoreInitializer::initialize(
    boost::shared_ptr<FeatureStore>& U,
    vector<boost::shared_ptr<FeatureStore>>& V,
    const vector<int>& minibatch_indices) const {
  if (config.sparse_features) {
    initializeSparseStores(
        U, V, matcher->getFeatures(minibatch_indices));
  } else {
    initializeUnconstrainedStores(U, V);
  }
}

void FeatureStoreInitializer::initializeUnconstrainedStores(
    boost::shared_ptr<FeatureStore>& U,
    vector<boost::shared_ptr<FeatureStore>>& V) const {
  U = boost::make_shared<UnconstrainedFeatureStore>(config.classes);
  V.resize(config.classes);
  for (int i = 0; i < config.classes; ++i) {
    V[i] = boost::make_shared<UnconstrainedFeatureStore>(
        index->getClassSize(i));
  }
}

void FeatureStoreInitializer::initializeSparseStores(
    boost::shared_ptr<FeatureStore>& U,
    vector<boost::shared_ptr<FeatureStore>>& V,
    FeatureIndexesPairPtr feature_indexes_pair) const {
  U = boost::make_shared<SparseFeatureStore>(
      config.classes, feature_indexes_pair->getClassIndexes());
  V.resize(config.classes);
  for (int i = 0; i < config.classes; ++i) {
    V[i] = boost::make_shared<SparseFeatureStore>(
        index->getClassSize(i),
        feature_indexes_pair->getWordIndexes(i));
  }
}

} // namespace oxlm
