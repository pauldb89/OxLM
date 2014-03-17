#include "lbl/feature_store_initializer.h"

#include <boost/make_shared.hpp>

#include "lbl/sparse_feature_store.h"
#include "lbl/unconstrained_feature_store.h"

namespace oxlm {

FeatureStoreInitializer::FeatureStoreInitializer(
    const ModelData& config,
    const WordToClassIndex& index,
    const FeatureMatcher& matcher)
    : config(config), index(index), matcher(matcher) {}

void FeatureStoreInitializer::initialize(
    boost::shared_ptr<FeatureStore>& U,
    vector<boost::shared_ptr<FeatureStore>>& V,
    bool random_weights) const {
  if (config.sparse_features) {
    U = boost::make_shared<SparseFeatureStore>(
        config.classes, matcher.getClassFeatures(), random_weights);
    V.resize(config.classes);
    for (int i = 0; i < config.classes; ++i) {
      V[i] = boost::make_shared<SparseFeatureStore>(
          index.getClassSize(i), matcher.getWordFeatures(i), random_weights);
    }
  } else {
    U = boost::make_shared<UnconstrainedFeatureStore>(config.classes);
    V.resize(config.classes);
    for (int i = 0; i < config.classes; ++i) {
      V[i] = boost::make_shared<UnconstrainedFeatureStore>(
          index.getClassSize(i));
    }
  }
}

} // namespace oxlm
