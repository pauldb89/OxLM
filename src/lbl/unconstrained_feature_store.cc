#include "lbl/unconstrained_feature_store.h"

#include "utils/constants.h"

namespace oxlm {

UnconstrainedFeatureStore::UnconstrainedFeatureStore() {}

UnconstrainedFeatureStore::UnconstrainedFeatureStore(int vector_size)
    : vectorSize(vector_size) {}

VectorReal UnconstrainedFeatureStore::get(
    const vector<FeatureContext>& feature_contexts) const {
  VectorReal result = VectorReal::Zero(vectorSize);
  for (const FeatureContext& feature_context: feature_contexts) {
    auto it = featureWeights.find(feature_context);
    if (it != featureWeights.end()) {
      result += it->second;
    }
  }
  return result;
}

void UnconstrainedFeatureStore::update(
    const vector<FeatureContext>& feature_contexts,
    const VectorReal& values) {
  for (const FeatureContext& feature_context: feature_contexts) {
    update(feature_context, values);
  }
}

Real UnconstrainedFeatureStore::updateRegularizer(Real lambda) {
  Real result = 0;
  for (auto& entry: featureWeights) {
    entry.second -= lambda * entry.second;
    result += entry.second.array().square().sum();
  }
  return result;
}

void UnconstrainedFeatureStore::update(
const boost::shared_ptr<FeatureStore>& base_store) {
  boost::shared_ptr<UnconstrainedFeatureStore> store = cast(base_store);
  for (const auto& entry: store->featureWeights) {
    update(entry.first, entry.second);
  }
}

void UnconstrainedFeatureStore::updateSquared(
    const boost::shared_ptr<FeatureStore>& base_store) {
  boost::shared_ptr<UnconstrainedFeatureStore> store = cast(base_store);
  for (const auto& entry: store->featureWeights) {
    update(entry.first, entry.second.array().square());
  }
}

void UnconstrainedFeatureStore::updateAdaGrad(
    const boost::shared_ptr<FeatureStore>& base_gradient_store,
    const boost::shared_ptr<FeatureStore>& base_adagrad_store,
    Real step_size) {
  boost::shared_ptr<UnconstrainedFeatureStore> gradient_store =
      cast(base_gradient_store);
  boost::shared_ptr<UnconstrainedFeatureStore> adagrad_store =
      cast(base_adagrad_store);
  for (const auto& entry: gradient_store->featureWeights) {
    VectorReal weights = VectorReal::Zero(vectorSize);
    const VectorReal& gradient = entry.second;
    const VectorReal& adagrad = adagrad_store->featureWeights.at(entry.first);
    for (int r = 0; r < adagrad.rows(); ++r) {
      if (adagrad(r)) {
        weights(r) = -step_size * gradient(r) / sqrt(adagrad(r));
      }
    }
    update(entry.first, weights);
  }
}

void UnconstrainedFeatureStore::clear() {
  featureWeights.clear();
}

size_t UnconstrainedFeatureStore::size() const {
  return featureWeights.size();
}

bool UnconstrainedFeatureStore::operator==(
    const UnconstrainedFeatureStore& store) const {
  if (vectorSize != store.vectorSize ||
      featureWeights.size() != store.featureWeights.size()) {
    return false;
  }

  for (const auto& entry: featureWeights) {
    auto it = store.featureWeights.find(entry.first);
    if (it == store.featureWeights.end()) {
      return false;
    }

    if ((entry.second - it->second).cwiseAbs().maxCoeff() > EPS) {
      return false;
    }
  }

  return true;
}

void UnconstrainedFeatureStore::update(
    const FeatureContext& feature_context, const VectorReal& values) {
  auto it = featureWeights.find(feature_context);
  if (it != featureWeights.end()) {
    it->second += values;
  } else {
    featureWeights.insert(make_pair(feature_context, values));
  }
}

boost::shared_ptr<UnconstrainedFeatureStore> UnconstrainedFeatureStore::cast(
        const boost::shared_ptr<FeatureStore>& base_store) const {
  boost::shared_ptr<UnconstrainedFeatureStore> store =
      dynamic_pointer_cast<UnconstrainedFeatureStore>(base_store);
  assert(store != nullptr);
  return store;
}

} // namespace oxlm
