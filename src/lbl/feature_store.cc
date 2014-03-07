#include "feature_store.h"

namespace oxlm {

UnconstrainedFeatureStore::UnconstrainedFeatureStore() {}

UnconstrainedFeatureStore::UnconstrainedFeatureStore(int vector_size) :
    vector_size(vector_size) {}

VectorReal UnconstrainedFeatureStore::get(const vector<Feature>& features) const {
  VectorReal result = VectorReal::Zero(vector_size);
  for (const Feature& feature: features) {
    auto it = feature_weights.find(feature);
    if (it != feature_weights.end()) {
      result += it->second;
    }
  }

  return result;
}

void UnconstrainedFeatureStore::update(
    const vector<Feature>& features,
    const VectorReal& values) {
  for (const Feature& feature: features) {
    update(feature, values);
  }
}

Real UnconstrainedFeatureStore::updateRegularizer(Real lambda) {
  Real result = 0;
  for (auto& entry: feature_weights) {
    entry.second -= lambda * entry.second;
    result += entry.second.array().square().sum();
  }
  return result;
}

void UnconstrainedFeatureStore::update(const UnconstrainedFeatureStore& store) {
  for (const auto& entry: store.feature_weights) {
    update(entry.first, entry.second);
  }
}

void UnconstrainedFeatureStore::updateSquared(
    const UnconstrainedFeatureStore& store) {
  for (const auto& entry: store.feature_weights) {
    update(entry.first, VectorReal(entry.second.array().square()));
  }
}

void UnconstrainedFeatureStore::updateAdaGrad(
    const UnconstrainedFeatureStore& gradient_store,
    const UnconstrainedFeatureStore& adagrad_store,
    Real step_size) {
  for (const auto& entry: gradient_store.feature_weights) {
    VectorReal weights = VectorReal::Zero(vector_size);
    const VectorReal& gradient = entry.second;
    const VectorReal& adagrad = adagrad_store.feature_weights.at(entry.first);
    for (int c = 0; c < adagrad.rows(); ++c) {
      if (adagrad(c)) {
        weights(c) = -step_size * gradient(c) / sqrt(adagrad(c));
      }
    }
    update(entry.first, weights);
  }
}

void UnconstrainedFeatureStore::clear() {
  feature_weights.clear();
}

size_t UnconstrainedFeatureStore::size() const {
  return feature_weights.size();
}

void UnconstrainedFeatureStore::update(
    const Feature& feature,
    const VectorReal& values) {
  auto it = feature_weights.find(feature);
  if (it != feature_weights.end()) {
    it->second += values;
  } else {
    feature_weights.insert(make_pair(feature, values));
  }
}

} // namespace oxlm
