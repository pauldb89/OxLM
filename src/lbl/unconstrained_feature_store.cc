#include "lbl/unconstrained_feature_store.h"

#include "lbl/operators.h"
#include "utils/constants.h"

namespace oxlm {

UnconstrainedFeatureStore::UnconstrainedFeatureStore() {}

UnconstrainedFeatureStore::UnconstrainedFeatureStore(
    int vector_size,
    const boost::shared_ptr<FeatureContextExtractor>& extractor)
    : vectorSize(vector_size), extractor(extractor) {}

VectorReal UnconstrainedFeatureStore::get(const vector<int>& context) const {
  VectorReal result = VectorReal::Zero(vectorSize);
  for (int feature_context_id: extractor->getFeatureContextIds(context)) {
    auto it = featureWeights.find(feature_context_id);
    if (it != featureWeights.end()) {
      result += it->second;
    }
  }
  return result;
}

void UnconstrainedFeatureStore::update(
    const vector<int>& context, const VectorReal& values) {
  for (int feature_context_id: extractor->getFeatureContextIds(context)) {
    update(feature_context_id, values);
  }
}

void UnconstrainedFeatureStore::l2GradientUpdate(
    const boost::shared_ptr<MinibatchFeatureStore>&, Real sigma) {
  for (auto& entry: featureWeights) {
    entry.second -= sigma * entry.second;
  }
}

Real UnconstrainedFeatureStore::l2Objective(
    const boost::shared_ptr<MinibatchFeatureStore>&, Real factor) const {
  Real result = 0;
  for (const auto& entry: featureWeights) {
    result += entry.second.array().square().sum();
  }
  return factor * result;
}

void UnconstrainedFeatureStore::update(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<UnconstrainedFeatureStore> store = cast(base_store);
  for (const auto& entry: store->featureWeights) {
    update(entry.first, entry.second);
  }
}

void UnconstrainedFeatureStore::updateSquared(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<UnconstrainedFeatureStore> store = cast(base_store);
  for (const auto& entry: store->featureWeights) {
    update(entry.first, entry.second.array().square());
  }
}

void UnconstrainedFeatureStore::updateAdaGrad(
    const boost::shared_ptr<MinibatchFeatureStore>& base_gradient_store,
    const boost::shared_ptr<GlobalFeatureStore>& base_adagrad_store,
    Real step_size) {
  boost::shared_ptr<UnconstrainedFeatureStore> gradient_store =
      cast(base_gradient_store);
  boost::shared_ptr<UnconstrainedFeatureStore> adagrad_store =
      cast(base_adagrad_store);
  for (const auto& entry: gradient_store->featureWeights) {
    const VectorReal& gradient = entry.second;
    const VectorReal& adagrad = adagrad_store->featureWeights.at(entry.first);
    const VectorReal weights = -gradient.binaryExpr(
        adagrad, CwiseAdagradUpdateOp<Real>(step_size));
    update(entry.first, weights);
  }
}

void UnconstrainedFeatureStore::clear() {
  featureWeights.clear();
}

size_t UnconstrainedFeatureStore::size() const {
  return featureWeights.size();
}

boost::shared_ptr<UnconstrainedFeatureStore> UnconstrainedFeatureStore::cast(
        const boost::shared_ptr<FeatureStore>& base_store) {
  boost::shared_ptr<UnconstrainedFeatureStore> store =
      dynamic_pointer_cast<UnconstrainedFeatureStore>(base_store);
  assert(store != nullptr);
  return store;
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

bool UnconstrainedFeatureStore::operator==(
    const boost::shared_ptr<GlobalFeatureStore>& other) const {
  return operator==(*cast(other));
}

vector<pair<int, int>> UnconstrainedFeatureStore::getFeatureIndexes() const {
  vector<pair<int, int>> feature_indexes;
  for (const auto& entry: featureWeights) {
    for (int i = 0; i < vectorSize; ++i) {
      feature_indexes.push_back(make_pair(entry.first, i));
    }
  }

  return feature_indexes;
}

void UnconstrainedFeatureStore::updateFeature(
    const pair<int, int>& index, Real value) {
  featureWeights[index.first](index.second) += value;
}

Real UnconstrainedFeatureStore::getFeature(const pair<int, int>& index) const {
  auto it = featureWeights.find(index.first);
  return it == featureWeights.end() ? 0 : it->second(index.second);
}

UnconstrainedFeatureStore::~UnconstrainedFeatureStore() {}

void UnconstrainedFeatureStore::update(
    int feature_context_id, const VectorReal& values) {
  auto it = featureWeights.find(feature_context_id);
  if (it != featureWeights.end()) {
    it->second += values;
  } else {
    featureWeights.insert(make_pair(feature_context_id, values));
  }
}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::UnconstrainedFeatureStore)
