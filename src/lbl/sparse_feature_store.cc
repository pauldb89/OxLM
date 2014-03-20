#include "lbl/sparse_feature_store.h"

#include <random>

#include "utils/constants.h"

namespace oxlm {

SparseFeatureStore::SparseFeatureStore() {}

SparseFeatureStore::SparseFeatureStore(int vector_max_size)
    : vectorMaxSize(vector_max_size) {}

SparseFeatureStore::SparseFeatureStore(
    int vector_max_size,
    const MatchingContexts& matching_contexts,
    bool random_weights)
    : vectorMaxSize(vector_max_size) {
  random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<Real> gaussian(0, 0.1);
  for (const auto& matching_context: matching_contexts) {
    Real value = random_weights ? gaussian(gen) : 0;
    hintFeatureIndex(matching_context.first, matching_context.second, value);
  }
}

VectorReal SparseFeatureStore::get(
    const vector<FeatureContextId>& feature_context_ids) const {
  VectorReal result = VectorReal::Zero(vectorMaxSize);
  for (const FeatureContextId& feature_context_id: feature_context_ids) {
    auto it = featureWeights.find(feature_context_id);
    if (it != featureWeights.end()) {
      result += it->second;
    }
  }
  return result;
}

void SparseFeatureStore::update(
    const vector<FeatureContextId>& feature_context_ids,
    const VectorReal& values) {
  for (const FeatureContextId& feature_context_id: feature_context_ids) {
    observedContexts.insert(feature_context_id);
    update(feature_context_id, values);
  }
}

void SparseFeatureStore::l2GradientUpdate(Real sigma) {
  for (const FeatureContextId& feature_context_id: observedContexts) {
    SparseVectorReal& weights = featureWeights.at(feature_context_id);
    weights -= sigma * weights;
  }
}

Real SparseFeatureStore::l2Objective(Real factor) const {
  Real result = 0;
  for (const FeatureContextId& feature_context_id: observedContexts) {
    result += featureWeights.at(feature_context_id).cwiseAbs2().sum();
  }
  return factor * result;
}

void SparseFeatureStore::update(
    const boost::shared_ptr<FeatureStore>& base_store) {
  boost::shared_ptr<SparseFeatureStore> store = cast(base_store);
  for (const FeatureContextId& feature_context_id: store->observedContexts) {
    observedContexts.insert(feature_context_id);
    update(feature_context_id, store->featureWeights.at(feature_context_id));
  }
}

void SparseFeatureStore::updateSquared(
    const boost::shared_ptr<FeatureStore>& base_store) {
  boost::shared_ptr<SparseFeatureStore> store = cast(base_store);
  for (const FeatureContextId& feature_context_id: store->observedContexts) {
    observedContexts.insert(feature_context_id);
    SparseVectorReal values = store->featureWeights.at(feature_context_id);
    values = values.cwiseAbs2();
    update(feature_context_id, values);
  }
}

void SparseFeatureStore::updateAdaGrad(
    const boost::shared_ptr<FeatureStore>& base_gradient_store,
    const boost::shared_ptr<FeatureStore>& base_adagrad_store,
    Real step_size) {
  boost::shared_ptr<SparseFeatureStore> gradient_store =
      cast(base_gradient_store);
  boost::shared_ptr<SparseFeatureStore> adagrad_store =
      cast(base_adagrad_store);
  for (const FeatureContextId& feature_context_id: gradient_store->observedContexts) {
    observedContexts.insert(feature_context_id);
    SparseVectorReal& weights = featureWeights.at(feature_context_id);
    const SparseVectorReal& gradient =
        gradient_store->featureWeights.at(feature_context_id);
    const SparseVectorReal& adagrad =
        adagrad_store->featureWeights.at(feature_context_id);
    const SparseVectorReal& denominator =
        adagrad.cwiseSqrt().unaryExpr(CwiseDenominatorOp<Real>(EPS));
    weights -= step_size * gradient.cwiseProduct(denominator);
  }
}

void SparseFeatureStore::clear() {
  for (const FeatureContextId& feature_context_id: observedContexts) {
    SparseVectorReal& weights = featureWeights.at(feature_context_id);
    weights = weights.unaryExpr(CwiseSetValueOp<Real>(0));
  }
  observedContexts.clear();
}

size_t SparseFeatureStore::size() const {
  return featureWeights.size();
}

void SparseFeatureStore::hintFeatureIndex(
    const vector<FeatureContextId>& feature_context_ids,
    int feature_index, Real value) {
  for (const FeatureContextId& feature_context_id: feature_context_ids) {
    auto it = featureWeights.find(feature_context_id);
    if (it != featureWeights.end()) {
      it->second.coeffRef(feature_index) = value;
    } else {
      SparseVectorReal weights(vectorMaxSize);
      weights.coeffRef(feature_index) = value;
      featureWeights.insert(make_pair(feature_context_id, weights));
    }
  }
}

bool SparseFeatureStore::operator==(const SparseFeatureStore& store) const {
  if (vectorMaxSize != store.vectorMaxSize ||
      featureWeights.size() != store.featureWeights.size()) {
    return false;
  }

  for (const auto& entry: featureWeights) {
    auto it = store.featureWeights.find(entry.first);
    if (it == store.featureWeights.end()) {
      return false;
    }

    VectorReal diff = entry.second - it->second;
    if (diff.cwiseAbs2().maxCoeff() > EPS) {
      return false;
    }
  }

  return true;
}

void SparseFeatureStore::update(
    const FeatureContextId& feature_context_id, const VectorReal& values) {
  SparseVectorReal& weights = featureWeights.at(feature_context_id);
  VectorReal pattern = weights.unaryExpr(CwiseSetValueOp<Real>(1));
  VectorReal product = (values.array() * pattern.array()).matrix();
  weights += product.sparseView();
}

void SparseFeatureStore::update(
    const FeatureContextId& feature_context_id,
    const SparseVectorReal& values) {
  auto it = featureWeights.find(feature_context_id);
  // All features involved in gradient updates must be defined since the
  // construction of the sparse feature store.
  assert(it != featureWeights.end());
  it->second += values;
}

boost::shared_ptr<SparseFeatureStore> SparseFeatureStore::cast(
    const boost::shared_ptr<FeatureStore>& base_store) const {
  boost::shared_ptr<SparseFeatureStore> store =
      dynamic_pointer_cast<SparseFeatureStore>(base_store);
  assert(store != nullptr);
  return store;
}

} // namespace oxlm
