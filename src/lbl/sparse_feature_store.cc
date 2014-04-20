#include "lbl/sparse_feature_store.h"

#include <random>

#include "lbl/operators.h"
#include "utils/constants.h"

namespace oxlm {

SparseFeatureStore::SparseFeatureStore() {}

SparseFeatureStore::SparseFeatureStore(int vector_max_size)
    : vectorMaxSize(vector_max_size) {}

SparseFeatureStore::SparseFeatureStore(
    int vector_max_size,
    FeatureIndexesPtr feature_indexes,
    bool random_weights)
    : vectorMaxSize(vector_max_size) {
  random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<Real> gaussian(0, 0.1);
  for (const auto& feature_context_indexes: *feature_indexes) {
    for (int feature_index: feature_context_indexes.second) {
      Real value = random_weights ? gaussian(gen) : 0;
      hintFeatureIndex(feature_context_indexes.first, feature_index, value);
    }
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
    update(feature_context_id, values);
  }
}

void SparseFeatureStore::l2GradientUpdate(Real sigma) {
  for (auto& entry: featureWeights) {
    entry.second -= sigma * entry.second;
  }
}

Real SparseFeatureStore::l2Objective(Real factor) const {
  Real result = 0;
  for (const auto& entry: featureWeights) {
    result += entry.second.cwiseAbs2().sum();
  }
  return factor * result;
}

void SparseFeatureStore::update(
    const boost::shared_ptr<FeatureStore>& base_store) {
  boost::shared_ptr<SparseFeatureStore> store = cast(base_store);
  for (const auto& entry: store->featureWeights) {
    update(entry.first, entry.second);
  }
}

void SparseFeatureStore::updateSquared(
    const boost::shared_ptr<FeatureStore>& base_store) {
  boost::shared_ptr<SparseFeatureStore> store = cast(base_store);
  for (const auto& entry: store->featureWeights) {
    SparseVectorReal values = entry.second.cwiseAbs2();
    update(entry.first, values);
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
  for (const auto& entry: gradient_store->featureWeights) {
    SparseVectorReal& weights = featureWeights.at(entry.first);
    const SparseVectorReal& gradient = entry.second;
    const SparseVectorReal& adagrad =
        adagrad_store->featureWeights.at(entry.first);
    weights -= gradient.binaryExpr(
        adagrad, CwiseAdagradUpdateOp<Real>(step_size));
  }
}

void SparseFeatureStore::clear() {
  featureWeights.clear();
}

size_t SparseFeatureStore::size() const {
  return featureWeights.size();
}

void SparseFeatureStore::hintFeatureIndex(
    FeatureContextId feature_context_id, int feature_index, Real value) {
  auto it = featureWeights.find(feature_context_id);
  if (it != featureWeights.end()) {
    it->second.coeffRef(feature_index) = value;
  } else {
    SparseVectorReal weights(vectorMaxSize);
    weights.coeffRef(feature_index) = value;
    featureWeights.insert(make_pair(feature_context_id, weights));
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

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::SparseFeatureStore)
