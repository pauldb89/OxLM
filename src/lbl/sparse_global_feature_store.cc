#include "lbl/sparse_global_feature_store.h"

#include "lbl/operators.h"
#include "lbl/sparse_minibatch_feature_store.h"
#include "utils/constants.h"

namespace oxlm {

SparseGlobalFeatureStore::SparseGlobalFeatureStore() {}

SparseGlobalFeatureStore::SparseGlobalFeatureStore(
    int vector_max_size, int num_contexts)
    : vectorMaxSize(vector_max_size), featureWeights(num_contexts) {}

SparseGlobalFeatureStore::SparseGlobalFeatureStore(
    int vector_max_size, GlobalFeatureIndexesPtr feature_indexes)
    : vectorMaxSize(vector_max_size) {
  featureWeights = vector<SparseVectorReal>(
      feature_indexes->size(), SparseVectorReal(vectorMaxSize));
  for (size_t i = 0; i < feature_indexes->size(); ++i) {
    for (int feature_index: feature_indexes->at(i)) {
      hintFeatureIndex(i, feature_index);
    }
  }
}

VectorReal SparseGlobalFeatureStore::get(
    const vector<int>& feature_context_ids) const {
  VectorReal result = VectorReal::Zero(vectorMaxSize);
  for (int feature_context_id: feature_context_ids) {
    // We do not extract feature context ids for contexts that have not been
    // observed.
    result += featureWeights[feature_context_id];
  }
  return result;
}

void SparseGlobalFeatureStore::l2GradientUpdate(
    const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store,
    Real sigma) {
  boost::shared_ptr<SparseMinibatchFeatureStore> minibatch_store =
      SparseMinibatchFeatureStore::cast(base_minibatch_store);

  for (const auto& entry: minibatch_store->featureWeights) {
    featureWeights[entry.first] -= sigma * featureWeights[entry.first];
  }
}

Real SparseGlobalFeatureStore::l2Objective(
    const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store,
    Real factor) const {
  boost::shared_ptr<SparseMinibatchFeatureStore> minibatch_store =
      SparseMinibatchFeatureStore::cast(base_minibatch_store);

  Real result = 0;
  for (const auto& entry: minibatch_store->featureWeights) {
    result += featureWeights[entry.first].cwiseAbs2().sum();
  }

  return factor * result;
}

void SparseGlobalFeatureStore::updateSquared(
    const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store) {
  boost::shared_ptr<SparseMinibatchFeatureStore> store =
      SparseMinibatchFeatureStore::cast(base_minibatch_store);
  for (const auto& entry: store->featureWeights) {
    SparseVectorReal values = entry.second.cwiseAbs2();
    update(entry.first, values);
  }
}

void SparseGlobalFeatureStore::updateAdaGrad(
    const boost::shared_ptr<MinibatchFeatureStore>& base_gradient_store,
    const boost::shared_ptr<GlobalFeatureStore>& base_adagrad_store,
    Real step_size) {
  boost::shared_ptr<SparseMinibatchFeatureStore> gradient_store =
      SparseMinibatchFeatureStore::cast(base_gradient_store);
  boost::shared_ptr<SparseGlobalFeatureStore> adagrad_store =
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

size_t SparseGlobalFeatureStore::size() const {
  return featureWeights.size();
}

void SparseGlobalFeatureStore::hintFeatureIndex(
    int feature_context_id, int feature_index) {
  assert(0 <= feature_index &&
         feature_index < featureWeights[feature_context_id].size());
  featureWeights[feature_context_id].coeffRef(feature_index) = 0;
}

bool SparseGlobalFeatureStore::operator==(
    const SparseGlobalFeatureStore& store) const {
  if (vectorMaxSize != store.vectorMaxSize ||
      featureWeights.size() != store.featureWeights.size()) {
    return false;
  }

  for (size_t i = 0; i < featureWeights.size(); ++i) {
    VectorReal diff = featureWeights[i] - store.featureWeights[i];
    if (diff.cwiseAbs2().maxCoeff() > EPS) {
      return false;
    }
  }

  return true;
}

void SparseGlobalFeatureStore::update(
    int feature_context_id, const SparseVectorReal& values) {
  // All features involved in gradient updates must be defined since the
  // construction of the sparse feature store.
  featureWeights[feature_context_id] += values;
}

boost::shared_ptr<SparseGlobalFeatureStore> SparseGlobalFeatureStore::cast(
    const boost::shared_ptr<GlobalFeatureStore>& base_store) {
  boost::shared_ptr<SparseGlobalFeatureStore> store =
      dynamic_pointer_cast<SparseGlobalFeatureStore>(base_store);
  assert(store != nullptr);
  return store;
}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::SparseGlobalFeatureStore)
