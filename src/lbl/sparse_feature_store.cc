#include "lbl/sparse_feature_store.h"

#include "lbl/feature_generator.h"
#include "utils/constants.h"

namespace oxlm {

SparseFeatureStore::SparseFeatureStore() {}

SparseFeatureStore::SparseFeatureStore(int vector_max_size)
    : vectorMaxSize(vector_max_size) {}

SparseFeatureStore::SparseFeatureStore(
    int vector_max_size,
    const MatchingContexts& matching_contexts)
    : vectorMaxSize(vector_max_size) {
  for (const auto& matching_context: matching_contexts) {
    hintFeatureIndex(matching_context.first, matching_context.second);
  }
}

VectorReal SparseFeatureStore::get(
    const vector<FeatureContext>& feature_contexts) const {
  VectorReal result = VectorReal::Zero(vectorMaxSize);
  for (const FeatureContext& feature_context: feature_contexts) {
    auto it = featureWeights.find(feature_context);
    if (it != featureWeights.end()) {
      result += it->second;
    }
  }
  return result;
}

void SparseFeatureStore::update(
    const vector<FeatureContext>& feature_contexts,
    const VectorReal& values) {
  for (const FeatureContext& feature_context: feature_contexts) {
    update(feature_context, values);
  }
}

Real SparseFeatureStore::updateRegularizer(Real lambda) {
  Real result = 0;
  for (auto& entry: featureWeights) {
    entry.second -= lambda * entry.second;
    result += entry.second.cwiseAbs2().sum();
  }
  return result;
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
    SparseVectorReal squaredValues = entry.second.cwiseAbs2();
    update(entry.first, squaredValues);
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
  for (const auto& entry: adagrad_store->featureWeights) {
    SparseVectorReal& weights = featureWeights.at(entry.first);
    const SparseVectorReal& gradient =
        gradient_store->featureWeights.at(entry.first);
    const SparseVectorReal& adagrad = entry.second
        .cwiseSqrt().unaryExpr(CwiseNumeratorOp<Real>(EPS));
    weights -= step_size * gradient.cwiseProduct(adagrad);
  }
}

void SparseFeatureStore::clear() {
  for (auto& entry: featureWeights) {
    entry.second.setZero();
  }
}

size_t SparseFeatureStore::size() const {
  return featureWeights.size();
}

void SparseFeatureStore::hintFeatureIndex(
    const vector<FeatureContext>& feature_contexts,
    int feature_index) {
  for (const FeatureContext& feature_context: feature_contexts) {
    auto it = featureWeights.find(feature_context);
    if (it != featureWeights.end()) {
      it->second.coeffRef(feature_index) = 0;
    } else {
      SparseVectorReal weights(vectorMaxSize);
      weights.coeffRef(feature_index) = 0;
      featureWeights.insert(make_pair(feature_context, weights));
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
    const FeatureContext& feature_context, const VectorReal& values) {
  SparseVectorReal weights = featureWeights.at(feature_context);
  VectorReal pattern = weights.unaryExpr(CwiseIdentityOp<Real>());
  VectorReal product = (values.array() * pattern.array()).matrix();
  //TODO(paul): Should we normalize by summing only the defined features?
  featureWeights[feature_context] = weights + product.sparseView();
}

void SparseFeatureStore::update(
    const FeatureContext& feature_context, const SparseVectorReal& values) {
  auto it = featureWeights.find(feature_context);
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
