#include "lbl/sparse_minibatch_feature_store.h"

#include "lbl/operators.h"
#include "utils/constants.h"

namespace oxlm {

SparseMinibatchFeatureStore::SparseMinibatchFeatureStore() {}

SparseMinibatchFeatureStore::SparseMinibatchFeatureStore(
    int vector_max_size,
    const boost::shared_ptr<FeatureContextExtractor>& extractor)
    : vectorMaxSize(vector_max_size), extractor(extractor) {}

SparseMinibatchFeatureStore::SparseMinibatchFeatureStore(
    int vector_max_size,
    MinibatchFeatureIndexesPtr feature_indexes,
    const boost::shared_ptr<FeatureContextExtractor>& extractor)
    : vectorMaxSize(vector_max_size), extractor(extractor) {
  for (const auto& feature_context_indexes: *feature_indexes) {
    for (int feature_index: feature_context_indexes.second) {
      hintFeatureIndex(feature_context_indexes.first, feature_index);
    }
  }
}

VectorReal SparseMinibatchFeatureStore::get(const vector<int>& context) const {
  VectorReal result = VectorReal::Zero(vectorMaxSize);
  for (int feature_context_id: extractor->getFeatureContextIds(context)) {
    auto it = featureWeights.find(feature_context_id);
    if (it != featureWeights.end()) {
      result += it->second;
    }
  }
  return result;
}

void SparseMinibatchFeatureStore::update(
    const vector<int>& context, const VectorReal& values) {
  for (int feature_context_id: extractor->getFeatureContextIds(context)) {
    update(feature_context_id, values);
  }
}

void SparseMinibatchFeatureStore::update(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<SparseMinibatchFeatureStore> store = cast(base_store);
  for (const auto& entry: store->featureWeights) {
    update(entry.first, entry.second);
  }
}

void SparseMinibatchFeatureStore::clear() {
  featureWeights.clear();
}

size_t SparseMinibatchFeatureStore::size() const {
  return featureWeights.size();
}

void SparseMinibatchFeatureStore::hintFeatureIndex(
    int feature_context_id, int feature_index) {
  auto it = featureWeights.find(feature_context_id);
  if (it != featureWeights.end()) {
    it->second.coeffRef(feature_index) = 0;
  } else {
    SparseVectorReal weights(vectorMaxSize);
    weights.coeffRef(feature_index) = 0;
    featureWeights.insert(make_pair(feature_context_id, weights));
  }
}

bool SparseMinibatchFeatureStore::operator==(const SparseMinibatchFeatureStore& store) const {
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

SparseMinibatchFeatureStore::~SparseMinibatchFeatureStore() {}

void SparseMinibatchFeatureStore::update(
    int feature_context_id, const VectorReal& values) {
  SparseVectorReal& weights = featureWeights.at(feature_context_id);
  VectorReal pattern = weights.unaryExpr(CwiseSetValueOp<Real>(1));
  VectorReal product = (values.array() * pattern.array()).matrix();
  weights += product.sparseView();
}

void SparseMinibatchFeatureStore::update(
    int feature_context_id, const SparseVectorReal& values) {
  auto it = featureWeights.find(feature_context_id);
  // All features involved in gradient updates must be defined since the
  // construction of the sparse feature store.
  assert(it != featureWeights.end());
  it->second += values;
}

Real SparseMinibatchFeatureStore::getFeature(const pair<int, int>& index) const {
  auto it = featureWeights.find(index.first);
  return it == featureWeights.end() ? 0 : VectorReal(it->second)(index.second);
}

boost::shared_ptr<SparseMinibatchFeatureStore> SparseMinibatchFeatureStore::cast(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<SparseMinibatchFeatureStore> store =
      dynamic_pointer_cast<SparseMinibatchFeatureStore>(base_store);
  assert(store != nullptr);
  return store;
}

} // namespace oxlm
