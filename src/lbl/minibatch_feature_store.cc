#include "lbl/minibatch_feature_store.h"

namespace oxlm {

MinibatchFeatureStore::MinibatchFeatureStore(
    int vector_size, int hash_space, int feature_context_size,
    const boost::shared_ptr<FeatureContextHasher>& hasher,
    const boost::shared_ptr<FeatureFilter>& filter)
    : vectorSize(vector_size), hashSpace(hash_space),
      generator(feature_context_size), hasher(hasher), filter(filter) {}

VectorReal MinibatchFeatureStore::get(
    const vector<int>& context) const {
  VectorReal result = VectorReal::Zero(vectorSize);
  for (Hash context_hash: generator.getFeatureContextHashes(context)) {
    context_hash = hasher->getKey(context_hash);
    for (int i: filter->getIndexes(context_hash)) {
      auto it = featureWeights.find((context_hash + i) % hashSpace);
      if (it != featureWeights.end()) {
        result(i) += it->second;
      }
    }
  }

  return result;
}

void MinibatchFeatureStore::updateValue(
    int feature_index, const vector<int>& context, Real value) {
  for (Hash context_hash: generator.getFeatureContextHashes(context)) {
    context_hash = hasher->getKey(context_hash);
    if (filter->hasIndex(context_hash, feature_index)) {
      featureWeights[(context_hash + feature_index) % hashSpace] += value;
    }
  }
}

void MinibatchFeatureStore::update(
    const vector<int>& context, const VectorReal& values) {
  for (Hash context_hash: generator.getFeatureContextHashes(context)) {
    context_hash = hasher->getKey(context_hash);
    for (int i: filter->getIndexes(context_hash)) {
      featureWeights[(context_hash + i) % hashSpace] += values(i);
    }
  }
}

void MinibatchFeatureStore::update(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<MinibatchFeatureStore> store = cast(base_store);

  for (const auto& entry: store->featureWeights) {
    featureWeights[entry.first] += entry.second;
  }
}

size_t MinibatchFeatureStore::size() const {
  return featureWeights.size();
}

void MinibatchFeatureStore::clear() {
  featureWeights.clear();
}

Real MinibatchFeatureStore::getFeature(const pair<int, int>& index) const {
  auto it = featureWeights.find(index.first);
  return it == featureWeights.end() ? 0 : it->second;
}

boost::shared_ptr<MinibatchFeatureStore> MinibatchFeatureStore::cast(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<MinibatchFeatureStore> store =
      dynamic_pointer_cast<MinibatchFeatureStore>(base_store);
  assert(store != nullptr);
  return store;
}

MinibatchFeatureStore::~MinibatchFeatureStore() {}

} // namespace oxlm
