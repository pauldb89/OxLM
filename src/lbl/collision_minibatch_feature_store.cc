#include "lbl/collision_minibatch_feature_store.h"

namespace oxlm {

CollisionMinibatchFeatureStore::CollisionMinibatchFeatureStore(
    int vector_size, int hash_space, int feature_context_size,
    const boost::shared_ptr<FeatureContextHasher>& hasher,
    const boost::shared_ptr<FeatureFilter>& filter)
    : vectorSize(vector_size), hashSpace(hash_space),
      generator(feature_context_size), hasher(hasher), filter(filter) {}

VectorReal CollisionMinibatchFeatureStore::get(
    const vector<int>& context) const {
  VectorReal result = VectorReal::Zero(vectorSize);
  for (const auto& feature_context: generator.getFeatureContexts(context)) {
    int key = hasher->getKey(feature_context);
    for (int i: filter->getIndexes(feature_context)) {
      auto it = featureWeights.find((key + i) % hashSpace);
      if (it != featureWeights.end()) {
        result(i) += it->second;
      }
    }
  }

  return result;
}

void CollisionMinibatchFeatureStore::update(
    const vector<int>& context, const VectorReal& values) {
  for (const auto& feature_context: generator.getFeatureContexts(context)) {
    int key = hasher->getKey(feature_context);
    for (int i: filter->getIndexes(feature_context)) {
      featureWeights[(key + i) % hashSpace] += values(i);
    }
  }
}

void CollisionMinibatchFeatureStore::update(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<CollisionMinibatchFeatureStore> store = cast(base_store);

  for (const auto& entry: store->featureWeights) {
    featureWeights[entry.first] += entry.second;
  }
}

size_t CollisionMinibatchFeatureStore::size() const {
  return featureWeights.size();
}

void CollisionMinibatchFeatureStore::clear() {
  featureWeights.clear();
}

Real CollisionMinibatchFeatureStore::getFeature(const pair<int, int>& index) const {
  auto it = featureWeights.find(index.first);
  return it == featureWeights.end() ? 0 : it->second;
}

boost::shared_ptr<CollisionMinibatchFeatureStore> CollisionMinibatchFeatureStore::cast(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<CollisionMinibatchFeatureStore> store =
      dynamic_pointer_cast<CollisionMinibatchFeatureStore>(base_store);
  assert(store != nullptr);
  return store;
}

CollisionMinibatchFeatureStore::~CollisionMinibatchFeatureStore() {}

} // namespace oxlm
