#include "lbl/collision_minibatch_feature_store.h"

namespace oxlm {

CollisionMinibatchFeatureStore::CollisionMinibatchFeatureStore(
    int vector_size, int hash_space, int feature_context_size)
    : CollisionStore(vector_size, hash_space, feature_context_size) {}

void CollisionMinibatchFeatureStore::update(
    const vector<int>& context, const VectorReal& values) {
  for (int key: getKeys(context)) {
    markBatch(key, vectorSize);
    getBatch(key, vectorSize)->update(values);
  }
}

void CollisionMinibatchFeatureStore::update(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<CollisionMinibatchFeatureStore> store = cast(base_store);

  for (const auto& batch: store->observedBatches) {
    markBatch(batch);
    getBatch(batch)->update(store->getBatch(batch)->values());
  }
}

size_t CollisionMinibatchFeatureStore::size() const {
  int non_zeros = 0;
  for (const auto& batch: observedBatches) {
    non_zeros += batch.second;
  }

  return non_zeros;
}

void CollisionMinibatchFeatureStore::clear() {
  for (const auto& batch: observedBatches) {
    getBatch(batch)->setZero();
  }
  observedBatches.clear();
}

boost::shared_ptr<CollisionMinibatchFeatureStore> CollisionMinibatchFeatureStore::cast(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<CollisionMinibatchFeatureStore> store =
      dynamic_pointer_cast<CollisionMinibatchFeatureStore>(base_store);
  assert(store != nullptr);
  return store;
}

CollisionMinibatchFeatureStore::~CollisionMinibatchFeatureStore() {}

void CollisionMinibatchFeatureStore::markBatch(pair<int, int> batch) {
  // Search if the current batch crosses with a batch located to the right.
  // A batch starting at the same position, but having a longer length is also
  // covered here.
  auto it = observedBatches.upper_bound(batch);
  if (it != observedBatches.end()) {
    if (batch.first + batch.second >= it->first) {
      batch.second = it->second + it->first - batch.first;
      observedBatches.erase(it);
    }
  }

  // Search if the current batch crosses with a batch located to the left.
  // A batch starting at the same position, but having a shorter length is also
  // covered here.
  it = observedBatches.upper_bound(batch);
  if (it != observedBatches.begin()) {
    --it;
    if (it->first + it->second > batch.first) {
      batch = make_pair(it->first, batch.second + batch.first - it->first);
      observedBatches.erase(it);
    }
  }

  observedBatches.insert(batch);
}

void CollisionMinibatchFeatureStore::markBatch(int start_key, int length) {
  markBatch(make_pair(start_key, length));
}

} // namespace oxlm
