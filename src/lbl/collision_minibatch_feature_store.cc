#include "lbl/collision_minibatch_feature_store.h"

#include <boost/make_shared.hpp>

#include "lbl/circular_feature_batch.h"
#include "lbl/continuous_feature_batch.h"

namespace oxlm {

CollisionMinibatchFeatureStore::CollisionMinibatchFeatureStore(
    int vector_size, int hash_space, int feature_context_size)
    : vectorSize(vector_size), hashSpace(hash_space),
      keyer(feature_context_size) {
  assert(vectorSize <= hashSpace);
  featureWeights = new Real[hashSpace];
  VectorRealMap weights(featureWeights, hashSpace);
  weights.setZero();
}

VectorReal CollisionMinibatchFeatureStore::get(
    const vector<int>& context) const {
  VectorReal result = VectorReal::Zero(vectorSize);
  for (int key: getKeys(context)) {
    result += getBatch(key, vectorSize)->values();
  }

  return result;
}

void CollisionMinibatchFeatureStore::update(
    const vector<int>& context, const VectorReal& values) {
  for (int key: getKeys(context)) {
    markBatch(key, vectorSize);
    getBatch(key, vectorSize)->add(values);
  }
}

void CollisionMinibatchFeatureStore::update(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<CollisionMinibatchFeatureStore> store = cast(base_store);

  for (const auto& batch: store->observedBatches) {
    markBatch(batch);
    getBatch(batch)->add(store->getBatch(batch)->values());
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

vector<int> CollisionMinibatchFeatureStore::getKeys(
    const vector<int>& context) const {
  vector<int> keys;
  for (size_t raw_key: keyer.getKeys(context)) {
    keys.push_back(raw_key % hashSpace);
  }
  return keys;
}

boost::shared_ptr<FeatureBatch> CollisionMinibatchFeatureStore::getBatch(
    const pair<int, int>& batch) const {
  if (batch.first + batch.second < hashSpace) {
    return boost::make_shared<ContinuousFeatureBatch>(
        featureWeights, batch.first, batch.second);
  } else {
    return boost::make_shared<CircularFeatureBatch>(
        featureWeights, batch.first, batch.second, hashSpace);
  }
}

boost::shared_ptr<FeatureBatch> CollisionMinibatchFeatureStore::getBatch(
    int start_key, int length) const {
  return getBatch(make_pair(start_key, length));
}

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
