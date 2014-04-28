#include "lbl/collision_store.h"

#include <boost/make_shared.hpp>

#include "lbl/circular_feature_batch.h"
#include "lbl/continuous_feature_batch.h"

namespace oxlm {

CollisionStore::CollisionStore()
    : vectorSize(0), hashSpace(0), featureWeights(0) {}

CollisionStore::CollisionStore(const CollisionStore& other) {
  deepCopy(other);
}

CollisionStore::CollisionStore(
    int vector_size, int hash_space, int feature_context_size)
    : vectorSize(vector_size), hashSpace(hash_space),
      keyer(feature_context_size) {
  assert(vectorSize <= hashSpace);
  featureWeights = new Real[hashSpace];
  VectorRealMap featureWeightsMap(featureWeights, hashSpace);
  featureWeightsMap = VectorReal::Zero(hashSpace);
}

VectorReal CollisionStore::get(
    const vector<int>& context) const {
  VectorReal result = VectorReal::Zero(vectorSize);
  for (int key: getKeys(context)) {
    result += getBatch(key, vectorSize)->values();
  }

  return result;
}

CollisionStore& CollisionStore::operator=(const CollisionStore& other) {
  deepCopy(other);
  return *this;
}

CollisionStore::~CollisionStore() {
  delete[] featureWeights;
}

vector<int> CollisionStore::getKeys(const vector<int>& context) const {
  vector<int> keys;
  for (size_t raw_key: keyer.getKeys(context)) {
    keys.push_back(raw_key % hashSpace);
  }
  return keys;
}

boost::shared_ptr<FeatureBatch> CollisionStore::getBatch(
    const pair<int, int>& batch) const {
  if (batch.first + batch.second < hashSpace) {
    return boost::make_shared<ContinuousFeatureBatch>(
        featureWeights, batch.first, batch.second);
  } else {
    return boost::make_shared<CircularFeatureBatch>(
        featureWeights, batch.first, batch.second, hashSpace);
  }
}

boost::shared_ptr<FeatureBatch> CollisionStore::getBatch(
    int start_key, int length) const {
  return getBatch(make_pair(start_key, length));
}

void CollisionStore::deepCopy(const CollisionStore& other) {
  vectorSize = other.vectorSize;
  hashSpace = other.hashSpace;
  keyer = other.keyer;
  featureWeights = new Real[hashSpace];
  memcpy(featureWeights, other.featureWeights, hashSpace * sizeof(Real));
}

} // namespace oxlm
