#include "lbl/collision_global_feature_store.h"

namespace oxlm {

CollisionGlobalFeatureStore::CollisionGlobalFeatureStore()
    : CollisionStore() {}

CollisionGlobalFeatureStore::CollisionGlobalFeatureStore(
    int vector_size, int hash_space, int feature_context_size)
    : CollisionStore(vector_size, hash_space, feature_context_size) {}

void CollisionGlobalFeatureStore::l2GradientUpdate(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store, Real sigma) {
  boost::shared_ptr<CollisionMinibatchFeatureStore> store =
      CollisionMinibatchFeatureStore::cast(base_store);

  for (const auto& batch: store->observedBatches) {
    getBatch(batch)->l2Update(sigma);
  }
}

Real CollisionGlobalFeatureStore::l2Objective(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store,
    Real sigma) const {
  boost::shared_ptr<CollisionMinibatchFeatureStore> store =
      CollisionMinibatchFeatureStore::cast(base_store);

  Real result = 0;
  for (const auto& batch: store->observedBatches) {
    result += getBatch(batch)->l2Objective(sigma);
  }
  return result;
}

void CollisionGlobalFeatureStore::updateSquared(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<CollisionMinibatchFeatureStore> store =
      CollisionMinibatchFeatureStore::cast(base_store);

  for (const auto& batch: store->observedBatches) {
    getBatch(batch)->updateSquared(store->getBatch(batch)->values());
  }
}

void CollisionGlobalFeatureStore::updateAdaGrad(
    const boost::shared_ptr<MinibatchFeatureStore>& base_gradient_store,
    const boost::shared_ptr<GlobalFeatureStore>& base_adagrad_store,
    Real step_size) {
  boost::shared_ptr<CollisionMinibatchFeatureStore> gradient_store =
      CollisionMinibatchFeatureStore::cast(base_gradient_store);
  boost::shared_ptr<CollisionGlobalFeatureStore> adagrad_store =
      CollisionGlobalFeatureStore::cast(base_adagrad_store);

  for (const auto& batch: gradient_store->observedBatches) {
    getBatch(batch)->updateAdaGrad(
        gradient_store->getBatch(batch)->values(),
        adagrad_store->getBatch(batch)->values(),
        step_size);
  }
}

size_t CollisionGlobalFeatureStore::size() const {
  return hashSpace;
}

boost::shared_ptr<CollisionGlobalFeatureStore> CollisionGlobalFeatureStore::cast(
    const boost::shared_ptr<GlobalFeatureStore>& base_store) {
  boost::shared_ptr<CollisionGlobalFeatureStore> store =
      dynamic_pointer_cast<CollisionGlobalFeatureStore>(base_store);
  assert(store != nullptr);
  return store;
}

bool CollisionGlobalFeatureStore::operator==(
    const CollisionGlobalFeatureStore& other) const {
  if (vectorSize != other.vectorSize ||
      hashSpace != other.hashSpace ||
      !(keyer == other.keyer)) {
    return false;
  }

  for (int i = 0; i < hashSpace; ++i) {
    if (featureWeights[i] != other.featureWeights[i]) {
      return false;
    }
  }

  return true;
}

CollisionGlobalFeatureStore::~CollisionGlobalFeatureStore() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::CollisionGlobalFeatureStore)
