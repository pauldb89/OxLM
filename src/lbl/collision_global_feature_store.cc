#include "lbl/collision_global_feature_store.h"

#include "lbl/operators.h"

namespace oxlm {

CollisionGlobalFeatureStore::CollisionGlobalFeatureStore() {}

CollisionGlobalFeatureStore::CollisionGlobalFeatureStore(
    const CollisionGlobalFeatureStore& other) {
  deepCopy(other);
}

CollisionGlobalFeatureStore::CollisionGlobalFeatureStore(
    int vector_size, int hash_space, int feature_context_size,
    const boost::shared_ptr<FeatureFilter>& filter)
    : vectorSize(vector_size), hashSpace(hash_space),
      generator(feature_context_size), keyer(hash_space), filter(filter) {
  assert(vectorSize <= hashSpace);
  featureWeights = new Real[hashSpace];
  VectorRealMap featureWeightsMap(featureWeights, hashSpace);
  featureWeightsMap = VectorReal::Zero(hashSpace);
}

VectorReal CollisionGlobalFeatureStore::get(const vector<int>& context) const {
  VectorReal result = VectorReal::Zero(vectorSize);
  for (const auto& feature_context: generator.getFeatureContexts(context)) {
    int key = keyer.getKey(feature_context);
    for (int i: filter->getIndexes(feature_context)) {
      result(i) += featureWeights[(key + i) % hashSpace];
    }
  }

  return result;
}

void CollisionGlobalFeatureStore::l2GradientUpdate(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store, Real sigma) {
  boost::shared_ptr<CollisionMinibatchFeatureStore> store =
      CollisionMinibatchFeatureStore::cast(base_store);

  for (const auto& entry: store->featureWeights) {
    featureWeights[entry.first] -= sigma * featureWeights[entry.first];
  }
}

Real CollisionGlobalFeatureStore::l2Objective(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store,
    Real sigma) const {
  boost::shared_ptr<CollisionMinibatchFeatureStore> store =
      CollisionMinibatchFeatureStore::cast(base_store);

  Real result = 0;
  for (const auto& entry: store->featureWeights) {
    result += featureWeights[entry.first] * featureWeights[entry.first];
  }
  return sigma * result;
}

void CollisionGlobalFeatureStore::updateSquared(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<CollisionMinibatchFeatureStore> store =
      CollisionMinibatchFeatureStore::cast(base_store);

  for (const auto& entry: store->featureWeights) {
    featureWeights[entry.first] += entry.second * entry.second;
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

  CwiseAdagradUpdateOp<Real> op(step_size);
  for (const auto& entry: gradient_store->featureWeights) {
    featureWeights[entry.first] -=
        op(entry.second, adagrad_store->featureWeights[entry.first]);
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

CollisionGlobalFeatureStore& CollisionGlobalFeatureStore::operator=(
    const CollisionGlobalFeatureStore& other) {
  deepCopy(other);
  return *this;
}

CollisionGlobalFeatureStore::~CollisionGlobalFeatureStore() {
  delete[] featureWeights;
}

void CollisionGlobalFeatureStore::deepCopy(
    const CollisionGlobalFeatureStore& other) {
  vectorSize = other.vectorSize;
  hashSpace = other.hashSpace;
  generator = other.generator;
  keyer = other.keyer;
  filter = other.filter;
  featureWeights = new Real[hashSpace];
  memcpy(featureWeights, other.featureWeights, hashSpace * sizeof(Real));
}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::CollisionGlobalFeatureStore)
