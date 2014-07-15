#include "lbl/collision_global_feature_store.h"

#include "lbl/operators.h"

namespace oxlm {

CollisionGlobalFeatureStore::CollisionGlobalFeatureStore() {}

CollisionGlobalFeatureStore::CollisionGlobalFeatureStore(
    int vector_size, int hash_space_size, int feature_context_size,
    const boost::shared_ptr<GlobalCollisionSpace>& space,
    const boost::shared_ptr<FeatureContextHasher>& hasher,
    const boost::shared_ptr<FeatureFilter>& filter)
    : vectorSize(vector_size), hashSpaceSize(hash_space_size),
      generator(feature_context_size), hasher(hasher), filter(filter),
      space(space) {}

VectorReal CollisionGlobalFeatureStore::get(const vector<int>& context) const {
  VectorReal result = VectorReal::Zero(vectorSize);
  for (const auto& feature_context: generator.getFeatureContexts(context)) {
    int key = hasher->getKey(feature_context);
    for (int i: filter->getIndexes(feature_context)) {
      result(i) += space->featureWeights[(key + i) % hashSpaceSize];
    }
  }

  return result;
}

void CollisionGlobalFeatureStore::l2GradientUpdate(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store, Real sigma) {
  boost::shared_ptr<CollisionMinibatchFeatureStore> store =
      CollisionMinibatchFeatureStore::cast(base_store);

  for (const auto& entry: store->featureWeights) {
    space->featureWeights[entry.first] -= sigma * space->featureWeights[entry.first];
  }
}

Real CollisionGlobalFeatureStore::l2Objective(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store,
    Real sigma) const {
  boost::shared_ptr<CollisionMinibatchFeatureStore> store =
      CollisionMinibatchFeatureStore::cast(base_store);

  Real result = 0;
  for (const auto& entry: store->featureWeights) {
    result += space->featureWeights[entry.first] * space->featureWeights[entry.first];
  }

  return sigma * result;
}

void CollisionGlobalFeatureStore::updateSquared(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<CollisionMinibatchFeatureStore> store =
      CollisionMinibatchFeatureStore::cast(base_store);

  for (const auto& entry: store->featureWeights) {
    space->featureWeights[entry.first] += entry.second * entry.second;
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
    space->featureWeights[entry.first] -=
        op(entry.second, adagrad_store->space->featureWeights[entry.first]);
  }
}

size_t CollisionGlobalFeatureStore::size() const {
  return hashSpaceSize;
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
  return vectorSize == other.vectorSize
      && hashSpaceSize == other.hashSpaceSize
      && *space == *other.space;
}

bool CollisionGlobalFeatureStore::operator==(
    const boost::shared_ptr<GlobalFeatureStore>& other) const {
  return operator==(*cast(other));
}

vector<pair<int, int>> CollisionGlobalFeatureStore::getFeatureIndexes() const {
  vector<pair<int, int>> feature_indexes;

  for (size_t i = 0; i < hashSpaceSize; ++i) {
    feature_indexes.push_back(make_pair(i, 0));
  }

  return feature_indexes;
}

void CollisionGlobalFeatureStore::updateFeature(
    const pair<int, int>& index, Real value) {
  space->featureWeights[index.first] += value;
}

CollisionGlobalFeatureStore::~CollisionGlobalFeatureStore() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::CollisionGlobalFeatureStore)
