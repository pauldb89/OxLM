#include "lbl/global_feature_store.h"

#include "lbl/operators.h"

namespace oxlm {

GlobalFeatureStore::GlobalFeatureStore() {}

GlobalFeatureStore::GlobalFeatureStore(
    int vector_size, int hash_space_size, int feature_context_size,
    const boost::shared_ptr<GlobalCollisionSpace>& space,
    const boost::shared_ptr<FeatureContextHasher>& hasher,
    const boost::shared_ptr<FeatureFilter>& filter)
    : vectorSize(vector_size), hashSpaceSize(hash_space_size),
      generator(feature_context_size), hasher(hasher), filter(filter),
      space(space) {}

VectorReal GlobalFeatureStore::get(const vector<int>& context) const {
  VectorReal result = VectorReal::Zero(vectorSize);
  for (Hash context_hash: generator.getFeatureContextHashes(context)) {
    context_hash = hasher->getKey(context_hash);
    for (int i: filter->getIndexes(context_hash)) {
      result(i) += space->featureWeights[(context_hash + i) % hashSpaceSize];
    }
  }

  return result;
}

Real GlobalFeatureStore::getValue(
    int feature_index, const vector<int>& context) const {
  Real result = 0;
  for (Hash context_hash: generator.getFeatureContextHashes(context)) {
    context_hash = hasher->getKey(context_hash);
    if (filter->hasIndex(context_hash, feature_index)) {
      result += space->featureWeights[(context_hash + feature_index) % hashSpaceSize];
    }
  }

  return result;
}

void GlobalFeatureStore::l2GradientUpdate(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store, Real sigma) {
  boost::shared_ptr<MinibatchFeatureStore> store =
      MinibatchFeatureStore::cast(base_store);

  for (const auto& entry: store->featureWeights) {
    space->featureWeights[entry.first] -= sigma * space->featureWeights[entry.first];
  }
}

Real GlobalFeatureStore::l2Objective(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store,
    Real sigma) const {
  boost::shared_ptr<MinibatchFeatureStore> store =
      MinibatchFeatureStore::cast(base_store);

  Real result = 0;
  for (const auto& entry: store->featureWeights) {
    result += space->featureWeights[entry.first] * space->featureWeights[entry.first];
  }

  return sigma * result;
}

void GlobalFeatureStore::updateSquared(
    const boost::shared_ptr<MinibatchFeatureStore>& base_store) {
  boost::shared_ptr<MinibatchFeatureStore> store =
      MinibatchFeatureStore::cast(base_store);

  for (const auto& entry: store->featureWeights) {
    space->featureWeights[entry.first] += entry.second * entry.second;
  }
}

void GlobalFeatureStore::updateAdaGrad(
    const boost::shared_ptr<MinibatchFeatureStore>& base_gradient_store,
    const boost::shared_ptr<GlobalFeatureStore>& base_adagrad_store,
    Real step_size) {
  boost::shared_ptr<MinibatchFeatureStore> gradient_store =
      MinibatchFeatureStore::cast(base_gradient_store);
  boost::shared_ptr<GlobalFeatureStore> adagrad_store =
      GlobalFeatureStore::cast(base_adagrad_store);

  CwiseAdagradUpdateOp<Real> op(step_size);
  for (const auto& entry: gradient_store->featureWeights) {
    space->featureWeights[entry.first] -=
        op(entry.second, adagrad_store->space->featureWeights[entry.first]);
  }
}

size_t GlobalFeatureStore::size() const {
  return hashSpaceSize;
}

boost::shared_ptr<GlobalFeatureStore> GlobalFeatureStore::cast(
    const boost::shared_ptr<GlobalFeatureStore>& base_store) {
  boost::shared_ptr<GlobalFeatureStore> store =
      dynamic_pointer_cast<GlobalFeatureStore>(base_store);
  assert(store != nullptr);
  return store;
}

bool GlobalFeatureStore::operator==(
    const GlobalFeatureStore& other) const {
  return vectorSize == other.vectorSize
      && hashSpaceSize == other.hashSpaceSize
      && *space == *other.space;
}

bool GlobalFeatureStore::operator==(
    const boost::shared_ptr<GlobalFeatureStore>& other) const {
  return operator==(*cast(other));
}

vector<pair<int, int>> GlobalFeatureStore::getFeatureIndexes() const {
  vector<pair<int, int>> feature_indexes;

  for (size_t i = 0; i < hashSpaceSize; ++i) {
    feature_indexes.push_back(make_pair(i, 0));
  }

  return feature_indexes;
}

void GlobalFeatureStore::updateFeature(
    const pair<int, int>& index, Real value) {
  space->featureWeights[index.first] += value;
}

GlobalFeatureStore::~GlobalFeatureStore() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::GlobalFeatureStore)
