#pragma once

#include <set>

#include "lbl/collision_store.h"
#include "lbl/feature_batch.h"
#include "lbl/minibatch_feature_store.h"

namespace oxlm {

class CollisionMinibatchFeatureStore
    : public CollisionStore, public MinibatchFeatureStore {
 public:
  CollisionMinibatchFeatureStore(
      int vector_size, int hash_space, int feature_context_size);

  virtual void update(const vector<int>& context, const VectorReal& values);

  virtual void update(const boost::shared_ptr<MinibatchFeatureStore>& store);

  virtual size_t size() const;

  virtual void clear();

  static boost::shared_ptr<CollisionMinibatchFeatureStore> cast(
      const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store);

  virtual ~CollisionMinibatchFeatureStore();

 private:
  friend class CollisionGlobalFeatureStore;

  void markBatch(pair<int, int> batch);

  void markBatch(int start_key, int length);

  // <start_key, length> for each non-zero batch in featureWeights.
  set<pair<int, int>> observedBatches;
};

} // namespace oxlm
