#pragma once

#include <set>

#include "lbl/feature_batch.h"
#include "lbl/feature_context_keyer.h"
#include "lbl/minibatch_feature_store.h"

namespace oxlm {

class CollisionMinibatchFeatureStore : public MinibatchFeatureStore {
 public:
  CollisionMinibatchFeatureStore(
      int vector_size, int hash_space, int feature_context_size);

  virtual VectorReal get(const vector<int>& context) const;

  virtual void update(const vector<int>& context, const VectorReal& values);

  virtual void update(const boost::shared_ptr<MinibatchFeatureStore>& store);

  virtual size_t size() const;

  virtual void clear();

  static boost::shared_ptr<CollisionMinibatchFeatureStore> cast(
      const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store);

 private:
  vector<int> getKeys(const vector<int>& context) const;

  boost::shared_ptr<FeatureBatch> getBatch(const pair<int, int>& batch) const;

  boost::shared_ptr<FeatureBatch> getBatch(int key, int length) const;

  void markBatch(pair<int, int> batch);

  void markBatch(int start_key, int length);

  int vectorSize, hashSpace;
  FeatureContextKeyer keyer;
  // <start_key, length> for each non-zero batch in featureWeights.
  set<pair<int, int>> observedBatches;
  Real* featureWeights;
};

} // namespace oxlm
