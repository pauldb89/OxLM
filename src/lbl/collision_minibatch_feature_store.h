#pragma once

#include <set>

#include "lbl/feature_context_generator.h"
#include "lbl/feature_context_keyer.h"
#include "lbl/feature_filter.h"
#include "lbl/minibatch_feature_store.h"

namespace oxlm {

class CollisionMinibatchFeatureStore : public MinibatchFeatureStore {
 public:
  CollisionMinibatchFeatureStore(
      int vector_size, int hash_space, int feature_context_size,
      const boost::shared_ptr<FeatureFilter>& fiter);

  virtual VectorReal get(const vector<int>& context) const;

  virtual void update(const vector<int>& context, const VectorReal& values);

  virtual void update(const boost::shared_ptr<MinibatchFeatureStore>& store);

  virtual size_t size() const;

  virtual void clear();

  static boost::shared_ptr<CollisionMinibatchFeatureStore> cast(
      const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store);

  virtual ~CollisionMinibatchFeatureStore();

 private:
  friend class CollisionGlobalFeatureStore;

  int vectorSize, hashSpace;
  FeatureContextGenerator generator;
  FeatureContextKeyer keyer;
  boost::shared_ptr<FeatureFilter> filter;
  unordered_map<int, Real> featureWeights;
};

} // namespace oxlm