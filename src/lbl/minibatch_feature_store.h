#pragma once

#include <set>

#include "lbl/feature_context_generator.h"
#include "lbl/feature_context_hasher.h"
#include "lbl/feature_filter.h"

namespace oxlm {

class MinibatchFeatureStore {
 public:
  MinibatchFeatureStore(
      int vector_size, int hash_space, int feature_context_size,
      const boost::shared_ptr<FeatureContextHasher>& hasher,
      const boost::shared_ptr<FeatureFilter>& filter);

  virtual VectorReal get(const vector<int>& context) const;

  virtual void updateValue(
      int feature_index, const vector<int>& context, Real value);

  virtual void update(const vector<int>& context, const VectorReal& values);

  virtual void update(const boost::shared_ptr<MinibatchFeatureStore>& store);

  virtual size_t size() const;

  virtual void clear();

  virtual Real getFeature(const pair<int, int>& index) const;

  static boost::shared_ptr<MinibatchFeatureStore> cast(
      const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store);

  virtual ~MinibatchFeatureStore();

 private:
  friend class GlobalFeatureStore;

  int vectorSize, hashSpace;
  FeatureContextGenerator generator;
  boost::shared_ptr<FeatureContextHasher> hasher;
  boost::shared_ptr<FeatureFilter> filter;
  unordered_map<int, Real> featureWeights;
};

} // namespace oxlm
