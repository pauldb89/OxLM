#pragma once

#include "lbl/feature_store.h"

namespace oxlm {

class MinibatchFeatureStore : virtual public FeatureStore {
 public:
  virtual void update(
      const vector<int>& context, const VectorReal& values) = 0;

  virtual void update(
      const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store) = 0;

  virtual void clear() = 0;

  virtual Real getFeature(const pair<int, int>& index) const = 0;

  virtual ~MinibatchFeatureStore();
};

}
