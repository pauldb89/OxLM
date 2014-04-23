#pragma once

#include "lbl/feature_store.h"

namespace oxlm {

class MinibatchFeatureStore : virtual public FeatureStore {
 public:
  virtual void update(
      const vector<int>& feature_context_ids,
      const VectorReal& values) = 0;

  virtual void update(
      const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store) = 0;

  virtual void clear() = 0;
};

}
