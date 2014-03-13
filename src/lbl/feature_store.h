#pragma once

#include "lbl/feature_context.h"
#include "lbl/utils.h"

namespace oxlm {

class FeatureStore {
 public:
  virtual VectorReal get(
      const vector<FeatureContext>& feature_contexts) const = 0;

  virtual void update(
      const vector<FeatureContext>& feature_contexts,
      const VectorReal& values) = 0;
};

} // namespace oxlm
