#pragma once

#include "lbl/feature_batch.h"

namespace oxlm {

class ContinuousFeatureBatch : public FeatureBatch {
 public:
  ContinuousFeatureBatch(Real* feature_weights, int start_key, int vector_size);

  virtual VectorReal values() const;

  virtual void add(const VectorReal& values);

  virtual void setZero();

 private:
  VectorRealMap weights;
};

} // namespace oxlm
