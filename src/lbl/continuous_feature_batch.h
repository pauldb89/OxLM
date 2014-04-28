#pragma once

#include "lbl/feature_batch.h"

namespace oxlm {

class ContinuousFeatureBatch : public FeatureBatch {
 public:
  ContinuousFeatureBatch(Real* feature_weights, int start_key, int vector_size);

  virtual VectorReal values() const;

  virtual Real l2Objective(Real sigma) const;

  virtual void update(const VectorReal& values);

  virtual void updateSquared(const VectorReal& values);

  virtual void updateAdaGrad(
      const VectorReal& gradient, const VectorReal& adagrad, Real step_size);

  virtual void l2Update(Real sigma);

  virtual void setZero();

  virtual ~ContinuousFeatureBatch();

 private:
  VectorRealMap weights;
};

} // namespace oxlm
