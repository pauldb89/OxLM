#pragma once

#include "lbl/feature_batch.h"

namespace oxlm {

class CircularFeatureBatch : public FeatureBatch {
 public:
  CircularFeatureBatch(
      Real* feature_weights, int start_key, int vector_size, int hash_space);

  virtual VectorReal values() const;

  virtual Real l2Objective(Real sigma) const;

  virtual void update(const VectorReal& values);

  virtual void updateSquared(const VectorReal& values);

  virtual void updateAdaGrad(
      const VectorReal& gradient, const VectorReal& adagrad, Real step_size);

  virtual void l2Update(Real sigma);

  virtual void setZero();

  virtual ~CircularFeatureBatch();

 private:
  int vectorSize;
  // We consider the first batch to be the one located at the end of the vector
  // of feature weights. The second batch contains the remaining features and
  // starts at the beginning of the vector.
  VectorRealMap firstBatch, secondBatch;
};

} // namespace oxlm
