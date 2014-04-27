#pragma once

#include "lbl/feature_batch.h"

namespace oxlm {

class CircularFeatureBatch : public FeatureBatch {
 public:
  CircularFeatureBatch(
      Real* feature_weights, int start_key, int vector_size, int hash_space);

  virtual VectorReal values() const;

  virtual void add(const VectorReal& values);

  virtual void setZero();

 private:
  int vectorSize;
  // We consider the first batch to be the one located at the end of the vector
  // of feature weights. The second batch contains the remaining features and
  // starts at the beginning of the vector.
  VectorRealMap firstBatch, secondBatch;
};

} // namespace oxlm
