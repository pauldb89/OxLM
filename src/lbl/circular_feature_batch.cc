#include "lbl/circular_feature_batch.h"

namespace oxlm {

CircularFeatureBatch::CircularFeatureBatch(
    Real* feature_weights, int start_key, int vector_size, int hash_space)
    : vectorSize(vector_size),
      firstBatch(feature_weights + start_key, hash_space - start_key),
      secondBatch(feature_weights, vector_size - (hash_space - start_key)) {}

VectorReal CircularFeatureBatch::values() const {
  VectorReal ret(vectorSize);
  ret.segment(0, firstBatch.size()) = firstBatch;
  ret.segment(firstBatch.size(), secondBatch.size()) = secondBatch;
  return ret;
}

void CircularFeatureBatch::add(const VectorReal& values) {
  firstBatch += values.segment(0, firstBatch.size());
  secondBatch += values.segment(firstBatch.size(), secondBatch.size());
}

void CircularFeatureBatch::setZero() {
  firstBatch = VectorReal::Zero(firstBatch.size());
  secondBatch = VectorReal::Zero(secondBatch.size());
}

} // namespace oxlm
