#include "lbl/continuous_feature_batch.h"

namespace oxlm {

ContinuousFeatureBatch::ContinuousFeatureBatch(
    Real* feature_weights, int start_key, int vector_size)
    : weights(feature_weights + start_key, vector_size) {}

VectorReal ContinuousFeatureBatch::values() const {
  return weights;
}

void ContinuousFeatureBatch::add(const VectorReal& values) {
  weights += values;
}

void ContinuousFeatureBatch::setZero() {
  weights = VectorReal::Zero(weights.size());
}

} // namespace oxlm
