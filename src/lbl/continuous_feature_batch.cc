#include "lbl/continuous_feature_batch.h"

#include "lbl/operators.h"

namespace oxlm {

ContinuousFeatureBatch::ContinuousFeatureBatch(
    Real* feature_weights, int start_key, int vector_size)
    : weights(feature_weights + start_key, vector_size) {}

VectorReal ContinuousFeatureBatch::values() const {
  return weights;
}

Real ContinuousFeatureBatch::l2Objective(Real sigma) const {
  return sigma * weights.cwiseAbs2().sum();
}

void ContinuousFeatureBatch::update(const VectorReal& values) {
  weights += values;
}

void ContinuousFeatureBatch::updateSquared(const VectorReal& values) {
  weights += values.cwiseAbs2();
}

void ContinuousFeatureBatch::updateAdaGrad(
    const VectorReal& gradient, const VectorReal& adagrad, Real step_size) {
  weights -= gradient.binaryExpr(
      adagrad, CwiseAdagradUpdateOp<Real>(step_size));
}

void ContinuousFeatureBatch::l2Update(Real sigma) {
  weights -= sigma * weights;
}

void ContinuousFeatureBatch::setZero() {
  weights = VectorReal::Zero(weights.size());
}

ContinuousFeatureBatch::~ContinuousFeatureBatch() {}

} // namespace oxlm
