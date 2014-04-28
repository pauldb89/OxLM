#include "lbl/circular_feature_batch.h"

#include "lbl/operators.h"

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

Real CircularFeatureBatch::l2Objective(Real sigma) const {
  return sigma * (firstBatch.cwiseAbs2().sum() + secondBatch.cwiseAbs2().sum());
}

void CircularFeatureBatch::update(const VectorReal& values) {
  firstBatch += values.segment(0, firstBatch.size());
  secondBatch += values.segment(firstBatch.size(), secondBatch.size());
}

void CircularFeatureBatch::updateSquared(const VectorReal& values) {
  firstBatch += values.segment(0, firstBatch.size()).cwiseAbs2();
  secondBatch +=
      values.segment(firstBatch.size(), secondBatch.size()).cwiseAbs2();
}

void CircularFeatureBatch::updateAdaGrad(
    const VectorReal& gradient, const VectorReal& adagrad, Real step_size) {
  CwiseAdagradUpdateOp<Real> op(step_size);
  firstBatch -= gradient.segment(0, firstBatch.size())
      .binaryExpr(adagrad.segment(0, firstBatch.size()), op);
  secondBatch -= gradient.segment(firstBatch.size(), secondBatch.size())
      .binaryExpr(adagrad.segment(firstBatch.size(), secondBatch.size()), op);
}

void CircularFeatureBatch::l2Update(Real sigma) {
  firstBatch -= sigma * firstBatch;
  secondBatch -= sigma * secondBatch;
}

void CircularFeatureBatch::setZero() {
  firstBatch = VectorReal::Zero(firstBatch.size());
  secondBatch = VectorReal::Zero(secondBatch.size());
}

CircularFeatureBatch::~CircularFeatureBatch() {}

} // namespace oxlm
