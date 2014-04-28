#pragma once

#include "lbl/utils.h"

namespace oxlm {

class FeatureBatch {
 public:
  virtual VectorReal values() const = 0;

  virtual Real l2Objective(Real sigma) const = 0;

  virtual void update(const VectorReal& values) = 0;

  virtual void updateSquared(const VectorReal& values) = 0;

  virtual void updateAdaGrad(
      const VectorReal& gradient,
      const VectorReal& adagrad,
      Real step_size) = 0;

  virtual void l2Update(Real sigma) = 0;

  virtual void setZero() = 0;

  virtual ~FeatureBatch();
};

} // namespace oxlm
